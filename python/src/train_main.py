import torch
import os
import traceback
import collections
import numpy as np
collections.Iterable = collections.abc.Iterable
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter
from src.utility import get_kernel, get_time, parse_model_name
from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2, MiniFASNetV2SE
from src.data_io.dataset_loader import get_train_loader, get_val_loader
from src.utils.metrics import num_fn, num_fp, num_tn, num_tp, apcer, bpcer, acer, tpr, fpr
from src.utils.logs import *

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}

class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.board_loss_every_val = conf.board_loss_every_val
        self.save_every = conf.save_every
        self.device = conf.device
        self.step_train = 0
        self.step_val = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)
        self.val_loader = get_val_loader(self.conf)

    def train_model(self):
        self._init_model_param()    
        self._train_stage()


    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                lr=self.conf.lr,
                                weight_decay=5e-4,
                                momentum=self.conf.momentum)
        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)
    
    
    def _train_stage(self):
        self.model.train()
        metrics_train = Average()
        metrics_val = Average()
        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                writer = write_tensorboardX(self.conf.log_path)
                is_first = False
            print('epoch {} started'.format(e))
            print("lr: ", self.schedule_lr.get_lr())
            
            try:
                for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                    imgs = [sample, ft_sample]
                    labels = target
                    labels = torch.Tensor(labels)
                    loss, acc, apcer, bpcer, acer, loss_cls = self._train_batch_data(imgs, labels)
                    metrics_train.update(loss, loss_cls, acc, apcer, bpcer, acer)

                    self.step_train += 1

                    if self.step_train % self.board_loss_every == 0 and self.step_train != 0:
                        # evaluate data val
                        for val_sample, ft_val_sample, val_target in tqdm(iter(self.val_loader)):
                            imgs_val = [val_sample, ft_val_sample]
                            labels_val = val_target
                            labels_val = torch.Tensor(labels_val)
                            loss_val, acc_val, apcer_val, bpcer_val, acer_val, loss_cls_val = self._val_batch_data(imgs_val, labels_val)
                            metrics_val.update(loss_val, loss_cls_val, acc_val, apcer_val, bpcer_val, acer_val)
                            self.step_val += 1
                            if self.step_val % self.board_loss_every_val == 0 and self.step_val != 0:
                                write_results(writer, metrics_val, self.board_loss_every_val, self.step_val, 'val') # val
                                metrics_val.reset()
                                
                        write_results(writer, metrics_train, self.board_loss_every, self.step_train, 'train') # train 
                        
                        # Evaluate dataset val
                        metrics_train.reset()
                    if self.step_train % self.save_every == 0 and self.step_train != 0:
                        time_stamp = get_time()
                        self._save_state(time_stamp, extra=self.conf.job_name)
                self.schedule_lr.step()    
            
            except Exception as err:
                print(traceback.print_exc())
        time_stamp = get_time()
        self._save_state(time_stamp, extra=self.conf.job_name)
        writer.close()

    # Train each batch size
    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        # embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))
        embeddings = self.model.forward(imgs[0].to(self.conf.device))
        
        loss_cls = self.cls_criterion(embeddings, labels)
        # loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))
        # loss = 0.5*loss_cls + 0.5*loss_fea
        loss = loss_cls
        acc = self._get_accuracy(embeddings, labels)[0]
        apcer, bpcer, acer = self._get_metrics(embeddings, labels)
        loss.backward()
        self.optimizer.step()
        # return loss.item(), acc, apcer, bpcer, acer, loss_cls.item(), loss_fea.item()
        return loss.item(), acc, apcer, bpcer, acer, loss_cls.item()
        

    # Val each batch size
    def _val_batch_data(self, imgs, labels):
        # self.model.eval()
        labels = labels.to(self.conf.device)
        embeddings = self.model.forward(imgs[0].to(self.conf.device))
        # embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))
        
        loss_cls = self.cls_criterion(embeddings, labels)
        # loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))
        # loss = 0.5*loss_cls + 0.5*loss_fea
        loss = loss_cls
        acc = self._get_accuracy(embeddings, labels)[0]
        apcer, bpcer, acer = self._get_metrics(embeddings, labels)
        # return loss.item(), acc, apcer, bpcer, acer, loss_cls.item(), loss_fea.item()
        return loss.item(), acc, apcer, bpcer, acer, loss_cls.item()

        

    def _define_network(self):
        # Default use
        param = {
            'num_classes': self.conf.num_classes, # Num classes
            'img_channel': self.conf.input_channel, # Input
            'embedding_size': self.conf.embedding_size, # Size vector embedding
            'conv6_kernel': self.conf.kernel_size, # Kernel size
            # 'use_pretrained': self.conf.use_pretrained # Use pre-traiened model
        }

        model = MiniFASNetV2(**param).to(self.conf.device)
        model.load_state_dict(torch.load(self.conf.use_pretrained))
        model = torch.nn.DataParallel(model, self.conf.devices) # Parallel
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret
    
    
    def _get_metrics(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target = target.view(1, -1).expand_as(pred)
        
        # num fp, tp, fn, tn
        num_false_positive = num_fp(target, pred)
        num_false_negative = num_fn(target, pred)
        num_true_positive = num_tp(target, pred)
        num_true_negative = num_tn(target, pred)
        APCER, BPCER = apcer(num_false_positive, num_true_negative), \
            bpcer(num_false_negative, num_true_positive)
        ACER = (APCER + BPCER) / 2
        return APCER, BPCER, ACER
            
        
    # save state
    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        torch.save(self.model.state_dict(), save_path + '/' +
                   ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step_train)))

