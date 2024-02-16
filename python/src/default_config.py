"""
    Default config for training
"""

import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist, get_width_height, get_kernel


def get_default_config():
    conf = EasyDict()

    # ----------------------training---------------
    conf.lr = 1e-3
    conf.size = 80
    # [9, 13, 15]
    conf.milestones = [15, 120, 26]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 40
    conf.momentum = 0.9
    conf.batch_size = 128
    conf.val_batch_size = 32

    # model
    conf.num_classes = 2
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset
    conf.train_root_path = '/home/hoang/Documents/work/code/Data/Face_Anti_Spoofing/CelebA_Spoof'
    conf.val_root_path = '/home/hoang/Documents/work/code/Data/Face_Anti_Spoofing/CelebA_Spoof'
    # label
    conf.train_label_path = '/home/hoang/Documents/work/code/Data/Face_Anti_Spoofing/CelebA_Spoof/metas/intra_test/train_label.txt'
    conf.val_label_path = '/home/hoang/Documents/work/code/Data/Face_Anti_Spoofing/CelebA_Spoof/metas/intra_test/test_label.txt'
    
    # save file path
    conf.snapshot_dir_path = './saved_logs/snapshot'

    # log path
    conf.log_path = './saved_logs/jobs'
    # tensorboard
    conf.board_loss_every = 500
    conf.board_loss_every_val = 100
    # save model/iter
    conf.save_every = 2000

    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [80, 80]
    conf.kernel_size = (5, 5)
    print(conf.kernel_size)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"
    conf.use_pretrained = args.use_pretrained
    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(conf.log_path, job_name, current_time)
    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf
