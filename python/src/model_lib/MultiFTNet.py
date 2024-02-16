from torch import nn
import torch
import os
import torch.nn.functional as F
from src.model_lib.MiniFASNet import MiniFASNetV1,MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.utility import parse_model_name



class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(self, img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5), use_pretrained=None, model_name=''):
        super(MultiFTNet, self).__init__()
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.backbone = model_name
        # define model
        if self.use_pretrained is not None:
            model_name = os.path.basename(self.use_pretrained)
            h_input, w_input, model_type, _ = parse_model_name(model_name)
            self.model = MODEL_MAPPING[model_type](embedding_size=embedding_size,
                                                img_channel=self.img_channel,
                                                num_classes=self.num_classes,
                                                conv6_kernel=conv6_kernel)
            self._load_weights()
        else:
            self.model = MODEL_MAPPING[self.backbone](embedding_size=embedding_size,
                                                conv6_kernel=conv6_kernel,
                                                num_classes=num_classes,
                                                img_channel=img_channel)
            self._initialize_weights()
        self.FTGenerator = FTGenerator(in_channels=128)
   

    def _initialize_weights(self):
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    
    def _load_weights(self):
        # load weights
        state_dict = torch.load(self.use_pretrained)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if 'FTGenerator' not in key:
                    key = key.replace('module.model.', '')
                    name_key = key
                    new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict) 

        # # After load model => Freeze model include conv1 -> conv last
        # for name, param in self.model.named_parameters():
        #     if 'conv' in name:
        #         param.requires_grad = False   
    
    def _get_model(self):
        return self.model      
    
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)
        if self.training:
            ft = self.FTGenerator(x)
            return cls, ft
        else:
            return cls

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE,
    'MultiFTNet': MultiFTNet
}


