import os
import onnx
import torch
import torch.nn.functional as F
import torch.nn as nn
from src.utility import get_kernel, parse_model_name
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MobileNetv3 import *
# from src.util import *
from src.utility import parse_model_name
from src.default_config import get_default_config
from onnxsim import simplify

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE,
    'MultiFTNet': MultiFTNet
}

class Network(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,x):
        x = x.view(1,3,80,80)
        out = self.model(x)
        out = F.softmax(out)
        return out
    
    
def pth2onnx(input, output):
    """_summary_
        convert model pth to model onnx
    Args:
        input (_type_): model pth
        output (_type_): model 
    """
    conf = get_default_config()
    model_name = os.path.basename(input)
    device = 'cpu'
    
    h_input, w_input, model_type, _ = parse_model_name(model_name)
    kernel_size = get_kernel(h_input, w_input,)
    print(model_name)
    # default use 
    param = {
            'num_classes': conf.num_classes, # Num classes
            'img_channel': conf.input_channel, # Input
            'embedding_size': conf.embedding_size, # Size vector embedding
            'conv6_kernel': kernel_size, # Kernel size
            'use_pretrained': input # Use pre-traiened model
    }
    
    network = MultiFTNet(**param).to(device)
    network.eval()
    model = Network(network)
    # Convert model torch to onnx
    torch.onnx.export(model,
        torch.randn(3, 80, 80, requires_grad=True),
        output,
        verbose=False,
        input_names=["data"],
        output_names=["softmax"],
        export_params=True,
        opset_version=11
    )


def pth2onnx_MiniFAS(pth_input, onnx_output, num_class, bias):
    """
        Only use for miniFAS
    """

    model_name = os.path.basename(pth_input)
    device = 'cpu'

    h_input, w_input, model_type, _ = parse_model_name(model_name)
    kernel_size = get_kernel(h_input, w_input,)
    model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)
    model.prob = nn.Identity()
    model.prob = nn.Linear(in_features=128, out_features=num_class, bias=bias)

    state_dict = torch.load(pth_input, map_location=device)
    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name.find('module.') >= 0:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key[7:]
            # print("name_key", name_key)
            new_state_dict[name_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()

    _model_ = Network(model)
    _model_.eval()
    
    # Convert model torch to onnx
    torch.onnx.export(_model_,
        torch.randn(3, 80, 80, requires_grad=True),
        onnx_output,
        verbose=False,
        input_names=["data"],
        output_names=["softmax"],
        export_params=True,
        opset_version=11
    )


def tar2onnx(pth_input, onnx_output):
    # File config https://github.com/kprokofi/light-weight-face-anti-spoofing/tree/82a2c26bacff5462c6f9320c34ae0363ccbed497/configs
    config = read_py_config('/home/anlab/Documents/Face_Anti_Spoofing/spoofing/src/configs/config.py')
    parameters = dict(width_mult=config.model.width_mult,
                    prob_dropout=config.dropout.prob_dropout,
                    type_dropout=config.dropout.type,
                    mu=config.dropout.mu,
                    sigma=config.dropout.sigma,
                    embeding_dim=config.model.embeding_dim,
                    prob_dropout_linear = config.dropout.classifier,
                    theta=config.conv_cd.theta,
                    multi_heads = config.multi_task_learning)
    model = mobilenetv3_large(**parameters)
    model.forward = model.forward_to_onnx

    # Load weight in model mobilenet large
    checkpoint = torch.load(pth_input, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    dummy_input = torch.rand(size=(3, 128, 128), device='cpu')
    model.eval()
    torch.onnx.export(model,
        dummy_input,
        onnx_output,
        input_names=["data"],
        output_names=["softmax"],
        verbose=True,
        opset_version=12
    )


if __name__ == '__main__':
    
    inp = '/home/hoang/Documents/work/code/DATN/Research/saved_logs/snapshot/Anti_Spoofing_1_224x224/1_80x80_MiniFASNetV2.pth'
    out = '1_80x80_MultiFTNet.onnx'
    pth2onnx(inp, out)
    