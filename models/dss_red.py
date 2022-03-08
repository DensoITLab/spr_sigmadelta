from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from models.dss_net import *
import torch.nn.functional as F
from models.dss_layer import DSSConv2d, DSSInput, DSSInvolution, conv1x1, conv1x1_out, conv3x3
import torch.utils.checkpoint as cp
# Adopted from  Involution: Inverting the Inherence of Convolution for Visual Recognition (CVPR'21)
#  https://github.com/d-li14/involution/blob/main/cls/mmcls/models/backbones/rednet.py


class Block(nn.Module):
    def __init__(self, in_channels, out_channels,  expansion=4, stride=1, downsample=None, settings=[]):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.downsample = downsample

        self.mid_channels = out_channels // expansion
        self.conv1 = conv1x1(in_channels, self.mid_channels,  settings=settings)
        # self.conv2 = DSSConv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1,  stride=stride,  norm=nn.BatchNorm2d(self.mid_channels), bias=False, settings=settings)
        self.conv2 = DSSInvolution(self.mid_channels, 7, stride=1, settings=settings)
        self.conv3 = conv1x1(self.mid_channels, out_channels, activation='I', stride=stride, settings=settings)
        
    def forward(self, x):
        def _inner_forward(x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            identity = self.downsample(x) if self.downsample is not None else x
            out = (out+identity)
            return out

        if x.requires_grad:
            out = _inner_forward(x)
            # out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out
    
class RED(DSSNet):
    def __init__(self, input_ch, stage_config, head_config, stem_channels=64, base_channels=64, stem_stride=2, last_act='I', settings=[]):
        super(RED, self).__init__(settings=settings)
        # stem_channels=256

        self.q_in = DSSInput(input_ch, settings=settings)
        # Stem
        self.stem = nn.Sequential(
            conv3x3(input_ch, stem_channels//2, 3, stride=stem_stride, settings=settings),
            DSSInvolution(stem_channels//2, 3, stride=1, settings=settings),
            conv3x3(stem_channels//2, stem_channels, 3, settings=settings),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        c = base_channels
        # Res-layers
        self.res_layers = nn.Sequential()
        for stage_idx, n_block in enumerate(stage_config):
            if stage_idx==0:
                self.res_layers.add_module("sgate:"+ str(stage_idx), self._make_layer(stem_channels, base_channels, n_block, stride=1, settings=settings))
            else:
                self.res_layers.add_module("sgate:"+ str(stage_idx), self._make_layer(c, c*2, n_block, stride=2, settings=settings))
                c *= 2

        # Avg Pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Output layer
        self.output_layers = nn.Sequential()
        for head_idx in range(len(head_config)-2):
            self.output_layers.add_module("mlp" + str(head_idx), conv1x1(head_config[head_idx], head_config[head_idx+1], settings=settings))
        self.output_layers.add_module("mlp"+ str(len(head_config)-2), conv1x1_out(head_config[-2], head_config[-1], activation=last_act, settings=settings))

        self.search_DSS_cand()
        self.search_th_cand()
        print(len(self.DSS_cand))


    def _make_layer(self, in_channels, out_channels, blocks, stride=2,  settings=[]):
        # https://cv-tricks.com/keras/understand-implement-resnets/
        layers = []
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample =  DSSConv2d(in_channels, out_channels, stride=stride,  norm=nn.BatchNorm2d(out_channels), bias=False, activation='I', settings=settings)

        for blocks_idx in range(0, blocks):
            if blocks_idx==0:
                layers.append(Block(in_channels, out_channels, stride=stride, downsample=downsample,  settings=settings))
            else:
                layers.append(Block(out_channels, out_channels, downsample=None,  settings=settings))
        return nn.Sequential(*layers)

    def forward(self, input):    
        x = self.q_in(input)
        x = self.stem(x)
        x = self.res_layers(x)
        x = self.avgpool(x)
        x = self.output_layers(x).squeeze()

        if len(x.shape)==0:
            x = x.view([1,1,1,1])

        return self.ext_output(x)


if __name__ == '__main__':
    net = RED(layers=(3, 4, 6), kernels=[3, 7, 7], num_classes=10).cuda().eval()
    # net = san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=1000).cuda().eval()
    print(net)

    for itr in range(5):
        net.register_rand_hook()
        y, layerout = net(torch.randn(4, 1, 36, 36).cuda())
        # print(y.size())
        print('DSS output selected: %d, exp_x: %s, mask: %s'%(layerout[0], str(layerout[1].shape) ,  str(layerout[2].shape)))
