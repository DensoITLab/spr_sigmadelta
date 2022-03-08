import torch.nn as nn
from models.dss_layer import DSSConv2d, DSSInput
from models.dss_net import DSSNet
from models.dss_util import  compute_act_shape
from models.dss_net import *

class DSSNMNIST(DSSNet):
    def __init__(self, input_shape, settings=[]):
        super(DSSNMNIST, self).__init__(settings=settings)
        expantion_ratio=settings.expantion_ratio
        self.settings = settings
        n_ch = [2, 32, 64, 16, 10, 10] # Original 24C5−36C5−48C5−64C3−64C3 −1164FC−100FC− 50FC−10FC−10FC.
        n_kn = [5, 5, 3, 1, 1]

        self.layers = nn.ModuleList()
        self.layers.append(DSSInput(n_ch[0], settings=settings)) 
        act_shape = input_shape
        act_shape = compute_act_shape(act_shape, n_kn[0], 1, padding=2)
        self.layers.append(DSSConv2d(n_ch[0], n_ch[1]*expantion_ratio, act_shape, n_kn[0], stride=1, padding=2,  norm=nn.BatchNorm2d(n_ch[1]*expantion_ratio), pool=nn.AvgPool2d(kernel_size=2), bias=False, expantion_ratio=[1.0, expantion_ratio], settings=settings)) 
        
        act_shape = compute_act_shape([act_shape[0]//2, act_shape[1]//2], n_kn[1],  1, padding=2)
        self.layers.append(DSSConv2d(n_ch[1]*expantion_ratio, n_ch[2]*expantion_ratio, act_shape, n_kn[1], stride=1, padding=2,  norm=nn.BatchNorm2d(n_ch[2]*expantion_ratio), pool=nn.AvgPool2d(kernel_size=2),  bias=False, expantion_ratio=[expantion_ratio,expantion_ratio], settings=settings)) 

        act_shape = compute_act_shape([act_shape[0]//2, act_shape[1]//2], n_kn[2],  1, padding=1)
        self.layers.append(DSSConv2d(n_ch[2]*expantion_ratio, n_ch[3], act_shape, n_kn[2], stride=1, padding=1,  norm=nn.BatchNorm2d(n_ch[3]), bias=False, expantion_ratio=[expantion_ratio, 1.0], settings=settings)) 

        n_feat = (n_ch[3]*8*8)
        self.layers.append(DSSConv2d(n_feat,  n_ch[4], add_flat=True, settings=settings))
        self.layers.append(DSSConv2d(n_ch[4], n_ch[5], activation='I', settings=settings))

        self.search_DSS_cand()
        self.search_th_cand()

    def forward(self, x):
        self.pre_process()
        for _, module in enumerate(self.layers):
            x = module(x)
        x = x.squeeze()
        return self.ext_output(x)