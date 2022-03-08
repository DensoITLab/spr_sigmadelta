import torch.nn as nn
from models.dss_layer import DSSConv2d, DSSInput, DSSAlign
from models.dss_util import compute_act_shape
from models.dss_net import *

class DSSPilotNet(DSSNet):
    def __init__(self, input_shape, settings=[]):
        super(DSSPilotNet, self).__init__()
        self.settings = settings
        expantion_ratio=settings.expantion_ratio
        self.split_half = True
        n_ch = [3, 24, 36, 48, 64, 64, 1164, 100, 50, 10, 1] # Original 24C5−36C5−48C5−64C3−64C3 −1164FC−100FC− 50FC−10FC−1FC.
        print(n_ch)
        n_kn = [5, 5, 5, 3, 3, 1, 1, 1, 1]
        n_st = [2, 2, 2, 1, 1, 1, 1, 1, 1] # Original 24C5−36C5−48C5−64C3−64C3 −1164FC−100FC− 50FC−10FC−1FC.

        self.layers = nn.ModuleList()
        self.layers.append(DSSInput(n_ch[0], settings=settings)) 
        act_shape = input_shape
        act_shape = compute_act_shape(act_shape, n_kn[0], n_st[0])
        self.layers.append(DSSConv2d(n_ch[0], n_ch[1]*expantion_ratio, act_shape, n_kn[0], stride=n_st[0], padding=0,  norm=nn.BatchNorm2d(n_ch[1]*expantion_ratio), bias=False, expantion_ratio=[1.0,expantion_ratio], settings=settings)) 
        act_shape = compute_act_shape(act_shape, n_kn[1], n_st[1])
        self.layers.append(DSSConv2d(n_ch[1]*expantion_ratio, n_ch[2]*expantion_ratio, act_shape, n_kn[1], stride=n_st[1], padding=0,  norm=nn.BatchNorm2d(n_ch[2]*expantion_ratio), bias=False, expantion_ratio=[expantion_ratio,expantion_ratio], settings=settings)) 
        act_shape = compute_act_shape(act_shape, n_kn[2], n_st[2])
        self.layers.append(DSSConv2d(n_ch[2]*expantion_ratio, n_ch[3]*expantion_ratio, act_shape, n_kn[2], stride=n_st[2], padding=0,  norm=nn.BatchNorm2d(n_ch[3]*expantion_ratio), bias=False, expantion_ratio=[expantion_ratio,expantion_ratio], settings=settings)) 
        act_shape = compute_act_shape(act_shape, n_kn[3], n_st[3])
        self.layers.append(DSSConv2d(n_ch[3]*expantion_ratio, n_ch[4]*expantion_ratio, act_shape, n_kn[3], stride=n_st[3], padding=0,  norm=nn.BatchNorm2d(n_ch[4]*expantion_ratio), bias=False, expantion_ratio=[expantion_ratio,expantion_ratio], settings=settings))
        act_shape = compute_act_shape(act_shape, n_kn[4], n_st[4])
        self.layers.append(DSSConv2d(n_ch[4]*expantion_ratio, n_ch[5], act_shape, n_kn[4], stride=n_st[4], padding=0,  norm=nn.BatchNorm2d(n_ch[5]), bias=False, expantion_ratio=[expantion_ratio,1.0], settings=settings))

        n_feat = 1152

        self.layers.append(DSSConv2d(n_feat,  n_ch[6], norm=nn.BatchNorm2d(n_ch[6]), bias=False, drop_rate=0.0, add_flat=True, settings=settings)) # 1152->1164
        self.layers.append(DSSConv2d(n_ch[6], n_ch[7], norm=nn.BatchNorm2d(n_ch[7]), bias=False, settings=settings)) # 1164->100
        self.layers.append(DSSConv2d(n_ch[7], n_ch[8], norm=nn.BatchNorm2d(n_ch[8]), bias=False, settings=settings)) # 100->50
        self.layers.append(DSSConv2d(n_ch[8], n_ch[9], norm=nn.BatchNorm2d(n_ch[9]), bias=False, settings=settings)) # 50->10
        self.layers.append(DSSConv2d(n_ch[9], n_ch[10], activation='atan', settings=settings)) # 10->1

        self.selected_out =[]
        self.fhooks = []
        self.search_DSS_cand()
        self.search_th_cand()

    def forward(self, x):
        self.pre_process()
        for _, module in enumerate(self.layers):
            x = module(x)
        x = x.squeeze()
        return self.ext_output(x)


class DSSWPilotNet(DSSNet):
    def __init__(self, inputs_shape, settings=[]):
        super(DSSWPilotNet, self).__init__()
        self.is_aligned = True
        self.settings = settings
        expantion_ratio=settings.expantion_ratio
        self.split_half = True
        n_ch = [3, 24, 36, 48, 64, 64, 1164, 100, 50, 10, 1] # Original 24C5−36C5−48C5−64C3−64C3 −1164FC−100FC− 50FC−10FC−1FC.
        print(n_ch)
        n_kn = [5, 5, 5, 3, 3, 1, 1, 1, 1]
        n_st = [2, 2, 2, 1, 1, 1, 1, 1, 1] # Original 24C5−36C5−48C5−64C3−64C3 −1164FC−100FC− 50FC−10FC−1FC.

        self.layers = nn.ModuleList()
        self.layers.append(DSSInput(n_ch[0], settings=settings)) 
        act_shape = inputs_shape
        act_shape = compute_act_shape(act_shape, n_kn[0], n_st[0])
        self.layers.append(DSSConv2d(n_ch[0], n_ch[1]*expantion_ratio, act_shape, n_kn[0], stride=n_st[0], padding=0,  norm=nn.BatchNorm2d(n_ch[1]*expantion_ratio), bias=False, expantion_ratio=[1.0,expantion_ratio], settings=settings)) 
        self.layers.append(DSSAlign(n_ch[1]*expantion_ratio, expantion_ratio=[expantion_ratio,expantion_ratio], settings=settings))
        act_shape = compute_act_shape(act_shape, n_kn[1], n_st[1])
        self.layers.append(DSSConv2d(n_ch[1]*expantion_ratio, n_ch[2]*expantion_ratio, act_shape, n_kn[1], stride=n_st[1], padding=0,  norm=nn.BatchNorm2d(n_ch[2]*expantion_ratio), bias=False, expantion_ratio=[expantion_ratio,expantion_ratio], is_aligned=True, settings=settings)) 
        act_shape = compute_act_shape(act_shape, n_kn[2], n_st[2])
        self.layers.append(DSSConv2d(n_ch[2]*expantion_ratio, n_ch[3]*expantion_ratio, act_shape, n_kn[2], stride=n_st[2], padding=0,  norm=nn.BatchNorm2d(n_ch[3]*expantion_ratio), bias=False, expantion_ratio=[expantion_ratio,expantion_ratio], is_aligned=True, settings=settings)) 
        act_shape = compute_act_shape(act_shape, n_kn[3], n_st[3])
        self.layers.append(DSSConv2d(n_ch[3]*expantion_ratio, n_ch[4]*expantion_ratio, act_shape, n_kn[3], stride=n_st[3], padding=0,  norm=nn.BatchNorm2d(n_ch[4]*expantion_ratio), bias=False, expantion_ratio=[expantion_ratio,expantion_ratio], is_aligned=True, settings=settings))
        act_shape = compute_act_shape(act_shape, n_kn[4], n_st[4])
        self.layers.append(DSSConv2d(n_ch[4]*expantion_ratio, n_ch[5], act_shape, n_kn[4], stride=n_st[4], padding=0,  norm=nn.BatchNorm2d(n_ch[5]), bias=False, expantion_ratio=[expantion_ratio,1.0], is_aligned=True, settings=settings))

        n_feat = 1152

        self.layers.append(DSSConv2d(n_feat,  n_ch[6], norm=nn.BatchNorm2d(n_ch[6]), bias=False, drop_rate=0.0, add_flat=True, is_aligned=True, settings=settings)) # 1152->1164
        self.layers.append(DSSConv2d(n_ch[6], n_ch[7], norm=nn.BatchNorm2d(n_ch[7]), bias=False, is_aligned=True, settings=settings)) # 1164->100
        self.layers.append(DSSConv2d(n_ch[7], n_ch[8], norm=nn.BatchNorm2d(n_ch[8]), bias=False, is_aligned=True, settings=settings)) # 100->50
        self.layers.append(DSSConv2d(n_ch[8], n_ch[9], norm=nn.BatchNorm2d(n_ch[9]), bias=False, is_aligned=True, settings=settings)) # 50->10
        self.layers.append(DSSConv2d(n_ch[9], n_ch[10], activation='atan', is_aligned=True, settings=settings)) # 10->1

        self.selected_out =[]
        self.fhooks = []
        self.search_DSS_cand()
        self.search_th_cand()

    def forward(self, x):
        self.pre_process()
        for _, module in enumerate(self.layers):
            x = module(x)
        x = x.squeeze()
        return self.ext_output(x)