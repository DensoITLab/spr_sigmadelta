import torch.nn as nn
from models.dss_layer import DSSConv2d, DSSInput, conv3x3, conv3x3_pool
from models.dss_util import compute_act_shape
from models.dss_net import DSSNet
import numpy as np

class DSSObjectDet(DSSNet):
    def __init__(self, nr_classes, in_c=2, nr_box=2, small_out_map=True, input_size=(223, 287),  settings=[]):
        super().__init__()
        self.settings = settings
        self.split_half = True
        self.nr_box = nr_box
        self.nr_classes = nr_classes
        self.kernel_size = 3

        act_shape = []
        act_shape.append(compute_act_shape(input_size, self.kernel_size, 1, padding=1))
        act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
        act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
        act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
        act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
        act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))

        sparse_out_channels = 256
        self.layers = nn.ModuleList()
        self.layers.append(DSSInput(in_c, settings=self.settings))
        self.conv_block(in_c, 16, act_shape[0], 1, self.settings.expantion_ratio)
        self.conv_block(16, 32, act_shape[1], self.settings.expantion_ratio, self.settings.expantion_ratio)
        self.conv_block(32, 64, act_shape[2], self.settings.expantion_ratio, self.settings.expantion_ratio)
        self.conv_block(64, 128, act_shape[3], self.settings.expantion_ratio, self.settings.expantion_ratio)
        self.conv_block(128, 256, act_shape[4], self.settings.expantion_ratio, 1, max_pool=False)
        self.layers.append(DSSConv2d(256, 512, act_shape[5], norm=nn.BatchNorm2d(512), bias=False, kernel_size=3, stride=2, settings=self.settings))

        if small_out_map:
            self.cnn_spatial_output_size = [5, 7]
        else:
            self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * 512
        self.layers.append(DSSConv2d(self.linear_input_features, 1024, add_flat=True,  settings=self.settings))# %%
        self.layers.append(DSSConv2d(1024,  spatial_size_product*(nr_classes + 5*self.nr_box), activation='I', settings=self.settings))# %%

        self.search_DSS_cand()
        self.search_th_cand()

    def conv_block(self, in_c, out_c, act_shape,  expantion_ratio_in, expantion_ratio_out,max_pool=True,):
        self.layers.append(conv3x3(in_c*expantion_ratio_in, out_c*expantion_ratio_out,  act_shape, expantion_ratio=(expantion_ratio_in, expantion_ratio_out), settings=self.settings))
        
        if max_pool:
            self.layers.append(conv3x3_pool(out_c*expantion_ratio_out, out_c*expantion_ratio_out, act_shape, expantion_ratio=(expantion_ratio_out, expantion_ratio_out), settings=self.settings))
        else:
            self.layers.append(conv3x3(out_c*expantion_ratio_out, out_c*expantion_ratio_out, act_shape, expantion_ratio=(expantion_ratio_out, expantion_ratio_out), settings=self.settings))

    def forward(self, x):
        self.pre_process()
        for _, module in enumerate(self.layers):
            x = module(x)
        x = x.squeeze()
        return self.ext_output(x)
        