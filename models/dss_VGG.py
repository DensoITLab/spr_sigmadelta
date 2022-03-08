import torch.nn as nn
from models.dss_layer import DSSConv2d, DSSInput, conv3x3, conv3x3_pool
from models.dss_util import compute_act_shape
from models.dss_net import DSSNet

class DSSVGGCls(DSSNet):
    def __init__(self, nr_classes, in_c=2, vgg_12=False, input_size=(223, 287), settings=[]):
        super().__init__()
        self.settings = settings
        self.kernel_size = 3
        self.split_half = True
        if vgg_12:
            act_shape = []
            act_shape.append(compute_act_shape(input_size, self.kernel_size, 1, padding=1))
            act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
            act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
            act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
            act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
            act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))

            sparse_out_channels = 128
            self.layers = nn.ModuleList([
                self.conv_block(in_c,16, act_shape[0], settings=self.settings)[:],
                self.conv_block(16, 32, act_shape[1], settings=self.settings)[:],
                self.conv_block(32, 64, act_shape[2], settings=self.settings)[:],
                self.conv_block(64, 128, act_shape[3], settings=self.settings[:]),
                nn.ModuleList([DSSConv2d(128, 256, act_shape[4],  kernel_size=self.kernel_size, padding=1, stride=1, settings=self.settings, bias=False)]),
                nn.ModuleList([DSSConv2d(256, sparse_out_channels, act_shape[5],  kernel_size=self.kernel_size, padding=0, stride=2, settings=self.settings)]),
            ])

        else:
            act_shape = []
            act_shape.append(compute_act_shape(input_size, self.kernel_size, 1, padding=1))
            act_shape.append(compute_act_shape(act_shape[-1], self.kernel_size, 2, padding=1, offs=-1))
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
            self.conv_block(128, 256, act_shape[4], self.settings.expantion_ratio, 1)
            self.layers.append(conv3x3(256, 512, act_shape[5], settings=self.settings))
            self.layers.append(DSSConv2d(512, sparse_out_channels, act_shape[6], norm=nn.BatchNorm2d(sparse_out_channels),  kernel_size=3, padding=0, stride=2,  settings=self.settings, bias=False))

        self.linear_input_features = 2 * 3 * sparse_out_channels
        self.layers.append(DSSConv2d(self.linear_input_features, nr_classes, activation='I', settings=self.settings, add_flat=True))# %%
    

        self.search_DSS_cand()
        self.search_th_cand()

    def conv_block(self, in_c, out_c, act_shape, expantion_ratio_in, expantion_ratio_out):        
        self.layers.append(conv3x3(in_c*expantion_ratio_in, out_c*expantion_ratio_out,  act_shape, expantion_ratio=(expantion_ratio_in, expantion_ratio_out), settings=self.settings))
        self.layers.append(conv3x3_pool(out_c*expantion_ratio_out, out_c*expantion_ratio_out, act_shape, expantion_ratio=(expantion_ratio_out, expantion_ratio_out), settings=self.settings))

    def forward(self, x):
        self.pre_process()
        for _, module in enumerate(self.layers):
            x = module(x)
        x = x.squeeze()
        return self.ext_output(x)