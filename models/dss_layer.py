from numpy.lib.function_base import delete
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.dss_util import none_func, round_func, Ternarizer, TernarizerHard, Binarizer, Masker
from models.dss_util import init_weights_uniform, init_weights_normal, mask_weight
from models.activation import get_act

########################################################################
# MLCLayer Wrapper
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def conv1x1(in_planes, out_planes, bn=True, activation=None, stride=1, settings=[]):
    if bn==True:
        norm = nn.BatchNorm2d(out_planes)
        bias = False
    else:
        norm = none_func
        bias = True
    return DSSConv2d(in_planes, out_planes, kernel_size=1,  norm=norm, bias=bias, activation=activation, stride=stride, settings=settings)

def conv1x1_out(in_planes, out_planes,  activation='I', settings=[]):
    return DSSConv2d(in_planes, out_planes, kernel_size=1,  norm=none_func, bias=True, activation=activation, settings=settings)

def conv3x3(in_planes, out_planes, act_shape=[1,1], bn=True, activation=None, stride=1, expantion_ratio=1.0, settings=[]):
    if bn==True:
        norm = nn.BatchNorm2d(out_planes)
        bias = False
    else:
        norm = none_func
        bias = True
    return DSSConv2d(in_planes, out_planes, act_shape=act_shape, kernel_size=3,  padding=1, norm=norm, bias=bias, activation=activation, stride=stride, settings=settings)

def conv3x3_pool(in_planes, out_planes, act_shape=[1,1], bn=True, activation='relu', stride=1, expantion_ratio=1.0, settings=[]):
    if bn==True:
        norm = nn.BatchNorm2d(out_planes)
        bias = False
    else:
        norm = none_func
        bias = True
    return DSSConv2d(in_planes, out_planes, act_shape=act_shape, kernel_size=3, padding=1,  norm=norm, expantion_ratio=expantion_ratio, bias=bias, activation=activation, stride=stride, pool=nn.MaxPool2d(kernel_size=3, stride=2), settings=settings)


########################################################################
# Base Layer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class DSSLayer(nn.Module):
    def __init__(self, out_channels, settings=[]):
        super(DSSLayer, self).__init__()
        self.settings = settings
        self.out_channels = out_channels
        self.mp = []
        self.mp_lft = []
        self.max_scale()
        # self.training=False
        self.round =  round_func(self.settings.quantizer, self.settings.max_scale, self.settings.min_scale)

        # Function handle
        self.mask       =  lambda x,y: Masker().apply(x,y)
        self.ternalize  =  lambda x,y: Ternarizer().apply(x,y)
        self.binalize   =  lambda x,y: Binarizer().apply(x,y)
        # misc
        self.fuse_bn = False
        self.register_buffer("max_x", torch.tensor(1.0))
        self.map_size = torch.Size([1, 1])

    def max_scale(self):
        th_base=self.settings.max_scale - self.settings.min_scale
        # th_base=2.0
        quantizer = self.settings.quantizer

        if quantizer in ['MG_divmul_lin', 'floor', 'ULSQ_muldiv_lin']:
            th = 1.0/th_base
        elif quantizer in ['MG_muldiv_lin','ULSQ_divmul_lin', 'floor_inv', 'nan']:
            th = th_base
        elif quantizer in ['MG_divmul_log', 'floor_log','ULSQ_divmul_log']:
            # th = np.log(1.0/th_base).item()
            th = 0.0
        elif quantizer in ['MG_muldiv_log','ULSQ_muldiv_log', 'floor_inv_log']:
            # th = np.log(th_base).item()
            th = 0.0
        else:
            th = 0.0
            print('Not implemented')
    
        if self.settings.channel_wise_th==True:
            self.register_parameter("th", nn.Parameter(th*torch.ones(1, self.out_channels,1,1)))
        else:
            self.register_parameter("th", nn.Parameter(torch.tensor([th])))
        self.th.requires_grad = False
    
    def get_scale(self):
        th = self.th
        quantizer = self.settings.quantizer
        if quantizer in ['MG_divmul_lin', 'ULSQ_muldiv_lin','floor']:
            return 1.0/th.abs()
        elif quantizer in ['MG_muldiv_lin', 'ULSQ_divmul_lin', 'floor_inv', 'nan']:
            return th.abs()
        elif quantizer in ['MG_divmul_log', 'ULSQ_divmul_log','floor_log']:
            s0 = np.log(self.settings.max_scale-self.settings.min_scale)
            return 1.0/(th.sub().exp())
        elif quantizer in ['MG_muldiv_log', 'ULSQ_muldiv_log','floor_inv_log']:
            s0 = np.log(self.settings.max_scale-self.settings.min_scale)
            return th.add(s0).exp() + self.settings.min_scale
        else:
            print('ERROR:get_scale (not implemented)')
            return th
            
    def get_nbit(self):
        max_x_ = max(np.ceil(self.max_x.data.item()), 1.0)
        scale = self.get_scale().mean().data.item()
        nbit = np.log2(max_x_*scale)+1
        if nbit>64:
            print('get_nbit too large %f'%(nbit))
        return nbit
        # return np.ceil(np.log2(max_x*th))
    
    def change_quantizer_LSQ(self):
        quantizer = self.settings.quantizer
        if quantizer in ['MG_divmul_lin']:
            self.round =  round_func('ULSQ_muldiv_lin', self.settings.max_scale, self.settings.min_scale)
        elif quantizer in ['MG_muldiv_lin', 'nan']:
            self.round =  round_func('ULSQ_divmul_lin', self.settings.max_scale, self.settings.min_scale)
        elif quantizer in ['MG_divmul_log']:
            self.round =  round_func('ULSQ_divmul_log', self.settings.max_scale, self.settings.min_scale)
        elif quantizer in ['MG_muldiv_log']:
            self.round =  round_func('ULSQ_muldiv_log', self.settings.max_scale, self.settings.min_scale)
    
    def change_quantizer_MG(self):
        quantizer = self.settings.quantizer
        self.round =  round_func(quantizer, self.settings.max_scale, self.settings.min_scale)

########################################################################
# MLCInput
## 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class DSSInput(DSSLayer):
    def __init__(self, out_channels, settings=[]):
        super(DSSInput, self).__init__(out_channels, settings=settings)

    def forward(self, inputs):
        if self.training or 1:
            return self.forward_train(inputs)
        else:
            return self.forward_test(inputs)

    def forward_train(self, x): 
        x = self.round(x, self.th, self.training)
        with torch.no_grad():
            self.max_x = x.abs().max()
        return x

    def forward_test(self, inputs):
        x = inputs[0]
        if len(self.mp)==0:
            self.mp = x
            self.mp_lft = 0*x
        else:
            self.mp = self.mp + x

        do = self.mp - self.mp_lft

        if self.th.requires_grad:
            delta = self.round(do, self.th, self.training)
        else:
            delta = do

        epsilon = do - delta
        self.mp_lft = self.mp - epsilon
        return delta
    

########################################################################
# MLC2D
## kernel_mode :['conv_nan', 'lc_nan', 'conv_conv', 'conv_lc', 'lc_lc']
# norm : [nn.InstanceNorm2d, CBatchNorm2d] 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/bfeb268d61707441b08588e7edce000b919c9da4/src/Bayes_By_Backprop_Local_Reparametrization/model.py
class DSSConv2d(DSSLayer):
    def __init__(self, in_channels, out_channels, act_shape=[1,1], kernel_size=1, stride=1, padding=0, norm=none_func, pool=none_func, bias=True ,activation=None, drop_rate=0.0, add_flat=False, expantion_ratio=[1.0,1.0], is_aligned=False, settings=[]):
        super(DSSConv2d, self).__init__(out_channels, settings=settings)
        self.kernel_size = kernel_size
        self.activation = self.settings.activation if activation is None else activation
        self.drop_rate = drop_rate
        self.add_flat = add_flat
        self.act_shape = act_shape
        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.pool = pool
        self.expantion_ratio = expantion_ratio
        self.is_aligned = is_aligned

        # Activation
        self.act = get_act(self.activation, out_channels)

        # Weight
        if self.settings.kernel_mode in ['conv_nan', 'conv_conv', 'conv_lc']:
            self.register_parameter("w", nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size)))
            self.mask_th, self.w_mask = init_weights_uniform(self.w.data, expantion_ratio=expantion_ratio, base_th=self.settings.q_threshold)
            # self.w_mask = init_weights_normal(self.w.data, expantion_ratio=expantion_ratio, mask_th=self.settings.q_threshold)
        elif self.settings.kernel_mode in ['lc_nan', 'lc_lc']: 
            self.register_parameter("w", nn.Parameter(torch.zeros(out_channels, in_channels*kernel_size*kernel_size, act_shape[0]*act_shape[1])))
            w = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
            # init_weights_normal(w.data, mode='fan_out')
            init_weights_uniform(w.data, mode='fan_in')
            self.w.data = w.data.view([out_channels,  in_channels*kernel_size*kernel_size,1]).repeat([1,1,act_shape[0]*act_shape[1]]).clone()
        elif self.settings.kernel_mode in ['lrlc_nan', 'lrlc_lrlc']:
            self.register_parameter("cw_row", nn.Parameter(torch.ones(self.settings.spatial_rank, act_shape[0], 1)/self.settings.spatial_rank))
            self.register_parameter("cw_col", nn.Parameter(torch.ones(self.settings.spatial_rank, 1, act_shape[1])/self.settings.spatial_rank))
            self.register_parameter("w", nn.Parameter(torch.zeros(self.settings.spatial_rank, out_channels, in_channels, kernel_size, kernel_size)))
            for rank in range(self.settings.spatial_rank):
                init_weights_normal(self.w.data[rank,:,:,:,:])

        # Bias
        if bias:
            if self.settings.kernel_mode in ['lc_nan', 'lc_lc'] and self.settings.bias_pd:
                self.register_parameter("b", nn.Parameter(torch.zeros(out_channels, act_shape[0], act_shape[1])))
            elif self.settings.kernel_mode in ['lrlc_nan', 'lrlc_lrlc'] and self.settings.bias_pd:
                self.register_parameter("b_row", nn.Parameter(torch.zeros(1, 1, act_shape[0], 1)))
                self.register_parameter("b_col", nn.Parameter(torch.zeros(1, 1, 1, act_shape[1])))
                self.register_parameter("b", nn.Parameter(torch.zeros(1, out_channels,1,1)))
            else:
                self.register_parameter("b", nn.Parameter(torch.zeros(out_channels,1,1)))
        
        # Mask (We need to prepare mask separately when kernel and mask has different configulation)
        if self.settings.kernel_mode in ['conv_lc']:
            self.register_parameter("m", nn.Parameter(1e-2 + torch.zeros(out_channels, in_channels*kernel_size*kernel_size, act_shape[0]*act_shape[1])))
        

    def conv2d(self, x):
        if self.settings.kernel_mode=='conv_nan':
            output = F.conv2d(x, self.w,stride=self.stride, padding=self.padding)
        elif self.settings.kernel_mode=='conv_conv':
            output = F.conv2d(x, self.mask(self.w, self.mask_th), stride=self.stride, padding=self.padding)
        elif self.settings.kernel_mode=='conv_lc':
            input_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            output = torch.einsum('ikl, jkl, jk->ijl', input_unfold,  self.binalize(self.m),  self.w.view(self.w.shape[0], -1)).view(x.shape[0], self.out_channels, self.act_shape[0], self.act_shape[1])    
            # output = torch.einsum('ijkl,jk->ijl', input_unfold * self.round_m(self.m), self.w.view(self.w.shape[0], -1)).view(x.shape[0], self.out_channels, self.act_shape[0], self.act_shape[1])    
        elif self.settings.kernel_mode=='lc_nan':
            input_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            output = torch.einsum('ikl,jkl->ijl', input_unfold, self.w).view(x.shape[0], self.out_channels, self.act_shape[0], self.act_shape[1])
        elif self.settings.kernel_mode=='lc_lc':
            w = self.mask(self.w,self.mask_th)
            input_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            output = torch.einsum('ikl,jkl->ijl', input_unfold, w).view(x.shape[0], self.out_channels, self.act_shape[0], self.act_shape[1])
        elif self.settings.kernel_mode=='lrlc_nan':
            convs =  F.conv2d(x, self.w.flatten(0,1).squeeze(0), stride=self.stride, padding=self.padding)
            output = torch.einsum('bkcxy,kxy->bcxy', convs.view(x.shape[0], self.w.shape[0], self.w.shape[1], convs.shape[2],convs.shape[3]), torch.softmax(self.cw_row+self.cw_col, dim=0)).view(x.shape[0], self.out_channels, self.act_shape[0], self.act_shape[1])
        elif self.settings.kernel_mode=='lrlc_lrlc':
            input_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride).unsqueeze(1)
            w_flat = self.mask(torch.einsum('koi,kx->oix',  self.w.flatten(2), torch.softmax(self.cw_row+self.cw_col ,dim=0).flatten(-2)),self.mask_th)
            output = ((w_flat)*input_unfold).sum(2).view(x.shape[0], self.out_channels, self.act_shape[0], self.act_shape[1])
        else:
            print('Not supported ' + self.settings.kernel_mode)
            output = None

        if len(self.mp)==0 and hasattr(self, 'b'):
            if hasattr(self, 'b_col'):
                output.add_(self.b).add_(self.b_col).add_(self.b_row)
            else:
                output += self.b

        self.map_size = output.shape[2:]
        return output

    def get_exp(self, inputs):
        x = inputs[0]
        if self.add_flat:
            x = torch.flatten(x, start_dim=1).unsqueeze(-1).unsqueeze(-1)
        if self.drop_rate>0:
            x = F.dropout(x, p=0.5, training=self.training)         
        
        # Compute Mask
        if self.settings.kernel_mode=='conv_conv':
            mask = self.binalize(self.w, self.mask_th)
        elif self.settings.kernel_mode=='conv_lc':
            mask = self.binalize(self.m)
        elif self.settings.kernel_mode=='lc_lc':
            mask = self.ternalize(self.w, self.mask_th)   
        elif self.settings.kernel_mode=='lrlc_lrlc':
            w = torch.einsum('koi,kx->oix',  self.w.flatten(2), torch.softmax(self.cw_row+self.cw_col ,dim=0).flatten(-2))
            mask = self.ternalize(w, self.mask_th) 
        else:
            mask = None

        # Compute DSS/MAC
        if self.is_aligned:
            i_prv = 2
            i_cur = 1
            n_chunk = 3
        else:
            i_prv = 0
            i_cur = 1
            n_chunk = 2

        if self.settings.kernel_mode in ['conv_nan', 'conv_conv']:
            input_split = torch.chunk(x, n_chunk, dim=0)
            d_input=(input_split[i_cur]-input_split[i_prv])
        else:
            input_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            input_unfold_split = torch.chunk(input_unfold, n_chunk, dim=0)
            d_input=(input_unfold_split[i_cur]-input_unfold_split[i_prv])


        if self.settings.kernel_mode in ['conv_nan', 'lrlc_nan', 'lc_nan'] or self.settings.MAC_loss==0:
            with torch.no_grad():
                mac = torch.tensor((d_input!=0).abs().flatten(1).sum(1).mul(self.out_channels).to(x))
            dss = (d_input).abs().flatten(1).sum(1).mul(self.out_channels*self.kernel_size*self.kernel_size)
        elif self.settings.kernel_mode=='conv_conv':
            input_split = torch.chunk(x, n_chunk, dim=0)
            d_input=(input_split[i_cur]-input_split[i_prv])
            dss = F.conv2d(d_input.abs(), mask.detach(), stride=self.stride, padding=self.padding).flatten(1).sum(1)
            mac = F.conv2d((d_input.detach()!=0).to(x), mask, stride=self.stride, padding=self.padding).flatten(1).sum(1)
            # dss = torch.einsum('oi, bihw->b',  mask.flatten(1).detach(), d_input.abs())
            # mac  = torch.einsum('oi, bis->b',  mask, (d_input.detach()!=0).to(x))
        else:
            input_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            input_unfold_split = torch.chunk(input_unfold, n_chunk, dim=0)
            d_input=(input_unfold_split[i_cur]-input_unfold_split[i_prv])
            dss = torch.einsum('ois, bis->b', mask.abs(), d_input.abs()).flatten(1)
            mac = torch.einsum('ois, bis->b', mask, (d_input.detach()!=0).to(x))

        if torch.isnan(dss).sum()>0:
            print('torch.isnan(expand_d).sum()>0:')
            
        with torch.no_grad():
            if self.settings.kernel_mode in ['conv_nan', 'lrlc_nan', 'lc_nan']:
                dss_stats = (dss, mac, 1.0, 1.0)
            elif self.settings.kernel_mode in ['lrlc_lrlc', 'lc_lc']:
                dss_stats = (dss, mac, torch.count_nonzero(mask.detach()).item(), torch.numel(mask))
            else:
                dss_stats = (dss, mac, np.prod([*self.map_size])*torch.count_nonzero(mask.detach()).item(), np.prod([*self.map_size])*torch.numel(mask))

        return dss_stats    
    
    def dropout(self, x):
        if self.drop_rate>0:
            x = F.dropout(x, p=0.5, training=self.training)  
        return x
    
    def flatten(self, x):
        if self.add_flat:
            x = torch.flatten(x, start_dim=1).unsqueeze(-1).unsqueeze(-1)
        return x

    def updatemp(self, dx):
        if len(self.mp)==0:
            self.mp = dx
            self.mp_lft =  torch.zeros_like(dx)
        else:
            self.mp = self.mp + dx

    def forward(self, inputs):
        if self.training or 1:
            return self.forward_train(inputs)
        else:
            return self.forward_test(inputs)

    def reset_mask(self):
        if (self.w_mask is not None) and self.training:
            mask_weight(self.w.data, self.w_mask, mask_th=self.mask_th)
            
    def forward_train(self, x):
        # self.reset_mask()
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.round(x, self.th, self.training)
        x = self.pool(x)
        with torch.no_grad():
            self.max_x = x.abs().max()
            # if self.max_x==0:
            #     print('DSSConv2d max act is zero')
        return x

    def forward_test(self, dx):
        dx = self.flatten(dx)        
        dx = self.conv2d(dx, fuse_bn=True)
        self.updatemp(dx)
        do = self.act(self.mp)-self.act(self.mp_lft)
        delta = self.round(do, self.th, self.training)            
        epsilon = do - delta
        self.mp_lft = self.mp - epsilon

        return delta

class DSSInvolution(DSSLayer):
    def __init__(self, channels, kernel_size=3, stride=1, settings=[]):
        super(DSSInvolution, self).__init__(channels, settings=settings)
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 1
        self.groups = self.channels // self.group_channels
        self.conv1 = conv1x1(channels, channels // reduction_ratio, settings=settings)
        self.conv2 = conv1x1(channels // reduction_ratio, kernel_size**2 * self.groups, bn=False, activation='I', settings=settings)
        self.mask       =  lambda x: TernarizerHard(x).apply(x)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        
        # self.bn = nn.BatchNorm2d(self.channels)
        self.act = nn.ReLU(inplace=True)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        # https://github.com/hszhao/SAN/blob/d88b022fc3ec4920d4596f94983f0e5b1ced62c6/lib/sa/functions/aggregation_zeropad.py
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        self.map_size = torch.Size([self.groups, self.group_channels, self.kernel_size**2, h, w])
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)

        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        # out = self.bn(out)
        out = self.act(out)
        out = self.round(out, self.th, self.training)
        with torch.no_grad():
            self.max_x = out.abs().max()
            if self.max_x==0:
                print('DSSInvolution max act is zero')

        if torch.isnan(out).sum()>0 or out.max()==np.Inf:
            print('torch.isnan(x).sum()>0:')
            # out = self.round(out_, self.th)
        return out

    def get_exp(self, input):
        def bn_stats(x):
            return x.mean([0, 2, 3]), x.var([0, 2, 3], unbiased=False)

        x = input[0]
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)

        mask = self.mask(weight)
        input_unfold_split = torch.chunk(out, 2, dim=0)
        mask_split = torch.chunk(mask, 2, dim=0)

        dss = (mask_split[0]*input_unfold_split[0] - mask_split[1]*input_unfold_split[1]).abs()
        # dss = torch.einsum('oi, bis->b',  mask_split[0].abs(), input_unfold_split[0].sub(input_unfold_split[1]).abs())
        with torch.no_grad():
            mac = torch.count_nonzero(dss)/(b//2)/1e6
        dss = dss.flatten(1).sum(1)

        # expaned_x = mask * (self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w))
        mask_info = (torch.count_nonzero(mask.detach()).item(), torch.numel(mask), mac)
        return dss, mask_info


class DSSAlign(DSSLayer):
    def __init__(self, channels, kernel_size=3, stride=1, expantion_ratio=[1,1], settings=[]):
        super(DSSAlign, self).__init__(channels, settings=settings)
        inner_dim = channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.inner_dim = inner_dim
        self.to_q = DSSConv2d(channels, inner_dim, [1,1], 1, activation='I' , expantion_ratio=expantion_ratio, settings=settings)
        self.to_k = DSSConv2d(channels, inner_dim, [1,1], 1, activation='I' , expantion_ratio=expantion_ratio, settings=settings)
        # self.to_q = conv1x1(channels, inner_dim, bn=False, activation='I', settings=settings)
        # self.to_k = conv1x1(channels, inner_dim, bn=False, activation='I', settings=settings)

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def att_main(self, x):
        q = torch.chunk(self.to_q(x), 2, dim=0)
        k = torch.chunk(self.to_k(x), 2, dim=0)
        
        x_prv, x_cur = torch.chunk(x, 2, dim=0)
        b, c, h, w = x_prv.shape
        q = self.unfold(q[0]).view(b, self.channels, self.kernel_size**2, h, w)
        k = self.unfold(k[1]).view(b, self.channels, self.kernel_size**2, h, w)
        v = self.unfold(x_prv).view(b, self.channels, self.kernel_size**2, h, w)
        
        return x_prv, x_cur, q, k, v

    def forward(self, x):
        x_prv, x_cur, q, k, v = self.att_main(x)
        b, c, h, w = x_prv.shape
        sim = torch.einsum('b i k h w, b i l h w -> b k l h w', q, k)
        attn = sim.softmax(dim = 1)
        x_prv_warped = torch.einsum(' b k l h w, b i l h w -> b i k h w', attn, v).sum(dim=2)
        x_prv_warped = self.round(x_prv_warped, self.th, self.training)

        self.map_size = x_prv_warped.shape[2:]
        return torch.cat([x_prv, x_cur, x_prv_warped], dim=0)

    def get_exp(self, input):
        # We need to compute DSS for two module 1) Attention 2) Warp
        x = input[0]
        x_prv, x_cur, q, k, v = self.att_main(x)
        b, c, h, w = x_prv.shape
        sim = torch.einsum('b i k h w, b i l h w -> b k l h w', q, k)
        attn = sim.softmax(dim = 1)

        mask = self.binalize(attn, self.mask_th)
        x_prv_warped = attn.mul(v).sum(dim=1)

        dss = mask.detach()*((x_cur - x_prv_warped).abs()).flatten(1).sum(1)
        mac  = mask.flatten(1).sum(1)

        dss_stats = (dss, mac, torch.count_nonzero(mask.detach()).item(), torch.numel(mask))

        return dss_stats    
