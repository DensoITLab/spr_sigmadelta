import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import math
import pdb
from torch.nn.init import _calculate_correct_fan, calculate_gain
Qp = 6.0

def masked_kaiming_normal_(tensor, a=0, expantion_ratio=[1.0, 1.0], mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    # gain = 1.0
    std = gain / math.sqrt(fan/np.prod(expantion_ratio))
    with torch.no_grad():
        return tensor.normal_(0, 1), std

def masked_kaiming_uniform_(tensor, a=0, expantion_ratio=[1.0, 1.0], mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan/np.prod(expantion_ratio))
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        # return tensor.uniform_(-bound, bound), bound
        return tensor.uniform_(-1, 1), bound
# Mask neurons to keep initial MAC same as original network
def compute_mask(w_data, expantion_ratio):
    if (expantion_ratio[0]>1.0 or expantion_ratio[1]>1.0):
        mask_ratio = 1-1.0/(np.prod(expantion_ratio))
        w_mask = torch.rand_like(w_data).le(mask_ratio)
    else:
        w_mask = None
    return w_mask

def init_weights_uniform(w_data, nonlinearity='relu',  mode='fan_in',  expantion_ratio=[1.0, 1,0], base_th=1e-2):
    # expantion_ratio=[1.0, 1.0]
    with torch.no_grad():
        w_data, bound = masked_kaiming_uniform_(w_data, expantion_ratio=expantion_ratio, nonlinearity=nonlinearity,  mode=mode)
    
    mask_th = base_th*bound
    w_data*=(bound+mask_th)
    w_mask =  compute_mask(w_data, expantion_ratio)
    mask_weight(w_data, w_mask, mask_th=mask_th)
    return mask_th, w_mask

def init_weights_normal(w_data, nonlinearity='relu',  mode='fan_in',  expantion_ratio=[1.0, 1,0], mask_th=1e-2):
    expantion_ratio=[1.0, 1.0]
    with torch.no_grad():
        w_data, std = masked_kaiming_normal_(w_data, nonlinearity=nonlinearity,  mode=mode)
    w_data*=(std+mask_th)
    w_mask =  compute_mask(w_data, expantion_ratio)
    mask_weight(w_data, w_mask,  mask_th=mask_th)
    return w_mask

def mask_weight(w_data, w_mask, mask_th=5e-3):
    if w_mask is not None:
        w_unif = torch.rand_like(w_data).mul(2.).sub(1.).mul(mask_th)
        w_data[w_mask]=w_unif[w_mask]

def sine_init(w_data):
    with torch.no_grad():
        num_input = w_data.size(1)
        # See supplement Sec. 1.5 for discussion of factor 30
        w_data.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(w_data):
    with torch.no_grad():
        num_input = w_data.size(1)
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        w_data.uniform_(-1 / num_input, 1 / num_input)


########################################################################
# Helper class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def sat_ratio(x):
    with torch.no_grad():
        return (torch.count_nonzero(x.eq(6).detach())/x.numel()).item()

def compute_act_shape(input_shape, kernel_size, stride, padding=0, offs=0):
    output_width = (input_shape[0] - kernel_size + 2 * padding) // stride + 1 +offs
    output_hight = (input_shape[1] - kernel_size + 2 * padding) // stride + 1 +offs
    return (output_width, output_hight)


########################################################################
# MLCLayer specific Loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def compute_sparcity(model):
    masks = []
    with torch.no_grad():
        for module in model.layers:
            if hasattr(module, 'm') and module.m!=None:
                masks.append(module.get_mask())
            elif hasattr(module, 'kernel_mode') and module.kernel_mode in ['conv_conv', 'lc_lc', 'lrlc_lrlc']:
                masks.append(module.get_mask())

    numer = 0.0
    denom = 0.0
    for mask in masks:
        numer = numer + torch.count_nonzero(mask).item()
        denom = denom + mask.numel()
    if len(masks)==0:
        return 0
    else:
        return 1.0-numer/denom


# https://github.com/arunmallya/piggyback/blob/5a6094c45896c035a690d6f2fac0b102df176600/src/modnets/layers.py
DEFAULT_LAMBD = 5e-3
class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""
    @staticmethod
    def forward(ctx, inputs, lambd=DEFAULT_LAMBD):
        ctx.save_for_backward(inputs)
        return inputs.abs().gt(lambd).to(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        # pdb.set_trace()
        grad_output = (inputs[0].sgn()).to(inputs[0])*grad_output
        # grad_output = (inputs[0].sgn()*grad_output.sgn()).to(inputs[0])
        return grad_output, None

# https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html?highlight=shrink#torch.nn.Tanhshrink
class Ternarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, lambd=DEFAULT_LAMBD):
        idx_masked = inputs.abs().le(lambd)
        ctx.save_for_backward(inputs, idx_masked)
        ctx.lambd = lambd
        return F.softshrink(inputs, lambd=lambd).sgn().to(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, idx_masked = ctx.saved_tensors
        if 0:
            lambd = ctx.lambd
            # print(grad_output[idx_masked])
            grad_output[idx_masked]*=(1/lambd*inputs[idx_masked].abs())
            # print(grad_output)
        else:
            pdb.set_trace()
            grad_output[~idx_masked]=1
            # print(grad_output[idx_masked].abs().max())
            pass
            # grad_output[idx_masked]=0
        return grad_output, None

class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, lambd=DEFAULT_LAMBD):
        idx_masked = inputs.abs().le(lambd)
        ctx.save_for_backward(inputs, idx_masked)
        ctx.lambd = lambd
        # return F.hardshrink(inputs, lambd=lambd)
        return F.softshrink(inputs, lambd=lambd)

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output[idx_masked])
        inputs, idx_masked = ctx.saved_tensors
        if 1:
            lambd = ctx.lambd
            grad_output[idx_masked]*=(1/lambd*inputs[idx_masked].abs())
            # grad_output[idx_masked]*=0.001
            # grad_output[idx_masked]*=(0.0/lambd*inputs[idx_masked].abs())
        else:
            # pass
            grad_output[idx_masked]=0
        return grad_output, None

class TernarizerHard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs.sgn().to(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
 
def squash(m):
    return F.softsign(m)

def none_func(x):
    return x
    
def none_func2(x, other):
    return x

def none_func3(x, other, other2):
    return x
########################################################################
# Quantization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
def round_func(quantizer, max_scale=1024, min_scale=0):
    if quantizer=='MG_divmul_lin':
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=True, round_only=True, use_log=False)
    elif quantizer=='MG_muldiv_lin':
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=False, round_only=True, use_log=False)
    elif quantizer=='MG_divmul_log':
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=True, round_only=True, use_log=True)
    elif quantizer=='MG_muldiv_log':
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=False, round_only=True, use_log=True)
    elif quantizer in ['floor']:
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=False, round_only=True, use_log=False, use_floor=True)
    elif quantizer in ['floor_inv']:
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=False, round_only=True, use_log=False, use_floor=True)
    elif quantizer=='floor_log':
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=True, round_only=True, use_log=True, use_floor=True)
    elif quantizer=='floor_inv_log':
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=False, round_only=True, use_log=True, use_floor=True)
    elif quantizer in ['ULSQ_muldiv_lin']:
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=True, round_only=False, use_log=False)
    elif quantizer in ['ULSQ_divmul_log']:
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=True, round_only=False, use_log=True)
    elif quantizer in ['ULSQ_divmul_lin']:
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=False, round_only=False, use_log=False)
    elif quantizer in ['ULSQ_muldiv_log']:
        return lambda x, s, t: round(x, s, t, max_scale=max_scale, min_scale=min_scale, div_first=False, round_only=False, use_log=True)
    elif quantizer in ['nan']:
        return lambda x, s, t: none_func3(x, s, t)  
    else:
        print('Not implemented round_func')

# https://github.com/hustzxd/LSQuantization/blob/master/lsq.py
def ste_round_(x):
    return x + (x.round() - x).detach()

def ste_floor_(x):
    return x + (x.floor() - x).detach()

def grad_scale(s, scale):
    s_ = s
    s_grad = s * scale
    return s_.detach() - s_grad.detach() + s_grad

def round_prep(x, s, max_scale, min_scale, use_log, use_gradscale, div_first):
    if use_log:
        s0 = np.log(max_scale-min_scale)
        if div_first:
            s_ = s.sub(s0).exp()
        else:
            s_ = s.add(s0).exp()
    else:
        s_ = s.abs()

    if div_first:
        s_ = (s_.reciprocal().add(min_scale)).reciprocal()
    else:
        s_ = s_.add(min_scale)

    if use_gradscale:
        g = 1.0 / math.sqrt(x.numel() * Qp)
        return grad_scale(s_, g)
    else:
        return  s_

def round(x, s, training, max_scale, min_scale, round_only=False, use_log=False, use_gradscale=False, div_first=False, use_floor=False, add_noise=False):
    # if s.requires_grad==False:
    #     return x
    s_ = round_prep(x, s, max_scale, min_scale, use_log=use_log, use_gradscale=use_gradscale,  div_first=div_first)

    if add_noise and training:
        if div_first:
            x = x + (s_)*(torch.rand_like(x)-0.5)
        else:
            x = x + (s_.reciprocal())*(torch.rand_like(x)-0.5)

    if use_floor:
        ste_func = ste_floor_
    else:
        ste_func = ste_round_

    if div_first:
        if round_only:
            return ste_func(x.div(s_)).mul(s_.data)
        else:
            return ste_func(x.div(s_)).mul(s_)
    else:
        if round_only:
            return ste_func(x.mul(s_)).div(s_.data)
        else:
            return ste_func(x.mul(s_)).div(s_) 


########################################################################
# Batch Normalization fusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# https://discuss.pytorch.org/t/how-to-absorb-batch-norm-layer-weights-into-convolution-layer-weights/16412/3
# the fuse function code is refered from https://zhuanlan.zhihu.com/p/49329030
def fuse_conv(w, b, bn):
    if isinstance(bn, nn.BatchNorm2d):
        if bn.training:
            var_sqrt = torch.sqrt(bn.var + bn.eps)
            mean = bn.mean
        else:
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)
            mean = bn.running_mean

        beta = bn.weight
        gamma = bn.bias
        if b is None:
            b = mean.new_zeros(mean.shape)
        w = w * (beta / var_sqrt).reshape([w.shape[0], 1, 1, 1])
        b = (b - mean)/var_sqrt * beta + gamma
    return w, b

def fuse_lc(w, b, bn):
    if isinstance(bn, nn.BatchNorm2d):
        if bn.training:
            var_sqrt = torch.sqrt(bn.var + bn.eps)
            mean = bn.mean
        else:
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)
            mean = bn.running_mean

        beta = bn.weight
        gamma = bn.bias
        if b is None:
            b = mean.new_zeros(mean.shape)
        
        w = w * (beta / var_sqrt).reshape([w.shape[0], 1, 1])
        b = (b - mean)/var_sqrt * beta + gamma
    return w, b

def fuse_aggregation(w,  bn, c_x, c_w):
    if bn.training:
        var_sqrt = torch.sqrt(bn.var + bn.eps)
        mean = bn.mean
    else:
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        mean = bn.running_mean

    b = mean.new_zeros(mean.shape)
    beta = bn.weight        
    gamma = bn.bias

    w = w * (beta / var_sqrt).reshape([1, -1, c_w, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    return w, b

def fuse_involution(w, bn, shared_dim):
    if bn.training:
        var_sqrt = torch.sqrt(bn.var + bn.eps).view([1, -1, shared_dim, 1, 1, 1])
        mean = bn.mean.view([1, -1, shared_dim, 1, 1, 1])
    else:
        var_sqrt = torch.sqrt(bn.running_var + bn.eps).view([1, -1, shared_dim, 1, 1, 1])
        mean = bn.running_mean.view([1, -1, shared_dim, 1, 1, 1])

    gamma = bn.bias.view([1, -1, shared_dim, 1, 1, 1])
    beta = bn.weight.view([1, -1, shared_dim, 1, 1, 1])

    b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt)
    b = -mean/var_sqrt * beta + gamma
    return w, b

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    return fused_conv
