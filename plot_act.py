from numpy.core.fromnumeric import nonzero
from numpy.lib.function_base import delete
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.dss_util import round_muldiv, round_divmul, Thresh, floor_muldiv, floor_divmul
import matplotlib.pyplot as plt
import math


class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big # Thanks to @haolibai 
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        return grad_weight, grad_alpha, None, None, None


def plot_tanhshrink():
    v = torch.linspace(-2, 2, 2000)
    v = v.detach()
    lambd = 5e-3

    scale = 0.5/lambd

    m1 = nn.Tanhshrink()
    m2 = lambda x: F.softshrink(x, lambd=0.5)
    y1 = m1(v)
    y2 = m2(v)

    fig = plt.figure(21, figsize=(32,32),dpi=300)
    ax = plt.subplot(111)

    ax.plot(v, y1.squeeze().detach().numpy(), label=r'$\bar{v}$', color='g', linewidth=1.5)
    ax.plot(v, y2.squeeze().detach().numpy(), label=r'$\bar{v}$', color='r', linewidth=1.5)

    ax.grid()
    # plt.ylim([-3,7])
    ax.axis('equal')
    fig.savefig('resources/tanhshrink.png')
    # fig.savefig('run_act_test/HSwish.pdf')



def plot_act1():
    act_name = 'Swish'
    act_swish = get_act('swish', 1)
    act_hswish = get_act('hswish', 1)

    v = torch.linspace(-4, 7, 2000)
    v0 = v.clone()
    v.requires_grad=True

    s = torch.tensor([1.0/4.0])
    s.requires_grad=True

    v_bar = Thresh.apply(v, s)

    # Swish
    v_hat_swish = act_swish(v_bar).squeeze()
    func_act_swish =  lambda x: act_swish(Thresh.apply(x, s)).squeeze()
    jac_swish = [torch.autograd.grad(func_act_swish(v_), s)[0] for v_ in v]
    jac_swish = torch.cat(jac_swish)
    
    # H-Swish
    v_hat_hswish = act_hswish(v_bar).squeeze()
    func_act_hswish =  lambda x: act_hswish(Thresh.apply(x, s)).squeeze()
    jac_hswish = [torch.autograd.grad(func_act_hswish(v_), s)[0] for v_ in v]
    jac_hswish = torch.cat(jac_hswish)

    fig = plt.figure(21, figsize=(32,32),dpi=300)
    ax = plt.subplot(111)

    ax.plot(v0, v_bar.squeeze().detach().numpy(), label=r'$\bar{v}$', color='g', linewidth=1.5)
    ax.plot(v0, v_hat_swish.squeeze().detach().numpy(), label=r'$\hat{v}$(Swish)', color='r', linewidth=1.5)
    ax.plot(v0, v_hat_hswish.squeeze().detach().numpy(), label=r'$\hat{v}$(H-Swish)', color='m', linewidth=1.5)
    ax.plot(v0, jac_swish.squeeze().detach().numpy(), label=r'$\frac{\partial \hat{v}}{\partial s}$(Swish)', color='b', linewidth=1.5)
    ax.plot(v0, jac_hswish.squeeze().detach().numpy(), label=r'$\frac{\partial \hat{v}}{\partial s}$(H-Swish)', color='c', linewidth=1.5)
    ax.set_xlabel(r'$v$', fontsize=38)
    ax.set_ylabel(r'$\bar{v},\hat{v}, \frac{\partial \hat{v}}{\partial s}$', fontsize=38)
    ax.legend(loc='best', fontsize=38)
    ax.grid()
    # plt.ylim([-3,7])
    ax.axis('equal')
    fig.savefig('run_act_test/(H)Swish(2).png')
    # fig.savefig('run_act_test/HSwish.pdf')


def plot_act2():
    fontsize = 15
    Qp = 9.0
    Qn = 0.0
    act_name = 'relu_'
    # act = get_act('relu')
    act  = lambda x: F.relu(x, inplace=False)

    # act = FunLSQ.apply(v_i, th, g, 0, 6)

    g = 1.0 / math.sqrt(1 * Qp*0.1)

    v = torch.linspace(-2, Qp, 2000)
    v0 = v.clone()
    v.requires_grad=True
    
    use_log = False

    th_val = 9.0/5.0

    if use_log:
        th1 = torch.tensor([np.log(th_val)]) 
        th2 = torch.tensor([np.log(1/th_val).item()]) 
    else:
        th1 = torch.tensor([th_val]) 
        th2 = torch.tensor([1/th_val]) 

    th1.requires_grad=True
    th2.requires_grad=True
    ax = plt.subplot(111)


    dv__ds_DSS1 = [torch.autograd.grad(round_divmul(act(v_i), th1, round_only=True, use_log=use_log), th1)[0] for v_i in v]
    dv__ds_LSQ1 = [torch.autograd.grad(round_divmul(act(v_i), th1, round_only=False, use_log=use_log), th1)[0] for v_i in v]
    dv__ds_LSQ1 = torch.cat(dv__ds_LSQ1)
    dv__ds_DSS1 = torch.cat(dv__ds_DSS1)

    dv__ds_DSS2 = [torch.autograd.grad(round_muldiv(act(v_i), th2, round_only=True, use_log=use_log), th2)[0] for v_i in v]
    dv__ds_LSQ2 = [torch.autograd.grad(round_muldiv(act(v_i), th2, round_only=False, use_log=use_log), th2)[0] for v_i in v]
    dv__ds_LSQ2 = torch.cat(dv__ds_LSQ2)
    dv__ds_DSS2 = torch.cat(dv__ds_DSS2)

    v_hat = round_divmul(act(v.detach()), th1, round_only=False, use_log=use_log)

    fig = plt.figure(figsize=(9, 12))
    ax = plt.subplot(111)
    ax.set_xlim([-3,7])
    ax.set_ylim([-5,6.5])
    ax.plot(v0.detach().numpy(),  (v0.detach()).numpy(), label=r'$x$', color='k', linewidth=1.3, linestyle='-.')
    ax.plot(v0.detach().numpy(), v_hat.detach().numpy(), label=r'$\hat{x}$', color='b', linewidth=1.5)
    ax.plot(v0.detach().numpy(), dv__ds_LSQ1.detach().numpy(), label=r'$\frac{\partial \hat{x}}{\partial s}$(LSQ)', color='r', linewidth=1.5,linestyle='dashed')
    ax.plot(v0.detach().numpy(), dv__ds_DSS1.detach().numpy(), label=r'$\frac{\partial \hat{x}}{\partial s}$(MG)', color='g', linewidth=1.5,linestyle='dashed')
    # ax.plot(v0.detach().numpy(), dv__ds_LSQ2.detach().numpy(), label=r'$\frac{\partial \hat{x}}{\partial s}$(LSQ, mul->div)', color='r', linewidth=1.3)
    # ax.plot(v0.detach().numpy(), dv__ds_DSS2.detach().numpy(), label=r'$\frac{\partial \bar{x}}{\partial s}$(mul->div)', color='g', linewidth=1.3)
    
    ax.set_xlabel(r'$x$', fontsize=fontsize)
    ax.set_ylabel(r'$x, \hat{x}, \frac{\partial \hat{x}}{\partial s}$', fontsize=fontsize)
    # ax.set_ylabel(r'$v, \bar{v}, \hat{v}, \frac{\partial \hat{v}}{\partial s},\frac{\partial \bar{v}}{\partial s}$', fontsize=fontsize)
    ax.legend(loc='upper left', fontsize=fontsize)
    ax.grid()
    ax.axis('equal')

    fig.savefig('resources/' + 'LSQ_vs_DSS_log_' + str(use_log) + '.png')
    fig.savefig('resources/' + 'LSQ_vs_DSS_log_' + str(use_log) + '.pdf')


def plot_act3():
    fontsize = 15
    Qp = 9.0
    Qn = 0.0
    act_name = 'relu_'
    # act = get_act('relu')
    act  = lambda x: F.relu(x, inplace=False)

    g = 1.0 / math.sqrt(1 * Qp*0.1)

    v = torch.linspace(-2, Qp, 2000)
    v0 = v.clone()
    v.requires_grad=True
    
    use_log = False

    th_val = 9.0/5.0

    if use_log:
        th1 = torch.tensor([np.log(th_val)]) 
        th2 = torch.tensor([np.log(1/th_val).item()]) 
    else:
        th1 = torch.tensor([th_val]) 
        th2 = torch.tensor([1/th_val]) 

    th1.requires_grad=True
    th2.requires_grad=True
    ax = plt.subplot(111)


    dv__ds_DSS1 = [torch.autograd.grad(round_divmul(act(v_i), th1, round_only=True, use_log=use_log), th1)[0] for v_i in v]
    dv__ds_LSQ1 = [torch.autograd.grad(round_divmul(act(v_i), th1, round_only=False, use_log=use_log), th1)[0] for v_i in v]
    dv__ds_LSQ1 = torch.cat(dv__ds_LSQ1)
    dv__ds_DSS1 = torch.cat(dv__ds_DSS1)

    dv__ds_DSS2 = [torch.autograd.grad(floor_divmul(act(v_i), th2, round_only=True, use_log=use_log), th2)[0] for v_i in v]
    dv__ds_LSQ2 = [torch.autograd.grad(floor_divmul(act(v_i), th2, round_only=False, use_log=use_log), th2)[0] for v_i in v]
    dv__ds_LSQ2 = torch.cat(dv__ds_LSQ2)
    dv__ds_DSS2 = torch.cat(dv__ds_DSS2)

    v_hat1 = round_divmul(act(v.detach()), th1, round_only=False, use_log=use_log)
    v_hat2 = floor_divmul(act(v.detach()), th1, round_only=False, use_log=use_log)

    fig = plt.figure(figsize=(9, 12))
    ax = plt.subplot(111)
    ax.set_xlim([-3,7])
    ax.set_ylim([-5,6.5])
    ax.plot(v0.detach().numpy(),  (v0.detach()).numpy(), label=r'$x$', color='k', linewidth=1.3, linestyle='-.')
    ax.plot(v0.detach().numpy(), v_hat1.detach().numpy(), label=r'$\hat{x}1$', color='b', linewidth=1.5,linestyle='dashed')
    ax.plot(v0.detach().numpy(), v_hat2.detach().numpy(), label=r'$\hat{x}2$', color='b', linewidth=1.5)
    ax.plot(v0.detach().numpy(), dv__ds_LSQ1.detach().numpy(), label=r'$\frac{\partial \hat{x}}{\partial s}$(LSQ)', color='r', linewidth=1.5,linestyle='dashed')
    ax.plot(v0.detach().numpy(), dv__ds_DSS1.detach().numpy(), label=r'$\frac{\partial \hat{x}}{\partial s}$(MG)', color='g', linewidth=1.5,linestyle='dashed')
    ax.plot(v0.detach().numpy(), dv__ds_LSQ2.detach().numpy(), label=r'$\frac{\partial \hat{x}}{\partial s}$(LSQ-f)', color='r', linewidth=1.3)
    ax.plot(v0.detach().numpy(), dv__ds_DSS2.detach().numpy(), label=r'$\frac{\partial \bar{x}}{\partial s}$(MG-f)', color='g', linewidth=1.3)
    
    ax.set_xlabel(r'$x$', fontsize=fontsize)
    ax.set_ylabel(r'$x, \hat{x}, \frac{\partial \hat{x}}{\partial s}$', fontsize=fontsize)
    # ax.set_ylabel(r'$v, \bar{v}, \hat{v}, \frac{\partial \hat{v}}{\partial s},\frac{\partial \bar{v}}{\partial s}$', fontsize=fontsize)
    ax.legend(loc='upper left', fontsize=fontsize)
    ax.grid()
    ax.axis('equal')

    fig.savefig('resources/' + 'LSQ_vs_DSS_flooe_' + str(use_log) + '.png')
    fig.savefig('resources/' + 'LSQ_vs_DSS_flooe_' + str(use_log) + '.pdf')


def plot_act4():
    fontsize = 15
    Qp = 6.0
    Qn = 0.0
    # act = get_act('relu')

    v_i_3p3 = torch.tensor([2.3])
    v_i_3p0 = torch.tensor([2.4])

    th = torch.linspace(1/1024., 5.5, 2000)

    # ste_round(v_i, torch.tensor([3.0]))
    use_floor = True
    if use_floor:
        v_hat_3p3 = [floor_divmul(v_i_3p3, th_i, round_only=True, use_log=False) for th_i in th]
        v_hat_3p0 = [floor_divmul(v_i_3p0, th_i, round_only=True, use_log=False) for th_i in th]
    else:
        v_hat_3p3 = [round_divmul(v_i_3p3, th_i, round_only=True, use_log=False) for th_i in th]
        v_hat_3p0 = [round_divmul(v_i_3p0, th_i, round_only=True, use_log=False) for th_i in th]

    v_hat_3p3 = torch.cat(v_hat_3p3)
    v_hat_3p0 = torch.cat(v_hat_3p0)
    fig = plt.figure(figsize=(9, 12))
    # fig = plt.figure(figsize=(11, 13),dpi=200)
    ax = plt.subplot(111)
    ax.set_xlim([-0.1,5.5])
    # ax.set_ylim([-5,6.5])
    ax.plot(th.detach().numpy(), v_hat_3p3.detach().numpy(), label=r'$\hat{x}_{2.3}$', color='c', linewidth=1.5)
    ax.plot(th.detach().numpy(), v_hat_3p0.detach().numpy(), label=r'$\hat{x}_{2.4}$', color='m', linewidth=1.5)
    ax.plot(th.detach().numpy(), v_hat_3p3.sub(v_hat_3p0).detach().numpy(), label=r'$\hat{x}_{2.3}-\hat{x}_{2.4}$', color='gray', linewidth=1.5, linestyle='dashed')

    ax.set_xlabel(r'$s$', fontsize=fontsize)
    ax.set_ylabel(r'$\hat{x}, \Delta \hat{x}$', fontsize=fontsize)
    ax.legend(loc='upper left', fontsize=fontsize)
    ax.grid()
    ax.axis('equal')
    fig.savefig('resources/' + 's_vs_vhat_round.png')
    fig.savefig('resources/' + 's_vs_vhat_round.pdf')

if __name__ == "__main__":
    # plot_act1()
    # plot_act2()
    # plot_act3()
    # plot_act4()
    plot_tanhshrink()
