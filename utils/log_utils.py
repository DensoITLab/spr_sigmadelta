import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

class Struct(object): pass

def plot_stats(log, plot_supp=False):
    # from matplotlib import rc
    # rc('text', usetex=True)
    import matplotlib
    matplotlib.rcParams.update({'font.size': 15})
    if plot_supp:
        plt.figure(21, figsize=(40,16),dpi=200)
    else:
        plt.figure(21, figsize=(36,20),dpi=200)
        
    if log['dataset_name'] == 'NMNIST':
        CRT_MIN, CRT_MAX = 0, 0.04
        TGT_MIN, TGT_MAX = 0, 0.08
        DSS_MIN, DSS_MAX = 0, 0.03
        FLOPS_MIN, FLOPS_MAX = 0, 5
        THR_MIN, THR_MAX = 0, 1.5
    else:
        CRT_MIN, CRT_MAX = 0., 0.10
        TGT_MIN, TGT_MAX = 2*CRT_MIN, 2*CRT_MAX
        DSS_MIN, DSS_MAX = 0, 0.002
        FLOPS_MIN, FLOPS_MAX = 0, 5
        SPR_MIN, SPR_MAX = 0, 100.0

    THR_MIN, THR_MAX = 0, 1200
    SPR_MIN, SPR_MAX = 0, 100.0
    BIT_MIN, BIT_MAX = 0, 16

    fontsize0=15
    fontsize1=14

    if plot_supp:
        base_layout = 250
    else:
        base_layout = 230

    ax1 = plt.subplot(base_layout+1)
    plt.plot(log['train']['epoch'], log['train']['dss_criteria'], label="$Train$", color='r', linewidth=0.5)
    plt.plot(log['train']['epoch'], log['val']['dss_criteria'], label="$Val$", color='b', linewidth=0.5)
    plt.plot(log['train']['epoch'], log['test']['dss_criteria'], label="$Test$", color='g', linewidth=0.5)
    plt.legend(loc='lower right', fontsize=fontsize0)
    plt.ylim((CRT_MIN, CRT_MAX))
    plt.ylabel('Error', fontsize=fontsize1)
    if  plot_supp:
        suffix = ', best is %.4f at epock %d' %(log['best_test']['dss_criteria'], log['best']['epoch'])
    else:
        suffix = ''
    ax1.title.set_text('(a) Error' + suffix)

    # plt.plot(log['train']['epoch'], log['train']['criteria'], label="$Train$", color='r', linewidth=0.5)
    # plt.plot(log['train']['epoch'], log['val']['criteria'], label="$Val$", color='b', linewidth=0.5)
    # plt.plot(log['train']['epoch'], log['test']['criteria'], label="$Test$", color='g', linewidth=0.5)
    # plt.legend(loc='lower right', fontsize=fontsize0)
    # plt.ylim((CRT_MIN, CRT_MAX))
    # plt.ylabel('Error', fontsize=fontsize1)
    # if  plot_supp:
    #     suffix = ', best is %.4f at epock %d' %(log['best_test']['criteria'], log['best']['epoch'])
    # else:
    #     suffix = ''
    # ax1.title.set_text('(a) Error' + suffix)
    
    ax2 = plt.subplot(base_layout+2)
    # plt.plot(log['train']['epoch'], log['train']['DSS_weight'], color='b', linewidth=0.5)
    plt.plot(log['train']['epoch'], log['train']['DSS_weight'], color='b', linewidth=0.5)
    plt.ylabel('DSS Weight', fontsize=fontsize1)
    ax2.title.set_text('(b) Weight for DSS')

    ax3 = plt.subplot(base_layout+3)
    scale = np.stack(log['train']['scale'], axis=1)
    n_layer = scale.shape[0]
    if n_layer>10:
        [plt.plot(log['train']['epoch'], np.abs(scale[idx,:]), linewidth=0.5) for idx in range(n_layer)]
    else:
        [plt.plot(log['train']['epoch'], np.abs(scale[idx,:]), label='layer ' + str(idx), linewidth=0.5) for idx in range(n_layer)]
        plt.legend(loc='upper right', fontsize=fontsize0)
    # plt.yscale('log')
    plt.ylim((THR_MIN, THR_MAX))
    plt.ylabel('s', fontsize=fontsize1)
    ax3.title.set_text('(c) Inverse quantization step size')

    ax4 = plt.subplot(base_layout+4)
    plt.plot(log['train']['epoch'], log['train']['dss_loss'], label="$Train$", color='r', linewidth=0.5)
    plt.plot(log['train']['epoch'], log['val']['dss_loss'], label="$Val$", color='b', linewidth=0.5)
    plt.plot(log['train']['epoch'], log['test']['dss_loss'], label="$Test$", color='g', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=fontsize0)
    plt.ylim((DSS_MIN, DSS_MAX))
    plt.ylabel('DSS', fontsize=fontsize1)
    ax4.title.set_text('(d) DSS/DAM loss')

    ax5 = plt.subplot(base_layout+5)
    plt.plot(log['train']['epoch'], log['train']['mac'], label="$Train$", color='r', linewidth=0.5)
    plt.plot(log['train']['epoch'], log['val']['mac'], label="$Val$", color='b', linewidth=0.5)
    plt.plot(log['train']['epoch'], log['test']['mac'], label="$Test$", color='g', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=fontsize0)
    plt.ylim((FLOPS_MIN, FLOPS_MAX))
    plt.ylabel('MAC', fontsize=fontsize1)    
    if  plot_supp:
        suffix = ', best is %.4f at epock %d' %(log['best_test']['mac'],log['best']['epoch'])
    else:
        suffix = ''
    ax5.title.set_text('(e) MAC for update' + suffix)

    ax6 = plt.subplot(base_layout+6)
    plt.plot(np.array(log['train']['epoch']), 100*np.array(log['train']['w_spr']), label="$Train$", color='r', linewidth=0.5)
    plt.plot(np.array(log['train']['epoch']), 100*np.array(log['test']['w_spr']), label="$Val$", color='b', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=fontsize0)
    plt.ylim((SPR_MIN, SPR_MAX))
    plt.ylabel('Sparcity [%]', fontsize=fontsize1)
    if  plot_supp:
        suffix = ', best is %.4f at epock %d' %(log['best_test']['w_spr'], log['best']['epoch'])
    else:
        suffix = ''

    ax6.title.set_text('(f) Sparcity of synaptic connection' + suffix)
    
    if plot_supp:
        ax7 = plt.subplot(base_layout+7)
        plt.plot(log['train']['epoch'], log['train']['tgt_loss'], label="$Train$", color='r', linewidth=0.5)
        plt.plot(log['train']['epoch'], log['val']['tgt_loss'], label="$Val$", color='b', linewidth=0.5)
        plt.plot(log['train']['epoch'], log['test']['tgt_loss'], label="$Test$", color='g', linewidth=0.5)
        plt.legend(loc='lower right', fontsize=fontsize0)
        plt.ylim((TGT_MIN, TGT_MAX))
        plt.ylabel('L', fontsize=fontsize1)
        ax7.title.set_text('(g) Target Loss')

        ax8 = plt.subplot(base_layout+8)
        plt.plot(log['train']['epoch'], log['train']['total_loss'], label="$Train$", color='r', linewidth=0.5)
        plt.plot(log['train']['epoch'], log['val']['total_loss'], label="$Val$", color='b', linewidth=0.5)
        plt.legend(loc='lower right', fontsize=fontsize0)
        # plt.ylim((TGT_MIN, TGT_MAX))
        plt.ylabel('L', fontsize=fontsize1)
        ax8.title.set_text('(h) Total Loss')


        ax9 = plt.subplot(base_layout+9)
        nbit = np.stack(log['train']['bit'], axis=1)
        n_layer = nbit.shape[0]
        if n_layer>10:
            [plt.plot(log['train']['epoch'], nbit[idx,:], linewidth=0.5) for idx in range(n_layer)]
        else:
            [plt.plot(log['train']['epoch'], nbit[idx,:], label='layer ' + str(idx), linewidth=0.5) for idx in range(n_layer)]
            plt.legend(loc='lower left', fontsize=6)
        plt.ylim((BIT_MIN, BIT_MAX))
        plt.ylabel('max nbit')
        ax9.title.set_text('(h) Bit required')
    
    # plt.suptitle('Various Straight Lines',fontsize=20)
    fig = plt.gcf()
    if plot_supp:
        fig.savefig(os.path.join(log['exp_dir'],'b', log['exp_name'] + '.pdf'))
        shutil.copyfile(os.path.join(log['exp_dir'],'b', log['exp_name'] + '.pdf'), os.path.join(log['log_dir'], log['exp_name'] + '.pdf'))
    else:
        fig.savefig(os.path.join(log['exp_dir'],'a', log['exp_name'] + '.pdf'))
    plt.close()
 
# def plot_train_stats_PilotNet(log, plot_supp=False):
#     # from matplotlib import rc
#     # rc('text', usetex=True)
#     import matplotlib
#     matplotlib.rcParams.update({'font.size': 15})
#     if plot_supp:
#         plt.figure(21, figsize=(32,16),dpi=200)
#     else:
#         plt.figure(21, figsize=(32,20),dpi=200)
#     TGT_MIN, TGT_MAX = 0, 0.08
#     SLOW_MIN, SLOW_MAX = 0, 0.0008
#     FLOPS_MIN, FLOPS_MAX = 0, 5
#     ACC_MIN, ACC_MAX = 0., 0.03
#     SPR_MIN, SPR_MAX = 0, 100.0
#     THR_MIN, THR_MAX = 0, 3.0
#     BIT_MIN, BIT_MAX = 0, 16

#     fontsize0=15
#     fontsize1=14
    
#     if plot_supp:
#         base_layout = 240
#     else:
#         base_layout = 230
    
#     ax1 = plt.subplot(base_layout+1)
#     plt.plot(log['train']['epoch'], log['train']['dss_criteria'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['val']['dss_criteria'], label="$Val$", color='b', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test']['dss_criteria'], label="$Test$", color='g', linewidth=0.5)
#     plt.legend(loc='lower right', fontsize=fontsize0)
#     plt.ylim((ACC_MIN, ACC_MAX))
#     plt.ylabel('Accuracy', fontsize=fontsize1)
#     ax1.title.set_text('(a) Prediction error')
    
#     ax2 = plt.subplot(base_layout+2)
#     plt.plot(log['train']['epoch'], log['train']['DSS_weight'], color='b', linewidth=0.5)
#     plt.ylabel('DSS Weight', fontsize=fontsize1)
#     ax2.title.set_text('(b) Weight for DSS')

#     ax3 = plt.subplot(base_layout+3)
#     scale = np.stack(log['train']['scale'], axis=1)
#     n_layer = scale.shape[0]
#     [plt.plot(log['train']['epoch'], np.abs(scale[idx,:]), label='layer ' + str(idx), linewidth=0.5) for idx in range(n_layer)]
#     plt.legend(loc='upper right', fontsize=fontsize0)
#     plt.ylim((THR_MIN, THR_MAX))
#     plt.ylabel('Scale', fontsize=fontsize1)
#     ax3.title.set_text('(c) Quantization scald')

#     ax4 = plt.subplot(base_layout+4)
#     plt.plot(log['train']['epoch'], log['train_slow'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_slow'], label="$Test$", color='g', linewidth=0.5)
#     plt.legend(loc='upper right', fontsize=fontsize0)
#     plt.ylim((SLOW_MIN, SLOW_MAX))
#     plt.ylabel('DSS', fontsize=fontsize1)
#     ax4.title.set_text('(d) DSS loss')

#     ax5 = plt.subplot(base_layout+5)
#     plt.plot(log['train']['epoch'], log['train_mac'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_mac'], label="$Test$", color='g', linewidth=0.5)
#     plt.legend(loc='upper right', fontsize=fontsize0)
#     plt.ylim((FLOPS_MIN, FLOPS_MAX))
#     plt.ylabel('MAC', fontsize=fontsize1)
#     ax5.title.set_text('(e) MAC for update')

#     ax6 = plt.subplot(base_layout+6)
#     plt.plot(np.array(log['train']['epoch']),  100*np.array(log['w_spr']), color='r', linewidth=0.5)
#     plt.legend(loc='upper left', fontsize=fontsize0)
#     plt.ylim((SPR_MIN, SPR_MAX))
#     plt.ylabel('Sparcity [%]', fontsize=fontsize1)
#     ax6.title.set_text('(f) Sparcity of synaptic connection')
    
#     if plot_supp:
#         ax8 = plt.subplot(base_layout+8)
#         nbit = np.stack(log['train']['bit'],axis=1)
#         n_layer = nbit.shape[0]
#         [plt.plot(log['train']['epoch'], nbit[idx,:], label='l:' + str(idx), linewidth=0.5) for idx in range(n_layer)]
#         plt.legend(loc='lower left', fontsize=6)
#         plt.ylim((BIT_MIN, BIT_MAX))
#         plt.ylabel('max nbit')
#         ax8.title.set_text('(h) Bit required')


#     fig = plt.gcf()
#     if plot_supp:
#         fig.savefig(os.path.join(log['log_dir'], log['exp_name'] + '.pdf'))
#     else:
#         fig.savefig(os.path.join('../log', log['dataset_name'], 'summary' ,log['exp_name'] + '.pdf'))
#     plt.close()

# def plot_train_stats_mnist(log):
#     plt.figure(21, figsize=(32,16),dpi=200)
#     fontsize0=15
#     fontsize1=14

#     TGT_MIN, TGT_MAX = 0, 0.08
#     SLOW_MIN, SLOW_MAX = 0, 0.03
#     FLOPS_MIN, FLOPS_MAX = 0, 8
#     ACC_MIN, ACC_MAX = 0.95, 1.0
#     SPR_MIN, SPR_MAX = 0, 1.0
#     THR_MIN, THR_MAX = 0, 1200
#     BIT_MIN, BIT_MAX = 0, 16

#     plt.subplot(base_layout+241)
#     plt.plot(log['train']['epoch'], log['train_tgt'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_tgt'], label="$Val$", color='g', linewidth=0.5)
#     plt.legend(loc='lower left', fontsize=6)
#     plt.ylim((TGT_MIN, TGT_MAX))
#     plt.ylabel('Cross Entropy Loss')

#     plt.subplot(base_layout+242)
#     plt.plot(log['train']['epoch'], log['train_slow'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_slow'], label="$Val$", color='g', linewidth=0.5)
#     plt.legend(loc='upper left', fontsize=6)
#     plt.ylim((SLOW_MIN, SLOW_MAX))
#     plt.ylabel('DSS')
    
#     plt.subplot(base_layout+243)
#     plt.plot(log['train']['epoch'], log['w_spr'], label="$Train$", color='r', linewidth=0.5)
#     plt.legend(loc='upper left', fontsize=6)
#     plt.ylim((SPR_MIN, SPR_MAX))
#     plt.ylabel('Weight sparsity (best:' + str(np.array(log['w_spr']).max())+ ')')

#     plt.subplot(base_layout+244)
#     plt.plot(log['train']['epoch'], log['train_mac'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_mac'], label="$Val$", color='g', linewidth=0.5)
#     plt.legend(loc='lower left', fontsize=6)
#     plt.ylim((FLOPS_MIN, FLOPS_MAX))
#     plt.ylabel('MAC (best:' + str(np.array(log['test_mac']).min())+ ')')
    
#     plt.subplot(base_layout+245)
#     plt.plot(log['train']['epoch'], log['train_acc'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_acc'], label="$Val$", color='g', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['pred_acc'], label="$Test$", color='b', linewidth=0.5)
#     plt.legend(loc='lower left', fontsize=6)
#     plt.ylim((ACC_MIN, ACC_MAX))
#     plt.ylabel('ACC (best:' + str(np.array(log['pred_acc']).max()) + ')')

#     plt.subplot(base_layout+246)
#     thr = np.stack(log['thr'],axis=1)
#     n_layer = thr.shape[0]
#     [plt.plot(log['train']['epoch'], np.abs(thr[idx,:]), label='l:' + str(idx), linewidth=0.5) for idx in range(n_layer)]
#     plt.legend(loc='upper left', fontsize=6)
#     plt.ylim((THR_MIN, THR_MAX))
#     plt.ylabel('Thr')
        
#     plt.subplot(base_layout+247)
#     nbit = np.stack(log['bit'],axis=1)
#     n_layer = nbit.shape[0]
#     [plt.plot(log['train']['epoch'], nbit[idx,:], label='l:' + str(idx), linewidth=0.5) for idx in range(n_layer)]
#     plt.legend(loc='lower left', fontsize=6)
#     plt.ylim((BIT_MIN, BIT_MAX))
#     plt.ylabel('max nbit')

#     # plt.plot(log['train']['epoch'], log['sat'], color='b', linewidth=0.5)
#     # plt.ylabel('Saturation ratio')

#     plt.subplot(base_layout+248)
#     plt.plot(log['train']['epoch'], log['train']['DSS_weight'], color='b', linewidth=0.5)
#     plt.ylabel('DSS_weight')

#     fig = plt.gcf()
#     fig.savefig(os.path.join(log['log_dir'], log['exp_name'] + '.pdf'))
#     # fig.savefig(os.path.join(log['log_dir'], log['exp_name'] + '_train(s).pdf'))
#     plt.close()
 
 
# def plot_train_stats_drive(log):
#     plt.figure(21, figsize=(32,16),dpi=200)

#     TGT_MIN, TGT_MAX = 0, 0.08
#     SLOW_MIN, SLOW_MAX = 0, 0.001
#     FLOPS_MIN, FLOPS_MAX = 0, 10
#     SPR_MIN, SPR_MAX = 0, 1.0
#     THR_MIN, THR_MAX = 0, 1200

#     plt.subplot(base_layout+241)
#     plt.plot(log['train']['epoch'], log['train_tgt'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_tgt'], label="$Val$", color='g', linewidth=0.5)
#     plt.legend(loc='lower left', fontsize=6)
#     plt.ylim((TGT_MIN, TGT_MAX))
#     plt.ylabel('MSE Loss (best:' + str(np.array(log['test_tgt']).min()) + ')')

#     plt.subplot(base_layout+242)
#     plt.plot(log['train']['epoch'], log['train_slow'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_slow'], label="$Val$", color='g', linewidth=0.5)
#     plt.legend(loc='upper left', fontsize=6)
#     plt.ylim((SLOW_MIN, SLOW_MAX))
#     plt.ylabel('DSS')
    
#     plt.subplot(base_layout+243)
#     plt.plot(log['train']['epoch'], log['w_spr'], label="$Train$", color='r', linewidth=0.5)
#     plt.legend(loc='upper left', fontsize=6)
#     plt.ylim((SPR_MIN, SPR_MAX))
#     plt.ylabel('Weight sparsity (best:' + str(np.array(log['w_spr']).max())+ ')')

#     plt.subplot(base_layout+244)
#     plt.plot(log['train']['epoch'], log['train_mac'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_mac'], label="$Val$", color='g', linewidth=0.5)
#     plt.legend(loc='lower left', fontsize=6)
#     plt.ylim((FLOPS_MIN, FLOPS_MAX))
#     plt.ylabel('MAC (best:' + str(np.array(log['test_mac']).min())+ ')')

#     plt.subplot(base_layout+245)
#     plt.plot(log['train']['epoch'], log['train_acc'], label="$Train$", color='r', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['test_acc'], label="$Val$", color='g', linewidth=0.5)
#     plt.plot(log['train']['epoch'], log['pred_acc'], label="$Test$", color='b', linewidth=0.5)
#     plt.legend(loc='lower left', fontsize=6)
#     plt.ylim((TGT_MIN, TGT_MAX))
#     plt.ylabel('ACC (best:' + str(np.array(log['pred_acc']).min()) + ')')

#     plt.subplot(base_layout+246)
#     thr = np.stack(log['thr'],axis=1)
#     n_layer = thr.shape[0]
#     [plt.plot(log['train']['epoch'], thr[idx,:], label='l:' + str(idx), linewidth=0.5) for idx in range(n_layer)]
#     plt.legend(loc='lower left', fontsize=6)
#     plt.ylim((THR_MIN, THR_MAX))
#     plt.ylabel('Thr')
    
#     plt.subplot(base_layout+248)
#     plt.plot(log['train']['epoch'], log['train']['DSS_weight'], color='b', linewidth=0.5)
#     plt.ylabel('DSS_weight')

#     fig = plt.gcf()
#     # fig.savefig(os.path.join(log['log_dir'], log['exp_name'] + '_train.pdf'))
#     fig.savefig(os.path.join(log['log_dir'], log['exp_name'] + '.pdf'))
#     plt.close()

def plot_traj(log, is_best=False):
    if is_best:
        suffix = '_best'
    else:
        suffix = ''
    ref_traj_plot = log['running']['traj_ref']
    est_traj_plot = log['running']['traj_est']
    pred_mse_loss = log['test']['accuracy'][-1]
    epoch = log['test']['epoch'][-1]

    plt.figure(21, figsize=(16,9),dpi=200)
    plt.plot(ref_traj_plot, label='ref', linewidth=0.5, color='b')
    plt.plot(est_traj_plot, label=log['exp_name'] + "_epoch_" + str(epoch) , color='r', linewidth=0.5, linestyle= '-')
    plt.legend(loc='upper left', fontsize=5)
    plt.ylabel('Prediction angles mse [rad]' + str(pred_mse_loss))    
    
    fig = plt.gcf()
    fig.savefig(os.path.join(log['exp_dir'],'c', log['exp_name'] + suffix + '.pdf'))
    shutil.copyfile(os.path.join(log['exp_dir'],'c', log['exp_name'] + suffix + '.pdf'), os.path.join(log['log_dir'], log['exp_name'] + '_traj.pdf'))
    plt.close()