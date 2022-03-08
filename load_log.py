# Load log for exp_trainer
import os
import pickle
import numpy as np
from utils.log_utils import plot_stats

def loadLog(log_dir):
    print('Loading Log: %s'% (log_dir))
    path_files = os.path.join('../log', log_dir, 'visualization','log.bin')
    with open(path_files, "rb") as fp:   #Pickling
        log = pickle.load(fp)
    return log


########################################################################
# Main method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
if  __name__ =='__main__':
    result_list = [
        "NMNIST/0/kernel_conv_conv_nps_1_dss_1_tgt_0.02_step_2.0_relu_round_inv_log",
        "NMNIST/0/kernel_conv_conv_nps_2_dss_1_tgt_0.02_step_2.0_relu_round_inv_log",
        "NMNIST/0/kernel_conv_conv_nps_4_dss_1_tgt_0.02_step_2.0_relu_round_inv_log_2",
        "NMNIST/0/kernel_conv_conv_nps_8_dss_1_tgt_0.02_step_2.0_relu_round_inv_log_2",
    ]  
    # result_list = [
    #     "PilotNet/0/kernel_conv_conv_nps_1_dss_1_tgt_0.02_step_4.0_relu_round_inv_log",
    #     "PilotNet/0/kernel_conv_conv_nps_2_dss_1_tgt_0.02_step_4.0_relu_round_inv_log",
    #     "PilotNet/0/kernel_conv_conv_nps_4_dss_1_tgt_0.02_step_4.0_relu_round_inv_log",
    #     "PilotNet/0/kernel_conv_conv_nps_8_dss_1_tgt_0.02_step_4.0_relu_round_inv_log",
    # ]  

    for log_dir in result_list: 
        log = loadLog(log_dir)
        best_epoch = log['best']['epoch']
        print(log['test']['w_spr'][best_epoch])
        plot_stats(log)
        # thr = np.stack(log['thr'],axis=1)
        # print('%s threshold:  %.6f,w_sparsity: %.6f' % (log_dir, np.abs(thr[:,-1]).mean(), log['w_spr'][-1]))
