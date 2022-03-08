from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import signal
import sys
import os
import tqdm

from training.abstract_trainer import AbstractTrainer
from models.dss_red import RED
from models.dss_pilotnet import DSSPilotNet, DSSWPilotNet
from models.dss_mnist import DSSNMNIST
from utils.event_util import getEvent

# Surpress traceback in case of user interrupt
signal.signal(signal.SIGINT, lambda x0,y: sys.exit(0))

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("Creating folder: " + path)
    else:
        print("Override to existing folder: " + path)

def compute_alpha(y0, y1, mean_diff):
    return (y0-y1).abs().div(mean_diff)

class ExpModel(AbstractTrainer): 
    def buildModel(self):
        self.debug = False
        """Creates the specified model"""

        ########################################################################
        # Define configuration, log and network instance
        # ^^^^^^^^^^^^^^^^^^^^
        self.model_input_size = (self.settings.height, self.settings.width)
        self.stage = 0
        
        if self.settings.dataset_name in ['udacity', 'PilotNet']:
            self.criterion = nn.MSELoss()
            input_ch = 3
            if self.settings.kernel_mode=='red':
                self.settings.kernel_mode = 'conv_conv'
                self.model = RED(input_ch=input_ch, stage_config=(1, 2, 3), head_config=(256, 100, 50, 10, 1), last_act='atan', settings=self.settings)
            elif self.settings.kernel_mode=='conv_conv':
                self.model = DSSPilotNet(self.model_input_size, settings=self.settings)
            mkdir(os.path.join(self.settings.vis_dir, 'pred'))
        else:
            self.criterion = nn.CrossEntropyLoss()  
            input_ch = 2

            if self.settings.kernel_mode=='red':
                self.settings.kernel_mode = 'conv_conv'    
                self.model = RED(input_ch=input_ch, stage_config=(1, 2, 4), head_config=(256, 10), stem_stride=1 ,settings=self.settings)
            else:
                self.model = DSSNMNIST(self.model_input_size, settings=self.settings)

        self.model.to(self.settings.dev)
        self.dummy_input = torch.randn(2, input_ch, self.model_input_size[0], self.model_input_size[1]).to(self.settings.dev)

    
    ########################################################################
    # Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    def train(self):
        # Compute n_connections
        self.model = self.model.eval()
        self.model(self.dummy_input)
        self.model.compute_n_connection()
        print(np.sum(self.model.n_connections)/1e6)
        print(self.model.get_scale())
        self.optimizer.zero_grad()

        for epoch in range(self.settings.n_epoch):  # loop over the dataset multiple times
            self.epoch_step = epoch
            self.model.epoch_step = self.epoch_step

            # Schedule weight/Enable trainig scale and mask
            self.DSS_weight_schedule()
            # Train -> Val -> Test
            print("Start epoch %d %s, %s" %(self.epoch_step, self.settings.dataset_name , self.log['exp_name']))
            self.trainEpoch()
            self.validationEpoch()
            self.testEpoch()
            if self.scheduler is not None:
                self.scheduler.step()

            self.update_criteria()
            self.check_best()

            # Log
            self.saveLog(self.log)
 
            self.plot_stats(plot_supp=True)
            self.plot_stats(plot_supp=False)

            # Failure check
            if self.settings.n_warpup_epochs==0:
                failure_check_epoch = 10
            else:
                failure_check_epoch = self.settings.n_warpup_epochs

            if epoch>=failure_check_epoch and self.log['best']['dss_criteria']>self.settings.dss_criteria and self.log['test']['dss_criteria'][-1]>self.settings.dss_criteria:
                print('Train failed restart')
                return 0
        
        print('Finished Training: best (epoch, mac, criteria):%d, %.6f,%.6f'%(
            self.epoch_step, self.log['best']['mac'], self.log['best']['dss_criteria']))        
        return 1

    def trainEpoch(self):
        self.mainEpoch('train')

    def validationEpoch(self):
        self.mainEpoch('val')

    def testEpoch(self):
        if self.settings.dataset_name in ['udacity', 'PilotNet']:
            self.mainEpoch('test')
            # self.test_pilotnet_online(diff_mode=False)
            self.plot_traj()
        else:
            # For MNIST we evaluated in sliding window manner, we use the other statistic such as MAC from validation since it is basically same
            self.test_mnist()
            for stat_name in ['total_loss', 'dss_loss', 'mac', 'w_spr']:
                self.log['test'][stat_name].append(self.log['val'][stat_name][-1])

    def mainEpoch(self, mode):
        # Set statistic
        self.log[mode]['epoch'].append(self.epoch_step)
        self.log[mode]['DSS_weight'].append(self.DSS_weight)
        self.log[mode]['scale'].append(self.model.get_scale())
        self.log[mode]['min_scale'].append(self.log[mode]['scale'][-1].min())

        if mode=='train':
            self.model = self.model.train()
            # self.model.set_training(True)
        else:
            self.model = self.model.eval()
        self.clear_accum_stats()
        self.model.split_half=True
        loader = self.getDataLoader(mode)
        self.pbar = tqdm.tqdm(total=len(loader), unit='Batch', unit_scale=True)

        for i_batch, data in enumerate(loader, 0):
            if self.debug and i_batch>20:
                break
            
            _, _, y0_ref, y1_ref, x0, x1 = data
            def _inner_forward(x0, x1, y0_ref, y1_ref,mode):
                # move to gpu
                x0, x1, y0_ref, y1_ref = x0.to(self.settings.dev), x1.to(self.settings.dev), y0_ref.to(self.settings.dev), y1_ref.to(self.settings.dev)

                if mode=='train':
                    self.model.register_rand_hook()
                    # self.model.register_all_hook()
                else:
                    self.model.register_all_hook()
                    
                y0_est, y1_est, dss_info, selected_exp = self.model(torch.cat([x0,x1]))

                # Compute Loss
                # tgt_loss = self.criterion(y0_, y0)
                tgt_loss = (self.criterion(y0_est, y0_ref) + self.criterion(y1_est, y1_ref))/2
                if self.settings.dataset_name in ['udacity', 'PilotNet']:
                    alpha = compute_alpha(y0_ref, y1_ref, self.train_loader.dataset.mean_diff)    
                    dss_loss, mac, w_spr = self.model.compute_dss_loss(dss_info, selected_exp, alpha=alpha)
                    accuracy = tgt_loss
                else:
                    dss_loss, mac, w_spr = self.model.compute_dss_loss(dss_info, selected_exp)
                    predicted_classes = y0_est.argmax(1)
                    accuracy = (predicted_classes == y0_ref).float().mean()

                if self.DSS_weight==0:
                    total_loss = tgt_loss
                else:
                    total_loss = tgt_loss + self.DSS_weight*dss_loss
                return y0_est, y1_est, accuracy, total_loss, tgt_loss, dss_loss, mac, w_spr

            if mode=='train':
                self.optimizer.zero_grad()
                y0_est, y1_est, accuracy, total_loss, tgt_loss, dss_loss, mac, w_spr = _inner_forward(x0, x1, y0_ref, y1_ref, mode)
                total_loss.backward()
                # w1, w_grad, mask_th = self.model.get_w()
                self.optimizer.step()
                # w2, _, mask_th = self.model.get_w()
            else:
                with torch.no_grad():
                    y0_est, y1_est, accuracy, total_loss, tgt_loss, dss_loss, mac, w_spr = _inner_forward(x0, x1, y0_ref, y1_ref, mode)
            
            self.log[mode]['min_scale'][-1]=self.model.get_scale().min()
            self.accum_stats('total_loss', total_loss.detach().item())
            self.accum_stats('tgt_loss', tgt_loss.detach().item())
            self.accum_stats('dss_loss', dss_loss.detach().item())
            self.accum_stats('accuracy', accuracy.item())
            self.accum_stats('mac', mac.detach().item())
            self.accum_stats('w_spr', w_spr)
            self.accum_stats('bit', self.model.get_nbit())
            self.update_pbar(mode, i_batch)
            self.batch_step += 1
            self.accum_stats('traj_ref', y0_ref.tolist())
            self.accum_stats('traj_est', y0_est.tolist())
        self.pbar.close() 
        self.set_stats_all(mode, len(loader))

    def test_mnist(self):
        self.clear_accum_stats()
        self.model.split_half=False
        self.model.reset_hook()
        mode = 'test'
        self.pbar = tqdm.tqdm(total=len(self.test_loader), unit='Batch', unit_scale=True)
        
        self.log[mode]['epoch'].append(self.epoch_step)
        self.log[mode]['DSS_weight'].append(self.DSS_weight)
        self.log[mode]['scale'].append(self.model.get_scale())        
        self.log[mode]['min_scale'].append(self.log[mode]['scale'][-1].min())
        
        for i_batch, data in enumerate(self.test_loader):
            if self.debug and i_batch>20:
                break

            _, _, y0, _, x0, _ = data
            x0, y0 = x0.to(self.settings.dev), y0.to(self.settings.dev)

            with torch.no_grad():
                y0_, _, _ = self.model(x0) 
                if x0.shape[0]==1:
                    y0_ = y0_.unsqueeze(0)

                y_est = y0_.mean(0).detach().unsqueeze(0)
                tgt_loss = self.criterion(y_est , y0)

            predicted_classes =y_est.argmax(1)
            accuracy = (predicted_classes == y0).float().mean()

            self.accum_stats('tgt_loss', tgt_loss.detach().item())
            self.accum_stats('accuracy', accuracy.item())
            self.accum_stats('bit', self.model.get_nbit())
            self.update_pbar(mode, i_batch)


        # Set statistic
        self.set_stats_from_accum(mode,'tgt_loss', len(self.test_loader))
        self.set_stats_from_accum(mode,'accuracy', len(self.test_loader))
        self.set_stats_from_accum(mode,'bit', len(self.test_loader))
        self.pbar.close() 


    def test_pilotnet_online(self, pred_traj_CNN=[], diff_mode=True):
        # set test mode
        self.model.eval()
        # self.model.reset_mp()
        self.model.reset_hook()

        running_total_loss, running_tgt_loss, running_dss_loss, running_mac, running_acc = 0, 0, 0, 0, 0
        pred_traj = []
        ref_traj = []

        if diff_mode:
            self.model.set_training(False)
        else:
            self.model.set_training(True)
        x_lft = []
        prev_y0=None
        for itr, data in enumerate(self.test_loader):
            _, _, y0, _, x0, _ = data
            x0, y0 = x0.to(self.settings.dev), y0.to(self.settings.dev)
            y0 = y0.clamp(-np.pi,np.pi)
            if prev_y0==None:
                prev_y0 = y0.clone()

            if len(pred_traj_CNN)>0:
                y0 = torch.tensor(pred_traj_CNN[itr]).to(self.settings.dev)

            with torch.no_grad():
                # self.model.register_rand_hook()
                if diff_mode:
                    delta, x_lft = getEvent(x0, x_lft, 0)
                    y0_, e0_, m0_, selected_exp = self.model(delta) 
                    y_est = y0_[-1].squeeze() + (pred_traj[-1] if len(ref_traj)>0 else 0.0)
                else:
                    # self.model.reset_mp()
                    y0_, e0_, m0_, selected_exp = self.model(x0) 
                    y_est = y0_

                y_est = y_est.detach()
                # Compute loss mse/slow
                tgt_loss = self.criterion(y_est, y0)
                alpha = compute_alpha(y0, prev_y0,self.test_loader.dataset.mean_diff)
                dss_loss, mac, w_spr = self.compute_dss_loss(self.model, e0_, [None], m0_, [None], selected_exp, alpha=alpha)
                loss = tgt_loss + self.DSS_weight*dss_loss

            prev_y0 = y0.clone()

            running_total_loss += loss.item()
            running_tgt_loss += tgt_loss.item()                
            running_dss_loss += dss_loss.item()
            running_acc += tgt_loss.item()
            running_mac += mac.sum().item()

            # save histry
            pred_traj.append(y_est.item())
            ref_traj.append(y0.item())

        loss = running_total_loss / (len(self.test_loader)) 
        tgt_loss = running_tgt_loss / (len(self.test_loader)) 
        mac = running_mac / (len(self.test_loader)) 
        dss_loss = running_dss_loss / (len(self.test_loader)) 
        acc = running_acc / (len(self.test_loader)) 

        print('[TEST] TGT: %.6f, SLOW: %.6f, MOPS: %.6f' % (tgt_loss,  dss_loss, mac))

        self.test_loss,  self.test_tgt, self.test_acc,  self.test_dss,  self.test_mac = loss, tgt_loss, acc, dss_loss, mac
        self.pred_traj, self.ref_traj = pred_traj, ref_traj

########################################################################
# Main method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
if  __name__ =='__main__': 
    gpu_idx = 0
    dataset='nmnist' # nvidia, udacity, 'nmnist'
    trainer = ExpModel()
    kernel_modes =  ['lc_lc', 'lrlc_lrlc', 'conv_lc',  'conv_conv', 'lrlc_nan', 'lc_nan', 'conv_nan']
    for kernel_mode in kernel_modes:
        trainer.train()