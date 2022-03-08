import os
import abc
from torch import tensor
import tqdm
import torch
import numpy as np
import torch.nn as nn

from models.dss_VGG import DSSVGGCls
import utils.visualizations as visualizations
from models.dss_red import RED

# MLC/DSS related
from training.abstract_trainer import AbstractTrainer
# TODO multi gpu bn
# https://github.com/AndyYuan96/pytorch-distributed-training/blob/master/multi_gpu_distributed.py
class DSSClsModel(AbstractTrainer):
    def buildModel(self):
        self.debug = False
        # Input size is determined by architecture
        if self.settings.dataset_name == 'NCaltech101':
            self.model_input_size = [191, 255]
            # self.model_input_size = [193, 257]
        elif self.settings.dataset_name == 'NCars':
            self.model_input_size = [95, 127]

        self.dummy_input = torch.randn(2, self.nr_input_channels, self.model_input_size[0], self.model_input_size[1]).to(self.settings.dev)

        """Creates the specified model"""
        if self.settings.kernel_mode=='red':
            self.settings.kernel_mode = 'conv_conv'
            self.model = RED(input_ch=self.nr_input_channels, stage_config=(1, 2, 4, 1), head_config=(512, self.nr_classes), settings=self.settings)
        else:
            self.model = DSSVGGCls(self.nr_classes, in_c=self.nr_input_channels,
                                vgg_12=(self.settings.dataset_name == 'NCars'),
                                input_size=self.model_input_size, settings=self.settings)
                               
        self.model.to(self.settings.dev)
        
        self.model(self.dummy_input)
        self.model.compute_n_connection()
        print(np.array(self.model.n_connections).sum()/1e6)

    def train(self):
        self.n_ma = 1
        print(self.settings.dataset_name + self.log['exp_name'])
        validation_step = 10
        """Main training and validation loop"""
        self.DSS_weight_schedule()
        self.DSS_weight = self.settings.DSS_weight

        # self.model.set_train_mode((True, True, True, True))
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])

        print(self.epoch_step)
        print(self.scheduler.last_epoch)

        for epock in range(self.settings.n_epoch):
            self.scale = self.model.get_scale()
            # self.scale_grad = self.model.get_grad_scale()
            self.min_scale = self.scale.mean()

            # self.DSS_weight = 0.1
            self.trainEpoch()
            self.update_criteria(mode_names=['train'])
            
            if (self.epoch_step % validation_step) == (validation_step - 1):
                print("Start epoch %d %s, %s" %(self.epoch_step, self.settings.dataset_name , self.log['exp_name']))
                self.validationEpoch()
                self.update_criteria(mode_names=['val'])
                self.check_best()
                self.DSS_weight_schedule()

            # Log
            self.saveLog(self.log)
            self.epoch_step += 1
            self.scheduler.step()

        print('Finished Training: best (epoch, mac, criteria):%d, %.6f,%.6f'%(
            self.epoch_step, self.log['best']['mac'], self.log['best']['dss_criteria'])) 
        return 1

    def trainEpoch(self):
        self.mainEpoch('train')

    def validationEpoch(self):
        self.mainEpoch('val')

    def testEpoch(self):
        self.mainEpoch('test') 

    def mainEpoch(self, mode):
        if mode=='train':
            self.model = self.model.train()
        else:
            self.model = self.model.eval()
        self.model.set_training(False if mode=='train' else True)
        self.clear_accum_stats()

        loss_function = nn.CrossEntropyLoss()
        loader = self.getDataLoader(mode)
        self.pbar = tqdm.tqdm(total=len(loader), unit='Batch', unit_scale=True)

        self.log[mode]['epoch'].append(self.epoch_step)
        self.log[mode]['DSS_weight'].append(self.DSS_weight)
        self.log[mode]['scale'].append(self.model.get_scale())
        self.log[mode]['min_scale'].append(self.log[mode]['scale'][-1].min())
        self.log[mode]['lr'].append(self.getLearningRate())

        for i_batch, sample_batched in enumerate(loader):
            _, _,  labels, _, x0, x1 = sample_batched

            if self.debug and i_batch>5:
                break

            def _inner_forward(x0, x1, labels, mode):
                # Change size to input size of sparse VGG
                x0 = torch.nn.functional.interpolate(x0, torch.Size(self.model_input_size))
                x1 = torch.nn.functional.interpolate(x1, torch.Size(self.model_input_size))
            
                # Forward
                if mode=='train':
                    self.model.register_rand_hook()
                else:
                    self.model.register_all_hook()
                y0_, y1_, dss_info, selected_exp = self.model(torch.cat([x0,x1]))
                if 0:
                    import matplotlib.pyplot as plt
                    plt.imshow(x0[0,0,:,:].cpu())
                    plt.savefig('histogram0.png')
                    plt.imshow(x1[0,0,:,:].cpu())
                    plt.savefig('histogram1.png')

                # print(selected_exp)
                dss_loss, mac, w_spr = self.model.compute_dss_loss(dss_info, selected_exp)
                tgt_loss = loss_function(y0_, target=labels) + loss_function(y1_, target=labels) 
                if self.DSS_weight>0:
                    total_loss = tgt_loss + self.DSS_weight*dss_loss
                else:
                    total_loss = tgt_loss
                
                predicted_classes = y0_.argmax(1)
                accuracy = (predicted_classes == labels).float().mean()
                return predicted_classes, accuracy, total_loss, tgt_loss, dss_loss, mac, w_spr

            if mode=='train':
                self.optimizer.zero_grad()
                predicted_classes, accuracy, total_loss, tgt_loss, dss_loss, mac, w_spr = _inner_forward(x0, x1, labels, mode)
                total_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    predicted_classes, accuracy, total_loss, tgt_loss, dss_loss, mac, w_spr = _inner_forward(x0, x1, labels, mode)

            # Save training statistics
            self.accum_stats('total_loss', total_loss.detach().item())
            self.accum_stats('tgt_loss', tgt_loss.detach().item())
            self.accum_stats('dss_loss', dss_loss.detach().item())
            self.accum_stats('accuracy', accuracy.item())
            self.accum_stats('mac', mac.detach().item())
            self.accum_stats('w_spr', w_spr)
            self.accum_stats('bit', self.model.get_nbit())
            np.add.at(self.log[mode]['confusion_matrix'], (predicted_classes.data.cpu().numpy(), labels.data.cpu().numpy()), 1)
            self.update_pbar(mode, i_batch)
            self.batch_step += 1

        self.set_stats_all(mode, len(loader))        
        self.publish_tb(mode)
        self.pbar.close() 