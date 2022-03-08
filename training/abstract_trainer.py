import os
import abc
from numpy.core.numeric import Inf
from torch import tensor
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pickle

import dataloader.dataset
from dataloader.loader import Loader
from utils.statistics_pascalvoc import BoundingBoxes, BoundingBox, BBType, VOC_Evaluator, MethodAveragePrecision
import utils.visualizations as visualizations
from utils.log_utils import Struct
from utils.log_utils import plot_stats, plot_traj


class AbstractTrainer(abc.ABC):
    def __init__(self, settings):
        self.settings = settings

        self.model = None
        self.scheduler = None
        self.nr_classes = None
        self.val_loader = None
        self.train_loader = None
        self.nr_val_epochs = None
        self.bounding_boxes =None
        self.object_classes = None
        self.nr_train_epochs = None
        self.model_input_size = None

        if self.settings.event_representation == 'histogram':
            self.nr_input_channels = 2
        elif self.settings.event_representation == 'event_queue':
            self.nr_input_channels = 30

        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)
        self.dataset_loader = Loader

        self.createDatasets()
        self.buildModel()
        self.model.compute_n_weight()
        
        self.writer = SummaryWriter(self.settings.ckpt_dir)

        scale_para = []
        other_para = []
        for name, value in self.model.named_parameters():
            if "th" in name:
                print(name)
                scale_para += [value]
            else:
                other_para += [value]
        self.scale_para = scale_para

        #https://programmersought.com/article/33683055585/
        if settings.model_name in ['dss_exp']:
            self.optimizer = optim.AdamW(
                [
                    {"params": other_para, 'lr': self.settings.init_lr},
                    {"params": scale_para, 'weight_decay': 0.0, 'lr': self.settings.init_lr/1.0},
                ],
            )
        elif settings.model_name in ['dss_cls', 'dss_det']:
            self.optimizer = optim.AdamW(
                [
                    {"params": other_para, 'lr': self.settings.init_lr},
                    {"params": scale_para, 'weight_decay': 0.0, 'lr': self.settings.init_lr/1.0},
                ],
            )
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.settings.init_lr)
            # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.settings.init_lr)
            
        if settings.lr_scheduler_config is not None:
            if isinstance(settings.lr_scheduler_config, list):
                print('Use MultiStepLR')
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=settings.lr_scheduler_config, gamma=settings.factor_lr)
            elif settings.lr_scheduler_config==-1:
                print('Use CosineAnnealingLR')
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.settings.n_epoch, eta_min=self.settings.init_lr*settings.factor_lr, last_epoch=-1)
            else:
                print('No  scheduler')

        self.batch_step = 0
        self.epoch_step = 0
        
        # tqdm progress bar
        self.pbar = None
        self.DSS_weight = 0

        if settings.resume_training:
            self.loadCheckpoint(self.settings.resume_ckpt_file)
            self.scheduler.last_epoch=self.epoch_step

        self.n_ma = 3
        self.init_log()

    ########################################################################
    # Log
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def plot_stats(self,  plot_supp=False):
        plot_stats(self.log, plot_supp=plot_supp)

    def plot_traj(self,is_best=False):
        plot_traj(self.log, is_best=is_best)

    def update_pbar(self, mode, i_batch):
        if i_batch>=0:
            self.pbar.set_postfix({
                "mode": mode,
                "epoch": self.epoch_step,
                "total": self.get_accum_stats('total_loss', i_batch+1),
                "tgt": self.get_accum_stats('tgt_loss', i_batch+1),
                "dss": self.get_accum_stats('dss_loss', i_batch+1),
                "acc": self.get_accum_stats('accuracy', i_batch+1), 
                "mac": self.get_accum_stats('mac', i_batch+1), 
                "w_spr": self.get_accum_stats('w_spr', i_batch+1),  
                "eta": self.DSS_weight,
                "s": self.log[mode]['min_scale'][-1]
            })
            self.pbar.update(1)
        else:
            self.pbar.set_postfix({
                "mode": mode,
                "epoch": self.epoch_step,
                "total": self.log[mode]['total_loss'][-1],
                "tgt": self.log[mode]['tgt_loss'][-1],
                "dss": self.log[mode]['dss_loss'][-1],
                "acc": self.log[mode]['accuracy'][-1], 
                "mac": self.log[mode]['mac'][-1], 
                "w_spr": self.log[mode]['w_spr'][-1], 
                "eta": self.DSS_weight,
                "s": self.log[mode]['min_scale'][-1]
            })

    def saveLog(self, log):
        # print('Saving Log: %s'% (self.settings.vis_dir))
        log_path = os.path.join(self.settings.vis_dir, 'log.bin')
        with open(log_path, "wb") as fp:   #Pickling
            pickle.dump(log, fp)

    def loadLog(self):
        print('Loading Log: %s'% (self.settings.vis_dir))
        path_files = os.path.join(self.settings.vis_dir, 'log.bin')
        with open(path_files, "rb") as fp:   #Pickling
            log = pickle.load(fp)
        return log

    def init_log(self):
        log = {}
        log['dataset_name'] = self.settings.dataset_name
        log['log_dir'] = self.settings.log_dir
        log['exp_dir'] = self.settings.exp_dir
        log['exp_name'] = self.settings.exp_name

        log['train'] = {}
        log['test'] = {}
        log['val'] = {}

        stat_names = ['total_loss', 'dss_loss', 'tgt_loss', 'mac', 'accuracy', 'w_spr', 'a_spar', 'dss_criteria', 'bit', 'epoch', 'traj_ref', 'traj_est', 'DSS_weight', 'scale', 'lr','min_scale']

        for mode_name in ['train', 'val', 'test']:
            log[mode_name] = {}
            for stat_name in stat_names:
                log[mode_name][stat_name]=[]
    
        self.stat_names = stat_names

        for mode_name in ['best', 'best_test']:
            log[mode_name] = {}
            for stat_name in stat_names:
                if stat_name=='dss_criteria':
                    val=1.0
                elif stat_name=='mac':
                    val=Inf
                elif stat_name in ['traj_ref', 'traj_est']:
                    val = []
                else:
                    val=0.0
                log[mode_name][stat_name]=val
        
        log['running'] = {}

        for mode_name in ['train', 'val', 'test']:
            log[mode_name]['confusion_matrix'] = np.zeros([self.nr_classes, self.nr_classes])

        self.log = log
        self.clear_accum_stats()

    def get_log(self, mode_name, stat_name=None):
        if stat_name==None:
            return getattr(self.stats, mode_name)
        else:
            return getattr(getattr(self.stats, mode_name), stat_name)

    def append_log(self, mode_name, stat_name, value):
        self.log[mode_name][stat_name].append(value)

    def set_stats_all(self, mode, N):
        self.set_stats_from_accum(mode,'total_loss', N)
        self.set_stats_from_accum(mode,'tgt_loss', N)
        self.set_stats_from_accum(mode,'dss_loss', N)
        self.set_stats_from_accum(mode,'accuracy', N)
        self.set_stats_from_accum(mode,'w_spr', N)
        self.set_stats_from_accum(mode,'bit', N)
        self.set_stats_from_accum(mode,'mac', N)

    def set_stats_from_accum(self, mode_name, stat_name, N):
        value = self.get_accum_stats(stat_name, N)
        self.log[mode_name][stat_name].append(value)

    def accum_stats(self, stat_name, val):
        if stat_name in ['traj_ref', 'traj_est']:
            self.log['running'][stat_name] += val
        else:
            self.log['running'][stat_name]+=val

    def get_accum_stats(self, stat_name, N):
        return self.log['running'][stat_name]/N

    def clear_accum_stats(self):
        self.batch_step = 0
        for mode_name in ['train', 'val', 'test']:
            self.log[mode_name]['confusion_matrix'] = np.zeros([self.nr_classes, self.nr_classes])

        for stat_name in self.stat_names:
            if stat_name in ['traj_ref', 'traj_est']:
                val = []
            else:
                val=0.0
            self.log['running'][stat_name]=val

    def publish_tb(self, mode, publish_cm=False):
        for stat_name in ['accuracy', 'total_loss', 'tgt_loss', 'dss_loss', 'mac', 'w_spr', 'DSS_weight', 'lr','min_scale']:
            self.writer.add_scalar(mode+ '/' +stat_name, self.log[mode][stat_name][-1], self.epoch_step)
        
        self.log[mode]['confusion_matrix'] = self.log[mode]['confusion_matrix'] / (np.sum(self.log[mode]['confusion_matrix'], axis=-1, keepdims=True) + 1e-9)
        if publish_cm:
            plot_confusion_matrix = visualizations.visualizeConfusionMatrix(self.log[mode]['confusion_matrix'])
            self.writer.add_image(mode + '/Confusion_Matrix', plot_confusion_matrix, self.epoch_step, dataformats='HWC')
            
    def update_criteria(self, mode_names=['train', 'val', 'test']):
        for mode_name in mode_names:
            if self.settings.dataset_name in ['NMNIST', 'NCaltech101', 'Prophesee']:
                self.log[mode_name]['dss_criteria'].append(1-self.log[mode_name]['accuracy'][-1])
            else:
                self.log[mode_name]['dss_criteria'].append(self.log[mode_name]['tgt_loss'][-1])

            if self.log[mode_name]['dss_criteria'][-1]<0:
                print('update_criteria error')


    def check_best(self):
        # Best model := Clear test accuracy criteria & achieve lowest MAC
        if (self.log['val']['mac'][-1] < self.log['best']['mac']) and self.log['val']['dss_criteria'][-1]< self.settings.dss_criteria:
            print('Best test score achieved!: (mac, criteria):%.6f,%.6f->%.6f,%.6f' % (
                self.log['best']['mac'], self.log['best']['dss_criteria'],
                self.log['val']['mac'][-1], self.log['val']['dss_criteria'][-1]))
            self.log['best']['mac'] =  self.log['val']['mac'][-1]
            self.log['best']['dss_criteria'] =  self.log['val']['dss_criteria'][-1] 
            self.log['best_test']['mac'] =  self.log['test']['mac'][-1] if len(self.log['test']['mac'])>0 else Inf
            self.log['best_test']['dss_criteria'] =  self.log['test']['dss_criteria'][-1] if len(self.log['test']['dss_criteria'])>0 else 1.0
            self.log['best_test']['dss_criteria'] =  self.log['test']['w_spr'][-1] if len(self.log['test']['w_spr'])>0 else 0.0
            self.log['best']['epoch'] = self.epoch_step
            self.saveCheckpoint()

    @abc.abstractmethod
    def buildModel(self):
        """Model is constructed in child class"""
        pass

    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = self.dataset_builder(self.settings.dataset_path,
                                             self.settings.object_classes,
                                             self.settings.height,
                                             self.settings.width,
                                             self.settings.nr_events_window,
                                             augmentation=True, 
                                             proc_rate = self.settings.proc_rate,
                                             mode='training',
                                             event_representation=self.settings.event_representation)

        self.nr_train_epochs = int(train_dataset.nr_samples / self.settings.batch_size) + 1
        self.nr_classes = train_dataset.nr_classes
        self.object_classes = train_dataset.object_classes

        val_dataset = self.dataset_builder(self.settings.dataset_path,
                                           self.settings.object_classes,
                                           self.settings.height,
                                           self.settings.width,
                                           self.settings.nr_events_window,
                                            proc_rate = self.settings.proc_rate,
                                           mode='validation',
                                           event_representation=self.settings.event_representation)
        self.nr_val_epochs = int(val_dataset.nr_samples / self.settings.batch_size) + 1


        test_dataset = self.dataset_builder(self.settings.dataset_path,
                                           self.settings.object_classes,
                                           self.settings.height,
                                           self.settings.width,
                                           self.settings.nr_events_window,
                                           proc_rate = self.settings.proc_rate,
                                           mode='testing',
                                           event_representation=self.settings.event_representation)
        self.nr_test_epochs = int(test_dataset.nr_samples / 1) + 1

        # Build loader
        self.train_loader = self.dataset_loader(train_dataset, batch_size=self.settings.batch_size,
                                                device=self.settings.dev,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=self.settings.pin_memory)
        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                              device=self.settings.dev,
                                              num_workers=self.settings.num_cpu_workers, pin_memory=self.settings.pin_memory)
        
        self.test_loader = self.dataset_loader(test_dataset, batch_size=1 if self.settings.dataset_name in ['NMNIST'] else self.settings.batch_size,
                                              device=self.settings.dev,
                                              num_workers=self.settings.num_cpu_workers, pin_memory=self.settings.pin_memory, shuffle=False)
    def getDataLoader(self, mode):
        if mode == 'train':
            return self.train_loader
        elif mode == 'val':
            return self.val_loader
        else:
            return self.test_loader
    
    @staticmethod
    def denseToSparse(dense_tensor):
        """
        Converts a dense tensor to a sparse vector.

        :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
        :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
        :return features: NumberOfActive x FeatureDimension
        """
        non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
        locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense_tensor[select_indices], dim=-2)

        return locations, features


    def saveBoundingBoxes(self, gt_bbox, detected_bbox):
        """
        Saves the bounding boxes in the evaluation format

        :param gt_bbox: gt_bbox[0, 0, :]: ['u', 'v', 'w', 'h', 'class_id']
        :param detected_bbox[0, :]: [batch_idx, u, v, w, h, pred_class_id, pred_class_score, object score]
        """
        image_size = self.model_input_size.cpu().numpy()
        for i_batch in range(gt_bbox.shape[0]):
            for i_gt in range(gt_bbox.shape[1]):
                gt_bbox_sample = gt_bbox[i_batch, i_gt, :]
                id_image = self.batch_step * self.settings.batch_size + i_batch
                if gt_bbox[i_batch, i_gt, :].sum() == 0:
                    break

                bb_gt = BoundingBox(id_image, gt_bbox_sample[-1], gt_bbox_sample[0], gt_bbox_sample[1],
                                    gt_bbox_sample[2], gt_bbox_sample[3], image_size, BBType.GroundTruth)
                self.bounding_boxes.addBoundingBox(bb_gt)

        for i_det in range(detected_bbox.shape[0]):
            det_bbox_sample = detected_bbox[i_det, :]
            id_image = self.batch_step * self.settings.batch_size + det_bbox_sample[0]

            bb_det = BoundingBox(id_image, det_bbox_sample[5], det_bbox_sample[1], det_bbox_sample[2],
                                 det_bbox_sample[3], det_bbox_sample[4], image_size, BBType.Detected,
                                 det_bbox_sample[6])
            self.bounding_boxes.addBoundingBox(bb_det)

    
    def compute_mAP(self, mode):
        evaluator = VOC_Evaluator()
        metrics = evaluator.GetPascalVOCMetrics(self.bounding_boxes,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
        acc_AP = 0
        total_positives = 0
        for metricsPerClass in metrics:
            acc_AP += metricsPerClass['AP']
            total_positives += metricsPerClass['total positives']
        mAP = acc_AP / self.nr_classes
        self.log[mode]['accuracy'][-1] = mAP

    def getLearningRate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def loadCheckpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.epoch_step = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'],  strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            self.log = self.loadLog()
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def saveCheckpoint(self):
        file_path = os.path.join(self.settings.ckpt_dir, 'model_step_' + str(self.epoch_step) + '.pth')
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch_step}, file_path)

    def DSS_weight_schedule(self):
        def adaptive_schedule(start_epock, grad_setting):
            if self.epoch_step==start_epock:
                print('Start  adaptive_schedule for DSS/DAM at %d %s' %(start_epock, grad_setting))
                self.model.set_train_mode(grad_setting)
                self.DSS_weight = self.settings.DSS_weight

            if self.epoch_step>max(start_epock, self.n_ma):
                # Adaptive scheduling using target accuracy
                ma_acc = [self.log['val']['dss_criteria'][-idx] for idx in range(1,self.n_ma+1)]
                ma_acc = np.array(ma_acc).mean()
                # ma_acc = (self.log['val']['dss_criteria'][-1]+ self.log['val']['dss_criteria'][-2])/2
                if ma_acc<self.settings.dss_criteria:
                    self.DSS_weight += self.settings.DSS_weight_step
                else:
                    self.DSS_weight = max(self.settings.DSS_weight, self.DSS_weight - self.settings.DSS_weight_step/10)
        
        def linear_schedule(start_epock, grad_setting):

            if self.epoch_step==start_epock:
                print('Start  linear_schedule for DSS/DAM at %d %s' %(start_epock, grad_setting))
                self.model.set_train_mode(grad_setting)
            
            # Linear scheduling
            if self.epoch_step < start_epock:
                self.DSS_weight = 0
            elif self.epoch_step < self.settings.n_linear_epochs:
                self.DSS_weight = (self.epoch_step-start_epock)*self.settings.DSS_weight/(self.settings.n_linear_epochs-start_epock)
            else:
                self.DSS_weight = self.settings.DSS_weight
            # print(self.DSS_weight)

        if self.epoch_step==0:
            self.model.set_train_mode((False, False, True, True))
            self.DSS_weight = 0.0
            print('Start scheduling for DSS/DAM at %d' %(self.settings.MAC_loss))
    
        if self.settings.MAC_loss==-1:
            # Fix weight and bias
            dss_start_epock = self.settings.n_epoch//2
            if self.epoch_step==dss_start_epock:
                self.model.layer_wise_DSS = True
            if self.settings.dss_criteria>0:
                adaptive_schedule(dss_start_epock, (True, False, False, False))
            else:
                linear_schedule(dss_start_epock, (True, False, False, False))
        else:
            # Fix weight and bias
            if self.settings.dss_criteria>0:
                adaptive_schedule(self.settings.n_warpup_epochs, (True, True, True, True))
            else:
                linear_schedule(self.settings.n_warpup_epochs, (True, True, True, True))
