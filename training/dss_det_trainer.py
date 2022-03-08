import os
import abc
from torch import tensor
import tqdm
import torch
import numpy as np

from models.dss_object_det import DSSObjectDet
from models.yolo_loss import yoloLoss
from models.yolo_detection import yoloDetect
from models.yolo_detection import nonMaxSuppression
from utils.statistics_pascalvoc import BoundingBoxes
import utils.visualizations as visualizations
from training.abstract_trainer import AbstractTrainer

class DSSDetModel(AbstractTrainer):
    def buildModel(self):
        self.debug = False
        from models.dss_red import RED
        self.nr_box=2
        # self.nr_box=5
        if self.settings.dataset_name == 'NCaltech101_ObjectDetection':
            self.model_input_size = [191, 255]
            self.cnn_spatial_output_size = [5, 7]
        elif self.settings.dataset_name == 'Prophesee':
            self.model_input_size = torch.tensor([223, 287])
            self.cnn_spatial_output_size = [6, 8]
        
        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]
        self.dummy_input = torch.randn(2, self.nr_input_channels, self.model_input_size[0], self.model_input_size[1]).to(self.settings.dev)

        """Creates the specified model"""

        if self.settings.kernel_mode=='red':
            self.settings.kernel_mode = 'conv_conv'
            self.model = RED(input_ch=self.nr_input_channels, stage_config=(1, 2, 4, 1), head_config=(512, 1024, spatial_size_product*(self.nr_classes + 5*self.nr_box)), settings=self.settings)
        else:
            self.model = DSSObjectDet(self.nr_classes, in_c=self.nr_input_channels,
                                        small_out_map=(self.settings.dataset_name == 'NCaltech101_ObjectDetection'),
                                        nr_box=self.nr_box,
                                        input_size=self.model_input_size,  settings=self.settings)
        
        self.model.to(self.settings.dev)
        self.model(self.dummy_input)
        self.model.compute_n_connection()
        print(np.array(self.model.n_connections).sum()/1e6)

        if self.settings.use_pretrained and (self.settings.dataset_name == 'NCaltech101_ObjectDetection' or
                                             self.settings.dataset_name == 'Prophesee'):
            self.loadPretrainedWeights()

    def loadPretrainedWeights(self):
        """Loads pretrained model weights"""
        checkpoint = torch.load(self.settings.pretrained_dense_vgg)
        try:
            pretrained_dict = checkpoint['state_dict']
        except KeyError:
            pretrained_dict = checkpoint['model']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'conv_layers.' in k and int(k[12]) <= 4}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def train(self):
        """Main training and validation loop"""
        self.n_ma = 1
        validation_step = 10
        self.DSS_weight = self.settings.DSS_weight
        self.DSS_weight_schedule()        

        for _ in range(self.settings.n_epoch):
            print("Start epoch %d %s, %s" %(self.epoch_step, self.settings.dataset_name , self.log['exp_name']))

            self.trainEpoch()
            self.update_criteria(mode_names=['train'])

            if (self.epoch_step % validation_step) == (validation_step - 1):
                self.validationEpoch()
                self.update_criteria(mode_names=['val'])
                self.check_best()
                self.DSS_weight_schedule()

            # Log
            # self.saveLog(self.log)
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
        self.bounding_boxes = BoundingBoxes()

        loss_function = yoloLoss
        loader = self.getDataLoader(mode)
        self.pbar = tqdm.tqdm(total=len(loader), unit='Batch', unit_scale=True)
        
        self.log[mode]['epoch'].append(self.epoch_step)
        self.log[mode]['DSS_weight'].append(self.DSS_weight)
        self.log[mode]['scale'].append(self.model.get_scale())
        self.log[mode]['min_scale'].append(self.log[mode]['scale'][-1].min())
        self.log[mode]['lr'].append(self.getLearningRate())

        for i_batch, sample_batched in enumerate(loader):
            if self.debug and i_batch>5:
                break

            _, _,  bounding_box, _, x0, x1 = sample_batched
            
            if self.debug and i_batch>4:
                break

            def _inner_forward(x0, x1, bounding_box, mode):
                # Change size to input size of sparse VGG
                x0 = torch.nn.functional.interpolate(x0, torch.Size(self.model_input_size))
                x1 = torch.nn.functional.interpolate(x1, torch.Size(self.model_input_size))

                if 0:
                    import matplotlib.pyplot as plt
                    plt.imshow(x0[0,0,:,:].cpu())
                    plt.savefig('histogram0.png')
                    plt.imshow(x1[0,0,:,:].cpu())
                    plt.savefig('histogram1.png')

                # Change x, width and y, height
                bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                                        / self.settings.width).long()
                bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                                        / self.settings.height).long()

                # Deep Learning Magic
                if mode=='train':
                    self.model.register_rand_hook()
                else:
                    self.model.register_all_hook()

                y0_, y1_, dss_info, selected_exp = self.model(torch.cat([x0,x1]))
                dss_loss, mac, w_spr = self.model.compute_dss_loss(dss_info, selected_exp)
                
                # Reshape for YOLO
                y0_ = y0_.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])

                tgt_loss = loss_function(y0_, bounding_box, self.model_input_size)[0]
                with torch.no_grad():
                    # detected_bbox = yoloDetect(y0_, self.model_input_size.to(y0_.device), threshold=0.3).long().cpu().numpy()
                    # detected_bbox = detected_bbox[detected_bbox[:, 0] == 0, 1:-2]

                    detected_bbox = yoloDetect(y0_, self.model_input_size.to(y0_.device), threshold=0.3)
                    detected_bbox = nonMaxSuppression(detected_bbox, iou=0.6)
                    detected_bbox = detected_bbox.cpu().numpy()

                # Save detected box to compute mAP
                self.saveBoundingBoxes(bounding_box.cpu().numpy(), detected_bbox)

                if self.DSS_weight>0:
                    total_loss = tgt_loss + self.DSS_weight*dss_loss
                else:
                    total_loss = tgt_loss

                return detected_bbox, total_loss, tgt_loss, dss_loss, mac, w_spr
            
            if mode=='train':
                self.optimizer.zero_grad()
                detected_bbox, total_loss, tgt_loss, dss_loss, mac, w_spr = _inner_forward(x0, x1, bounding_box, mode)
                total_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    detected_bbox, total_loss, tgt_loss, dss_loss, mac, w_spr= _inner_forward(x0, x1, bounding_box, mode)

            # Save training statistics
            self.accum_stats('total_loss', total_loss.detach().item())
            self.accum_stats('tgt_loss', tgt_loss.detach().item())
            self.accum_stats('dss_loss', dss_loss.detach().item())
            self.accum_stats('accuracy', total_loss.detach().item()) # Dummy replaced by mAP
            self.accum_stats('mac', mac.detach().item())
            self.accum_stats('w_spr', w_spr)
            self.accum_stats('bit', self.model.get_nbit())
            self.update_pbar(mode, i_batch)
            self.batch_step += 1


        self.set_stats_all(mode, len(loader))
        if mode!='train':
            self.compute_mAP(mode)

        self.pbar.close() 
        self.publish_tb(mode)
        print('mainEpoch done')

        # self.pbar.set_postfix(Epoch=self.epoch_step, Acc=self.log[mode]['accuracy'][-1], DSS=self.get_accum_stats('accuracy', i_batch+1), MOPS=self.get_accum_stats('accuracy', i_batch+1), eta=self.DSS_weight)
        if 0:
            vis_detected_bbox = detected_bbox[detected_bbox[:, 0] == 0, 1:-2].astype(np.int)
            image = visualizations.visualizeHistogram(x0[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
            image = visualizations.drawBoundingBoxes(image, bounding_box[0, :, :].cpu().numpy(),
                                                        class_name=[self.object_classes[i]
                                                                    for i in bounding_box[0, :, -1]],
                                                        ground_truth=True, rescale_image=True)
            image = visualizations.drawBoundingBoxes(image, vis_detected_bbox[:, :-1],
                                                        class_name=[self.object_classes[i]
                                                                    for i in vis_detected_bbox[:, -1]],
                                                        ground_truth=False, rescale_image=False)
            self.writer.add_image('mode' + '/Sample', image, self.epoch_step, dataformats='HWC')