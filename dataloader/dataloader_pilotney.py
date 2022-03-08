from __future__ import print_function, division
import os
import torch
import random as rn
import numpy as np
from torch.utils.data import Dataset
import random

from torchvision import transforms

import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def get_pilotnet_folder(proc_rate):
    if proc_rate==10:
        foldername="nvidia_drive_10"
    elif proc_rate==120:
        foldername="nvidia_drive_120"
    elif proc_rate==480:
        foldername="nvidia_drive_480"   
    return foldername
            
class PilotNet(Dataset):
    """Dataset wrapping input and target tensors for the driving simulation dataset.

    Arguments:
        set (String):  Dataset - train, test
        path (String): Path to the csv file with the image paths and the target values
        :param mode: 'training', 'testing' or 'validation'
    """

    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='training',
                 event_representation='histogram', shuffle=True, proc_rate=10):
        self.mode = mode
        # proc_rate=480

        root = os.path.join(root, get_pilotnet_folder(proc_rate))
        print(root)
        n_test = 2800*(proc_rate//10)

            
        self.data = pd.read_csv(os.path.join(root, 'driving_log.csv') , header=None)

        # First column contains the middle image paths
        # Fourth column contains the steering angle
        self.data = self.data[2:]
        
        if (mode == "training"):
            start = 0
            end = int(0.8*len(self.data))
        elif (mode == "validation"):
            start = int(0.8*len(self.data))+1
            end = len(self.data)
        elif (mode == "testing"):
            start = 0 
            end = n_test
        elif (mode == "testing_rand"):
            start = 0 
            end = n_test
        else:
            raise ValueError('Invalid mode %s ' % mode)

        image_paths = np.array(self.data.iloc[:, 0])
        targets = np.array(self.data.iloc[:, 3])

        # Preprocess and filter data
        targets = [float(target) for target in targets]
        targets = gaussian_filter1d(targets, 2) 

        self.mean_diff = (torch.tensor(targets[:-1])-torch.tensor(targets[1:])).abs().mean()
        if self.mean_diff<=0.1:
            self.mean_diff = 1.0

        if (mode == "training") or (mode == "validation")  or (mode == "testing_rand"):
        # if (mode == "training") or (mode == "validation")  or (mode == "testing_rand") or (mode == "testing"):
            # random.seed(1)
            ind = np.random.permutation(np.arange(1,len(self.data)-1))
            # ind = np.random.permutation(len(self.data))
            self.image_paths = image_paths[ind][start:end]
            self.targets = targets[ind][start:end]
            self.image_paths_prv = image_paths[ind-1][start:end]
            self.targets_prv = targets[ind-1][start:end]
            self.image_paths_nxt = image_paths[ind+1][start:end]
            self.targets_nxt = targets[ind+1][start:end]
        else:
            self.image_paths = image_paths[(start+1):(end-1)]
            self.targets = targets[(start+1):(end-1)]
            self.image_paths_prv = image_paths[(start):(end-2)]
            self.targets_prv = targets[(start):(end-2)]
            self.image_paths_nxt = image_paths[(start+2):(end)]
            self.targets_nxt = targets[(start+2):(end)]

        bias = 0.03
        self.image_paths = [os.path.join(root, image_path)   for image_path, target in zip(self.image_paths, self.targets) if abs(target) > bias]
        self.image_paths_prv = [os.path.join(root, image_path)   for image_path, target in zip(self.image_paths_prv, self.targets) if abs(target) > bias]
        self.image_paths_nxt = [os.path.join(root, image_path)   for image_path, target in zip(self.image_paths_nxt, self.targets) if abs(target) > bias]
        self.targets_prv = [target_ for target, target_ in zip(self.targets, self.targets_prv) if abs(target) > bias]
        self.targets_nxt = [target_ for target, target_ in zip(self.targets, self.targets_nxt) if abs(target) > bias]
        self.targets = [target for target in self.targets if abs(target) > bias]

        # self.image_paths = self.image_paths[:10]

        self.nr_samples = len(self.image_paths)

        self.nr_classes = 1
        self.object_classes = []
        

    def __getitem__(self, index):
         # Get image name from the pandas df
        image_path0 = self.image_paths[index]
        if np.random.rand()>0.5:
            image_path1 = self.image_paths_prv[index]
            label1 = self.targets_prv[index]   
        else:
            image_path1 = self.image_paths_nxt[index]
            label1 = self.targets_nxt[index]    
        # Open image
        histograms0 = Image.open(image_path0)
        histograms1 = Image.open(image_path1)

        label0 = self.targets[index]     
        return histograms0, histograms1, label0, label1

    def __len__(self):
        return min(len(self.image_paths),len(self.targets_nxt))

if  __name__ =='__main__':
    print('__main__')