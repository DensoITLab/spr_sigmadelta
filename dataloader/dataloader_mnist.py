from __future__ import print_function, division
# from config.settings import mkdir
import os
import random as rn
import numpy as np
from torch.utils.data import Dataset
import random

import matplotlib.pyplot as plt
from os import listdir
import pickle

# https://github.com/gorchard/event-Python/tree/78dd3b0a7fc508d551cecdbf93b959dc2d265765

class NMNIST:
    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='training',
                 event_representation='histogram', shuffle=True, proc_rate=100):
        """
        Creates an iterator over the N_MNIST dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        self.mode = mode
        if mode == 'training':
            mode = 'Train'
        elif mode == 'testing':
            mode = 'Test'
        if mode == 'validation':
            mode = 'Val'

        root = os.path.join(root, mode)
        # self.object_classes = listdir(root)
        self.object_classes = ['0','1','2','3','4','5','6','7','8','9']
        # self.object_classes = ['0']

        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.event_representation = event_representation

        self.files = []
        self.labels = []

        for i, object_class in enumerate(self.object_classes):
            new_files = [os.path.join(root, object_class, f) for f in listdir(os.path.join(root, object_class))]
            self.files += new_files
            self.labels += [i] * len(new_files)

        self.nr_samples = len(self.labels)
        self.proc_rate = proc_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.mode=='testing':
            return self.getitem_seq(idx)
        else:
            return self.getitem_rand(idx)

    def getitem_seq(self, idx):
        label = self.labels[idx]

        with open(self.files[idx], "rb") as fp:   #Pickling
            batch_inputs = pickle.load(fp)
        
        histograms = []
        for b, event in enumerate(batch_inputs):
            event[:,0] -= 1 
            event[:,1] -= 1 
            histogram = self.generate_input_representation(event, (self.height, self.width), no_t=True)
            histograms.append(histogram)

        histograms = np.stack(histograms, axis=3)
        return batch_inputs[0], batch_inputs[0], label, histograms, histograms


    def getitem_rand(self, idx):

        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        filename = self.files[idx]
        events = read_dataset(filename)
        nr_events = events.shape[0]
       
        window_start0 = 0
        window_start1 = min(nr_events,self.proc_rate)

        window_end0 = nr_events
        window_end1 = nr_events
        if self.augmentation:
            # events = random_shift_events(events, max_shift=1, resolution=(self.height, self.width))
            window_start0 = random.randrange(0, max(1, nr_events - self.nr_events_window-self.proc_rate))
            window_start1 = min(nr_events, window_start0+self.proc_rate)

        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end0 = min(nr_events, window_start0 + self.nr_events_window)
            window_end1 = min(nr_events, window_start1 + self.nr_events_window)

        # First Events
        events0 = events[window_start0:window_end0, :]
        histogram0= self.generate_input_representation(events0, (self.height, self.width))

        events1 = events[window_start1:window_end1, :]
        histogram1= self.generate_input_representation(events1, (self.height, self.width))


        if 0:
            plt.imshow(255*histogram0[:,:,0]/np.max(histogram0))
            plt.savefig('sample/nmnist/histogram0_' + str(idx)+ '.png')
            plt.imshow(255*histogram1[:,:,0]/np.max(histogram1))
            plt.savefig('sample/nmnist/histogram1_' + str(idx)+ '.png')

        return events0, events1, label, histogram0, histogram1


    def generate_input_representation(self, events, shape, no_t=False):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        if self.event_representation == 'histogram':
            return self.generate_event_histogram(events, shape, no_t=no_t)
        elif self.event_representation == 'event_queue':
            return self.generate_event_queue(events, shape)


    @staticmethod
    def generate_event_histogram(events, shape, no_t=False):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        if no_t:
            x, y, p = events.T
        else:
            x, y, t, p = events.T
        x = x.astype(np.int)
        y = y.astype(np.int)

        # if 1:
        #     x[x>(W-1)]= W-1
        #     y[y>(H-1)]= H-1


        img_pos = np.zeros((H * W,), dtype="float32")
        img_neg = np.zeros((H * W,), dtype="float32")

        np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
        np.add.at(img_neg, x[p == 0] + W * y[p == 0], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))/16

        return histogram

    @staticmethod
    def generate_event_queue(events, shape, K=15):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        events = events.astype(np.float32)

        if events.shape[0] == 0:
            return np.zeros([H, W, 2*K], dtype=np.float32)

        # [2, K, height, width],  [0, ...] time, [:, 0, :, :] newest events
        four_d_tensor = er.event_queue_tensor(events, K, H, W, -1).astype(np.float32)

        # Normalize
        four_d_tensor[0, ...] = four_d_tensor[0, 0, None, :, :] - four_d_tensor[0, :, :, :]
        max_timestep = np.amax(four_d_tensor[0, :, :, :], axis=0, keepdims=True)

        # four_d_tensor[0, ...] = np.divide(four_d_tensor[0, ...], max_timestep, where=max_timestep.astype(np.bool))
        four_d_tensor[0, ...] = four_d_tensor[0, ...] / (max_timestep + (max_timestep == 0).astype(np.float))

        return four_d_tensor.reshape([2*K, H, W]).transpose(1, 2, 0)


def random_shift_events(events, max_shift=1, resolution=(180, 240), bounding_box=None):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))

    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    if bounding_box is None:
        return events

    return events, bounding_box


def read_dataset(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    # NMIST: 34×34 pixels big
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7 #bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    #Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    #Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    events = np.stack([all_x[td_indices], all_y[td_indices], all_ts[td_indices], all_p[td_indices]], axis=1).astype(np.float32)
    # events[:,3] = 2*events[:,3]-1
    return events


def test_getitem_seq(root_path):
    valset = NMNIST(root_path, 34, 34, nr_events_window=2000, augmentation=True, mode='Val', shuffle=False)
    idx = random.randrange(0,  len(valset.files))
    histograms = get_val(valset, idx)
    for idx_b in range(histograms.shape[0]):
        file_path = 'sample/nmnist_val/histogram_' + str(idx) + '_' + str(idx_b) + '.png'
        print(file_path)
        plt.imshow(255*histograms[idx_b,0, :,:]/(histograms[idx_b,:, :,:]).max())
        plt.savefig(file_path)

def gen_val(root_path):
    # [mkdir(os.path.join(root_path, 'Val', str(i))) for i in range(10)]        

    testset = NMNIST(root_path, 10, 34, 34, nr_events_window=-1, augmentation=False, mode='testing', shuffle=True)

    for label, file in zip(testset.labels, testset.files):
        events = read_dataset(file)
        nr_events = events.shape[0]

        window_start = 0
        window_end = testset.nr_events_window

        batch_input = []
        while 1:
            if window_end>=nr_events:
                break
            else:
                batch_input.append(events[window_start:window_end,[0,1,3]].astype(np.uint8)) 
                window_start+=testset.de
                window_end+=testset.de
        if len(batch_input)==0:
            batch_input.append(events[:,[0,1,3]])

        path_dir = os.path.join(root_path, 'Val', str(label))
        path_files = os.path.join(path_dir, file[-9:])
        with open(path_files, "wb") as fp:   #Pickling
            pickle.dump(batch_input, fp)
        print(path_files)


def save_sample(root_path):
    trainset = NMNIST(root_path, 40, 40, nr_events_window=2000, augmentation=True, mode='Train', shuffle=True)
    print(trainset.__len__())
    for c in range(3):
        for i in range(trainset.__len__()):
            events0, events1, label, histogram0, histogram1 = trainset.__getitem__(i)
            print(label)

            plt.imshow(255*histogram0[:,:,0]/np.max(histogram0))
            plt.savefig('sample/nmnist/histogram0_' + str(i)+ '.png')
            plt.imshow(255*histogram1[:,:,0]/np.max(histogram1))
            plt.savefig('sample/nmnist/histogram0_' + str(i)+ '.png')

if  __name__ =='__main__':
    import matplotlib.pyplot as plt
    root_path = '/home/user/data/N_MNIST'
    gen_val(root_path)
