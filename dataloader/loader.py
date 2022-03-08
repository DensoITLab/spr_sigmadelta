import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, device, shuffle=True):
        self.device = device
        split_indices = list(range(len(dataset)))

        if dataset.dataset_name in ['PilotNet']:
            collate_fn =  lambda x: collate_pilotnet(x, random_trans=dataset.mode == 'training')
        
        elif dataset.dataset_name in ['NMNIST']:
            collate_fn =  lambda x: collate_events(x, perm=312)
        else:
            collate_fn = collate_events

        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      collate_fn=collate_fn)
        else:
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      collate_fn=collate_fn)
        self.dataset = dataset
    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data, perm=312):
    labels = []
    event0 = []
    histograms0 = []
    event1 = []
    histograms1 = []
    for i, d in enumerate(data):
        labels.append(d[2])
        histograms0.append(d[3])
        histograms1.append(d[4])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]), 1), dtype=np.float32)], 1)
        event0.append(ev)
        ev = np.concatenate([d[1], i*np.ones((len(d[1]), 1), dtype=np.float32)], 1)
        event1.append(ev)

    event0 = torch.from_numpy(np.concatenate(event0, 0))
    event1 = torch.from_numpy(np.concatenate(event1, 0))
    labels = default_collate(labels)

    histograms0 = default_collate(histograms0)
    histograms1 = default_collate(histograms1)


    if perm==312:
        if histograms0.ndim==5:
            histograms0 = histograms0.squeeze(0).permute(3, 2, 0, 1)
            histograms1 = histograms1.squeeze(0).permute(3, 2, 0, 1)
        else:
            histograms0 = histograms0.permute(0, 3, 1, 2)
            histograms1 = histograms1.permute(0, 3, 1, 2)

    return event0, event1, labels, labels, histograms0, histograms1, 


def collate_pilotnet(data, random_trans=False):
    labels0 = []
    labels1 = []
    histograms0 = []
    histograms1 = []
    for i, d in enumerate(data):
        labels0.append(nvidia_label_preprocess(d[2]))
        labels1.append(nvidia_label_preprocess(d[3]))
        histograms0.append(nvidia_image_preprocess(d[0], random_trans))
        histograms1.append(nvidia_image_preprocess(d[1], random_trans))

    labels0 = default_collate(labels0)
    labels1 = default_collate(labels1)

    histograms0 = default_collate(histograms0).permute(0, 3, 1, 2)
    histograms1 = default_collate(histograms1).permute(0, 3, 1, 2)

    return torch.tensor([0]), torch.tensor([0]),  labels0, labels1, histograms0, histograms1,


def nvidia_image_preprocess(full_image, random_trans=False):
    # https://github.com/lhzlhz/PilotNet/blob/master/src/run_dataset.py
    # https://github.com/SullyChen/Autopilot-TensorFlow/blob/master/driving_data.py
    cropped_image = full_image.crop((0, 150, 455, 256)) #box=(left, upper, right, lower)
    image = cropped_image.resize((200, 66))
    image = np.array(image).astype(np.float32)
    image = (image - 128.0)/128.
    return image

def nvidia_label_preprocess(target):
    target = (target*np.pi/180)
    target = np.clip(target, -np.pi, +np.pi).astype(np.float32)
    return target
