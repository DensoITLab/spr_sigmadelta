"""
Example command:
    python3 -m evaluation.flops_asyc  --settings_file config/settings_ncaltech.yaml
    python3 -m evaluation.flops_asyc  --settings_file config/settings_prophesee.yaml
    python3 -m evaluation.flops_asyc  --settings_file config/settings_exp.yaml
"""
# Computed avg. event-rate
# 
# 
from dataloader.dataset import getDataloader
import argparse
from config.settings import Settings
import matplotlib.pyplot as plt
import os
import utils.visualizations as visualizations
import numpy as np

def computed_avg_rate(settings):
    dataloader = getDataloader(settings.dataset_name)

    # ---- Create Input -----
    train_dataset = dataloader(settings.dataset_path, 'all', settings.height,
                                    settings.width, augmentation=False, mode='validation',
                                    nr_events_window=-1, shuffle=True)


    total_event = 0
    total_time = 0
    for data_idx in range(train_dataset.__len__()):
        events, _ , _, _, _ = train_dataset.__getitem__(data_idx)
        total_event+= events.shape[0]
        total_time+=(events[0,3]- events[-1,3])
        print('processing %d %d, %f' %(data_idx, total_event, total_time))

    print('total_event: %d total_time: %f event_rate:%f ' %(total_event, total_time, total_event/total_time))


def gen_sample(settings):
    dataloader = getDataloader(settings.dataset_name)

    if settings.dataset_name=='':
        nr_events_window=2000
        batch_size = [1, 10, 100]
    elif settings.dataset_name=='Prophesee':
        nr_events_window=25000
        batch_size = [12500]
        # batch_size = [1, 10, 12500]
    elif settings.dataset_name=='NCaltech101':
        nr_events_window=25000
        batch_size = [1, 10, 600, 1250]
    else:
        print('Not supported')
        return

    # ---- Create Input -----
    train_dataset = dataloader(settings.dataset_path, 'all', settings.height,
                                    settings.width, augmentation=False, mode='validation',
                                    nr_events_window=nr_events_window, shuffle=True)


    total_event = 0
    total_time = 0

    dir_name = os.path.join('debug',settings.dataset_name)

    for data_idx in range(train_dataset.__len__()):
        if data_idx>100:
            break
        for de in batch_size:
            train_dataset.proc_rate = de
            _, _ , _, histogram0, histogram1 = train_dataset.__getitem__(data_idx)
            histogram0 = histogram0*16
            histogram1 = histogram1*16
            histogramd = np.abs(histogram0-histogram1)

            img_path = os.path.join(dir_name, '%0.5d_h0.png'%(data_idx))
            visualizations.visualizeHistogram(histogram0.astype(np.int), img_path)
            img_path = os.path.join(dir_name, '%0.5d_h1_bs_%0.4d.png'%(data_idx, de))
            visualizations.visualizeHistogram(histogram1.astype(np.int), img_path)
            img_path = os.path.join(dir_name, '%0.5d_hd_bs_%0.4d.png'%(data_idx, de))
            visualizations.visualizeHistogram(histogramd.astype(np.int), img_path)

            print(train_dataset.proc_rate)
        print('processing %d %d, %f' %(data_idx, total_event, total_time))


def main():
    parser = argparse.ArgumentParser(description='Evaluate network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file

    settings = Settings(settings_filepath, generate_log=False)

    # computed_avg_rate(settings)
    gen_sample(settings)

if __name__ == "__main__":
    main()
