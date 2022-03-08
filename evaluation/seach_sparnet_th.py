""" 
# This script, search threshold for SparNet model by computing accuracy
# It also compute flops for SparNet model
#
# See Table 3 on the main paper
#
Example usage: 
CUDA_VISIBLE_DEVICES=1 python3 -m evaluation.seach_sparnet_th  --settings_file config/settings_ncaltech.yaml
CUDA_VISIBLE_DEVICES=1 python3 -m evaluation.seach_sparnet_th  --settings_file config/settings_prophesee.yaml
CUDA_VISIBLE_DEVICES=0 python3 -m evaluation.seach_sparnet_th  --settings_file config/settings_exp.yaml
"""

from config.settings import Settings
import numpy as np
import argparse
from training.object_cls_trainer import DSSClsModel
from training.object_det_trainer import DSSDetModel
from training.exp_trainer import ExpModel

from utils.log_utils import loadCheckpoint

if 0:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=False)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=False)
    # settings.batch_size=1
    th = [0, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28,  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print('Start evaluating thr-acc-flops relations of SparNet model on %s, ' % settings.dataset_name)
    # Build trainer
    if settings.model_name == 'dss_cls':
        trainer = DSSClsModel(settings)
    elif settings.model_name == 'dss_det':
        trainer = DSSDetModel(settings)
    elif settings.model_name == 'dss_exp':
        trainer = ExpModel(settings)     
    else:
        raise ValueError('Model name %s specified in the settings file is not implemented' % settings.model_name)

    loadCheckpoint(trainer.model, trainer.settings.resume_ckpt_file)  
    
    trainer.model.set_train_mode((True, True, True, True))

    for th_ in th:
        # trainer.model.set_train_mode((False, False, True, True))
        trainer.model.set_thr(th_)
        if settings.dataset_name=='NMNIST':
            trainer.testEpoch()
            print('NMNIST, %s threshold:  %.6f,trg_loss: %.6f, acc: %.6f, test_mac%.6f' % (settings.dataset_name, th_, trainer.test_tgt, trainer.test_acc, trainer.test_mac))
        else:
            trainer.validationEpoch()
            print('%s threshold:  %.6f,trg_loss: %.6f, acc: %.6f, test_mac%.6f' % (settings.dataset_name, th_, trainer.val_tgt, trainer.val_acc, trainer.val_mac))
           
if __name__ == "__main__":
    main()
