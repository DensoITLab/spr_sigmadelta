"""
Example usage: 
export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
python3 train.py --settings_file "config/settings_prophesee.yaml"
python3 train.py --settings_file "config/settings_ncaltech.yaml"
python3 train.py --settings_file "config/settings_pilotnet.yaml" 
python3 train.py --settings_file "config/settings_mnist.yaml" 
"""
import argparse
from config.settings import Settings
import os

if 0:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

str_option_list = ['channel_wise_th', 'kernel_mode', 'quantizer', 'note']
bool_option_list = []
float_option_list = []
int_option_list = ['MAC_loss', 'expantion_ratio']

def override_settings(settings, args):
    for option in str_option_list:
        att = getattr(args, option)
        if att!=None:
            print('Override %s %s -> %s'%(option, getattr(settings, option),  att))
            setattr(settings, option, att)
    for option in  bool_option_list:
        att = getattr(args, option)
        if att!=None:
            print('Override %s %s -> %s'%(option, getattr(settings, option),  att))
            setattr(settings, option, att=='True')
    
    for option in float_option_list:
        att = getattr(args, option)
        if att!=None:
            print('Override %s %f -> %f'%(option, float(getattr(settings, option)),  float(att)))
            setattr(settings, option, float(att))

    for option in int_option_list:
        att = getattr(args, option)
        if att!=None:
            print('Override %s %d -> %d'%(option, int(getattr(settings, option)),  int(att)))
            setattr(settings, option, int(att))

def get_proc_rate(dataset_name):
    if dataset_name=='NMNIST':
        return [10, 1]
    elif dataset_name=='PilotNet':
        return [120, 480]
    else:
        return [10, 1]


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    for option in str_option_list+bool_option_list+float_option_list+int_option_list:
        parser.add_argument('--' + option)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)

    # Override some settings
    override_settings(settings, args)
    settings.generate_logdir()

    def get_trainer(settings):
        if settings.model_name == 'dss_cls':
            from training.dss_cls_trainer import DSSClsModel
            trainer = DSSClsModel(settings)
        elif settings.model_name == 'dss_det':
            from training.dss_det_trainer import DSSDetModel
            trainer = DSSDetModel(settings)
        elif settings.model_name == 'dss_exp':
            from training.dss_exp_trainer import ExpModel
            trainer = ExpModel(settings)     
        else:
            raise ValueError('Model name %s specified in the settings file is not implemented' % settings.model_name)
        return trainer

    n_try = 100
    settings.generate_logdir()
    n_ok = 0
    if 0:
        settings.proc_rate = 1
    trainer = get_trainer(settings)

    proc_rate = get_proc_rate(settings.dataset_name)
    for i_try in range(n_try):
        print('Training  try %d ok=%d' %(i_try, n_ok))
        ret = trainer.train()

        # Load and eval MAC on different rate
        if ret==1:
            mac_all = []
            mac_all.append(trainer.log['best']['mac'])

            default_proc_rate = settings.proc_rate
            settings.resume_ckpt_file = os.path.join(settings.ckpt_dir, 'model_step_%d.pth'%(trainer.log['best']['epoch']))
            n_connections = trainer.model.n_connections
            DSS_scale = trainer.model.DSS_scale

            for proc_rate_ in proc_rate:
                settings.proc_rate = proc_rate_
                settings.resume_training = True

                trainer = get_trainer(settings)
                trainer.model.n_connections = n_connections
                trainer.model.DSS_scale = DSS_scale

                # Enable quantization
                trainer.model.set_train_mode((True, False, False, False))
                best_epoch = trainer.log['best']['epoch']
                if settings.dataset_name=='NMNIST':
                    trainer.validationEpoch()
                    mac = trainer.log['val']['mac'][-1]
                else:
                    trainer.testEpoch()
                    mac = trainer.log['test']['mac'][-1]
                mac_all.append(mac)

            fp = open(os.path.join(settings.log_dir, 'mac.txt'),"w")
            [fp.write(str(mac)+',') for mac in mac_all]
            fp.close()

            n_ok+=1
            settings.try_idx = n_ok

            # Configure settings to original state
            settings.proc_rate = default_proc_rate
            settings.resume_training = False
            settings.generate_logdir()
            trainer = get_trainer(settings)
        else:
            trainer = get_trainer(settings)

        if n_ok==3:
            break

if __name__ == "__main__":
    main()
