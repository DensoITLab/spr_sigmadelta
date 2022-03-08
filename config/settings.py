import os
import time
import yaml
import torch
import shutil


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("Creating folder: " + path)
    else:
        print("Override to existing folder: " + path)

class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            self.try_idx = 0
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings['hardware']
            dev = hardware['gpu_device']

            self.dev = torch.device("cpu") if dev == "cpu" else torch.device("cuda:" + str(dev))

            self.num_cpu_workers = hardware['num_cpu_workers']
            if self.num_cpu_workers < 0:
                self.num_cpu_workers = os.cpu_count()

            self.pin_memory = hardware['pin_memory']


            # --- Model ---
            model = settings['model']
            self.model_name = model['model_name']

            # --- dataset ---
            dataset = settings['dataset']
            self.dataset_name = dataset['name']
            self.event_representation = dataset['event_representation']
            
            dataset_specs = dataset[self.dataset_name.lower()]

            self.dataset_path = dataset_specs['dataset_path']
            print("Dataset path: " + self.dataset_path)
            assert os.path.isdir(self.dataset_path)
            self.object_classes = dataset_specs['object_classes']
            self.height = dataset_specs['height']
            self.width = dataset_specs['width']
            self.nr_events_window = dataset_specs['nr_events_window']
            self.proc_rate = dataset_specs['proc_rate']
            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']
            self.use_pretrained = checkpoint['use_pretrained']
            self.pretrained_dense_vgg = checkpoint['pretrained_dense_vgg']
            self.pretrained_sparse_vgg = checkpoint['pretrained_sparse_vgg']

            # --- optimization ---
            optimization = settings['optim']
            self.batch_size = optimization['batch_size']
            self.init_lr = float(optimization['init_lr'])
            self.lr_scheduler_config = optimization['lr_scheduler_config']
            self.factor_lr = float(optimization['factor_lr'])
            self.n_epoch = optimization['n_epoch']

            # --- DSS ---
            dss = settings['dss']
            self.kernel_mode = dss['kernel_mode']
            self.activation = dss['activation']
            # self.enable_slow = acs['enable_slow']
            self.DSS_weight = float(dss['DSS_weight']) 
            self.DSS_weight_step = float(dss['DSS_weight_step']) 
            self.expantion_ratio =  int(dss['expantion_ratio']) 
            self.dss_criteria = float(dss['dss_criteria']) 
            # --- directories ---
            directories = settings['dir']
            log_dir = directories['log']
            self.root_log_dir = log_dir
            
            self.settings_yaml = settings_yaml
            if  self.model_name in ['sparse_cls', 'sparse_det']:
                 self.kernel_mode='ssc'
            elif self.model_name in ['dense_cls', 'dense_det']:
                self.kernel_mode='conv_orig'
            
            self.q_threshold = float(dss['q_threshold'])
            self.channel_wise_th = dss['channel_wise_th']
            # self.use_sine = mlc_dss['use_sine']
            self.MAC_loss = dss['MAC_loss']
            self.spatial_rank = int(dss['spatial_rank'])
            self.bias_pd = dss['bias_pd']
            self.quantizer = dss['quantizer']
            self.n_warpup_epochs = dss['n_warpup_epochs']
            self.n_linear_epochs = dss['n_linear_epochs']
            self.max_scale = dss['max_scale']
            self.min_scale = dss['min_scale']
            self.base_n_connections = dss['base_n_connections']

            assert isinstance(self.bias_pd, bool)
            # --- misc ---
            self.note = dss['note']
            # self.generate_logdir(generate_log=generate_log)


    def generate_logdir(self):
        # --- logs ---
        if self.model_name in ['dss_cls', 'dss_exp', 'dss_det',  'dss_exp2',  'dss_exp3']:
            exp_name =  'kernel_' + self.kernel_mode  + '_nps_' + str(int(self.expantion_ratio)) + '_mac_' + str(int(self.MAC_loss))
            # exp_name =  'kernel_' + self.kernel_mode +  '_eta_' + str((self.DSS_weight)) + '_nps_' + str(int(self.expantion_ratio))
            if self.dss_criteria>0:
                exp_name = exp_name + '_tgt_' + '%.2f' %self.dss_criteria +'_step_' + str(self.DSS_weight_step)
            else:
                exp_name = exp_name + '_eta_' + str(int(self.DSS_weight))

        else:
            exp_name = 'kernel_' + self.kernel_mode
        
        exp_name = exp_name + '_' + self.activation + '_' + self.quantizer
        if len(self.note)>0:
            exp_name = exp_name + '_' + self.note
        
        self.exp_name = exp_name

        # directry name
        self.exp_dir = os.path.join(self.root_log_dir,  self.dataset_name, str(self.try_idx))
        self.log_dir = os.path.join(self.exp_dir, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, exp_name, 'checkpoints')
        self.vis_dir = os.path.join(self.exp_dir, exp_name, 'visualization')
        mkdir(os.path.join(self.exp_dir, 'a'))
        mkdir(os.path.join(self.exp_dir, 'b'))
        mkdir(os.path.join(self.exp_dir, 'c'))
        mkdir(self.vis_dir)
        mkdir(self.ckpt_dir)

        settings_copy_filepath = os.path.join(self.log_dir, 'settings.yaml')
        shutil.copyfile(self.settings_yaml, settings_copy_filepath)
