from models.dss_layer import DSSConv2d, DSSInput, DSSInvolution, DSSAlign
import torch.nn as nn
import numpy as np
import torch
import random

########################################################################
# Base Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class DSSNet(nn.Module):
    def __init__(self, settings=[]):
        super(DSSNet, self).__init__()
        self.settings = settings
        self.selected_out =[]
        self.fhooks = []
        self.split_half = True
        self.layer_wise_DSS  = False
        self.selected_idx = 0
        self.epoch_step = 0
        self.is_aligned = False

    # hooks
    def search_DSS_cand(self):
        DSS_cand = []
        for m in self.modules():
            if isinstance(m, DSSConv2d) or isinstance(m, DSSAlign):
                DSS_cand.append(m)
        self.DSS_cand = DSS_cand
        print(DSS_cand)
        return DSS_cand

    def search_th_cand(self):
        th_cand = []
        for m in self.modules():
            if isinstance(m, DSSConv2d) or isinstance(m, DSSAlign) or isinstance(m, DSSInput):
                th_cand.append(m)
        self.th_cand = th_cand
        return th_cand

    def reset_hook(self):
        for fhook in self.fhooks:
            fhook.remove()
        self.fhooks.clear()
        self.selected_out.clear()
        
    def register_rand_hook(self):
        for fhook in self.fhooks:
            fhook.remove()
        self.fhooks.clear()
        self.selected_out.clear()
        self.selected_idx = random.choices(range(len(self.DSS_cand)), weights=self.n_connections)[0]
        # self.selected_idx = np.random.randint(low=0, high=len(self.DSS_cand), size=1)[0]
        # print('register_rand_hook %d  %d '%(self.selected_idx, len(self.fhooks)))
        self.fhooks.append(self.DSS_cand[self.selected_idx].register_forward_hook(self.forward_hook(self.selected_idx)))

    # In Sigma-Delta DSS loss is computed layer-wises
    # Note: this module is compatible only for fully DSSConv2d network with DSSInput
    def mask_scale_grad(self):
        if self.layer_wise_DSS:
            for idx, m in enumerate(self.th_cand):
                if (idx==self.selected_idx):
                    m.th.requires_grad = True
                else:
                    m.th.requires_grad = False

    def reset_mask(self):
        for m in self.modules():
            if isinstance(m, DSSConv2d):
                m.reset_mask()

    def pre_process(self):
        self.clip_scale()
        # self.change_quantizer()
        self.mask_scale_grad()
        if self.epoch_step<self.settings.n_warpup_epochs:
            self.reset_mask()

    def register_all_hook(self):
        for fhook in self.fhooks:
            fhook.remove()
        self.fhooks.clear()
        self.selected_out.clear()

        # print('register_all_hook %d '%(len(self.fhooks)))
        for idx, module in enumerate(self.DSS_cand):
            self.fhooks.append(module.register_forward_hook(self.forward_hook(idx)))

    def forward_hook(self, selected_idx):
        def hook(module, input, output):
            # print('forward_hook idx:%d, numel %d'%(selected_idx, output.numel()))
            dss_stats = module.get_exp(input)
            self.selected_out.append([selected_idx, dss_stats])
        return hook

    def reset_mp(self):
        for module in self.modules():
            module.mp = []
            module.mp_lft = []

    def set_thr(self, scale_val, weight=[]):
        self.scale = scale_val
        if len(weight)==0 and isinstance(scale_val, list)==False:
            for module in self.modules():
                module.th.data = torch.tensor(scale_val)
                # print(module.th.data>0)
                module.th.requires_grad = (module.th.data>0).item()
        elif len(scale_val)>1:
            for idx, module in enumerate(self.modules):
                module.th.data = torch.tensor(scale_val[idx])
                module.th.requires_grad = module.th.data>0

        else:
            for module, weight_ in zip(self.modules, weight):
                module.th.data = scale_val*weight_
                module.th.requires_grad=True
                module.th.requires_grad = module.th.data


    def set_training(self, training):
        for module in self.modules():
            if isinstance(module, DSSConv2d) or isinstance(module, DSSInput):
                module.training = training
                
    def set_train_mode(self, flags):
        def _set_requires_grad(module, atr_name, value):
            if hasattr(module, atr_name) and getattr(module, atr_name)!=None:
                getattr(module, atr_name).requires_grad = value

        for module in self.modules():
            _set_requires_grad(module, 'th', flags[0])
            _set_requires_grad(module, 'm',  flags[1])
            _set_requires_grad(module, 'b',  flags[2])
            _set_requires_grad(module, 'w',  flags[3])


    def clip_scale(self):
        for module in self.modules():
            if isinstance(module, DSSConv2d) or isinstance(module, DSSInput):
                quantizer = self.settings.quantizer
                if quantizer in ['MG_divmul_lin', 'floor', 'ULSQ_muldiv_lin']:
                    module.th.data.clamp_(1.0/(self.settings.max_scale-self.settings.min_scale), None)
                elif quantizer in ['MG_muldiv_lin','ULSQ_divmul_lin', 'floor_inv', 'nan']:
                    module.th.data.clamp_(None, self.settings.max_scale-self.settings.min_scale)
                elif quantizer in ['MG_divmul_log', 'floor_log','ULSQ_divmul_log']:
                    module.th.data.clamp_(0, None)
                elif quantizer in ['MG_muldiv_log','ULSQ_muldiv_log', 'floor_inv_log']:
                    module.th.data.clamp_(None, 0)
                else:
                    print('Not implemented')

    def get_scale(self):
        scale = []
        for module in self.modules():
            if isinstance(module, DSSConv2d) or isinstance(module, DSSInput):
                scale.append(module.get_scale().mean().data.item())
        return np.array(scale)
    
    def get_grad_scale(self):
        grad_scale = []
        for module in self.modules():
            if isinstance(module, DSSConv2d) or isinstance(module, DSSInput):
                if module.th.grad is not None:
                    grad_scale.append(module.th.grad.data.item())
                else:
                    grad_scale.append(0.0)
        return np.array(grad_scale)
    
    def get_w(self):
        grad_w = []
        w = []
        mask_th =[]
        for module in self.modules():
            if isinstance(module, DSSConv2d):
                w.append(module.w.detach())
                mask_th.append(module.mask_th)
                if module.w.grad is not None:
                    grad_w.append(module.w.grad.detach())
                else:
                    grad_w.append(0.0)
        return w, grad_w, mask_th

    def compute_n_connection(self):
        n_connections = []
        for module in self.DSS_cand:
            if isinstance(module, DSSConv2d):
                if module.settings.kernel_mode in ['conv_nan', 'conv_conv']:
                    n_connections.append(np.prod([*module.w.shape])*np.prod([*module.map_size]))
                else:
                    n_connections.append(np.prod([*module.w.shape]))
            elif isinstance(module, DSSAlign):
                n_connections.append(np.prod([*module.map_size]))
            else:
                print('Unkown layer %s'%(module))

        self.n_connections = n_connections
        self.DSS_scale = np.array(n_connections).sum()/self.settings.base_n_connections

    def compute_n_weight(self):
        n_weight = []
        for module in self.DSS_cand:
            if isinstance(module, DSSConv2d):
                n_weight.append(np.prod([*module.w.shape]))
            elif isinstance(module, DSSAlign):
                n_weight.append(1000)
            else:
                print('Unkown layer %s'%(module))
        print('number of trainable parameters')
        print(np.array(n_weight).sum())
        self.n_weight = n_weight

    def get_n_dss_module(self):
        n_dss_module = 0
        for module in self.modules():
            if (isinstance(module, DSSConv2d) or isinstance(module, DSSInput) or isinstance(module, DSSAlign)):
                n_dss_module+=1
        return n_dss_module
    
    def get_nbit(self):
        n_bit = []
        for module in self.modules():
            if (isinstance(module, DSSConv2d) or isinstance(module, DSSInput) or isinstance(module, DSSAlign)):
                n_bit.append(module.get_nbit())
        return np.array(n_bit)

    def ext_output(self, x):
        chunk = 3 if self.is_aligned else 2 
        selected_idx = [selected_out_[0] for selected_out_ in self.selected_out]
        if self.split_half:
            x_split = torch.chunk(x, chunk, dim=0)
            dss_stats = [selected_out_[1] for selected_out_ in self.selected_out]
            return x_split[0], x_split[1], dss_stats, selected_idx
        else:
            y0 = [x]
            dss_stats = [selected_out_[1] for selected_out_ in self.selected_out]
            return y0[0], dss_stats, selected_idx

    def change_quantizer(self):
        if np.random.randint(0,2)==1 and self.epoch_step<int(self.settings.n_epoch*0.9):
            for module in self.modules():
                if (isinstance(module, DSSConv2d) or isinstance(module, DSSInput) or isinstance(module, DSSAlign)):
                    module.change_quantizer_LSQ()
        else:
            for module in self.modules():
                if (isinstance(module, DSSConv2d) or isinstance(module, DSSInput) or isinstance(module, DSSAlign)):
                    module.change_quantizer_MG()  

    def compute_dss_loss(self, dss_stats, selected_exp, alpha=torch.tensor([1.0])):
        dev = dss_stats[0][0].device
        dss_loss=0.0
        mac = 0.0
        n_active = 0.0
        n_weight = 0.0
        n_connections_ = np.sum([self.n_connections[idx] for idx in selected_exp])

        for dss_stats_ in dss_stats:
            if len(alpha)>1 or alpha>1.0:
                if dss_stats_[1].requires_grad:
                    with torch.no_grad():
                        scale = dss_stats_[0]/dss_stats_[1]
                    dss_loss+= (dss_stats_[1].mul(alpha.to(dev)).mul(scale).mean() + dss_stats_[0].mul(alpha.to(dev)).mean())
                else:
                    dss_loss+= dss_stats_[0].mul(alpha.to(dev)).mean()
            else:
                if dss_stats_[1].requires_grad:
                    with torch.no_grad():
                        scale = dss_stats_[0]/dss_stats_[1]
                    dss_loss+= (dss_stats_[1].mul(scale).mean() + dss_stats_[0].mean())
                else:
                    dss_loss+= dss_stats_[0].mean()

            if torch.isnan(dss_loss):
                print('torch.isnan(slow_loss)')
                dss_loss= torch.tensor([0.0]).to(dev)
            with torch.no_grad():
                # Flops for conv/fc
                mac+= dss_stats_[1].detach().mean()/1e6
            n_active+=dss_stats_[2]
            n_weight+=dss_stats_[3]

        dss_loss/=n_connections_
        mac*=(np.sum(self.n_connections)/n_connections_)
        sparcity=1-n_active/n_weight

        return dss_loss*self.DSS_scale, mac, sparcity