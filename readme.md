# E2E training of Sparse Sigma-delta Network
**Learning to Sparsify Differences of Synaptic Signal for Efficient Event Processing**(BMVC2021, Oral)


## Note
Our main contribution of the paper, mconv layer, TDSS loss, and *macro-grad* are implemented `models/dss_layer.py` and `models/dss_utils.py`. `DSSConv2d` layer is desinged to replace existing `Conv2d` layer with minimum modification.
Example network using the layer and the loss is implemented in `models/dss_net`, `models/dss_mnist.py`, `models/dss_pilotnet.py`, `models/dss_VGG.py`, and `models/dss_object_det.py`.

Dataloader and trainer are based on the code of [Event-based Asynchronous Sparse Convolutional Networks, ECCV 2020](https://github.com/uzh-rpg/rpg_asynet)
(*Asyc-SSC*, [PDF](http://rpg.ifi.uzh.ch/docs/ECCV20_Messikommer.pdf)).
Refer to their original [code and documentation](https://github.com/uzh-rpg/rpg_asynet) to evaluate "Asyc-SSC" and 'Dense' model.


## Usage
###  Main and ablation experiments
The following commands will run the main and ablation experiments shown in Tab.2. 
The script will automatically train  and evaluate  for three times. 
It will generate a folder depending on the settings under `log/`, and saves statistics and results during training.
```
bash run_exp.sh PI XXX // PilotNet, x=0,1,2,3,4,5,6,7
bash run_exp.sh NM XXX // N-MNIST,  x=0,1,2,3,4,5,6,7
bash run_exp.sh NC XXX // N-Caltech101, x=0,1,2,3
bash run_exp.sh PS XXX // Prophesee Gen1 Automotive, x=0,1,2,3
```
'XXX' corresponds to the row of Tab.2 in the main paper (below the line highlighted in cyan).

###  Accessing the result
Most statistics during training are saved in `visualization/log.bin`.
Use the following script to load the log (and plot figure like Fig.A.).
```
load_log.py
```

### Training and evaluating in a different settings
Use `train.py` with corresponding setting files as follows.
```
train.py --settings_file "config/settings_xxx.yaml`
```
`XXX` corresponds to a dataset name. It will output the same log file as described above.


### Settings
One can adjust most of the basic training parameters/settings in the `config/settings_XXX.yaml` file without modifying the code. 
`XXX` corresponds to a dataset to be evaluated.
Some of the essential parameters are described below.
- Loss for reducing MAC is configured in  `dss/MAC_loss` tag. `1` is our proposed TDSS loss of Eq.(12), `0` is TDAM loss of Eq.(6), and `-1` is also TDAM loss but it is evaluated layer-wise and the TDAM loss is not backpropagated through the network. 
- Kernel type is configured in  `dss/kernel_mode` tag. `conv_nan` represent to convolution, and `conv_conv`  represent masked convolution. Other kernels are not used in the main paper, but discussed in the supplement.). Available options are as follows:
    - `conv_nan`
    - `conv_conv` 
    - `lrlc_lrlc` (see supplement)
    - `lc_lc` (see supplement)
    - `lrlc_nan` (see supplement)
    - `lc_nan` (see supplement)

- Qunatizatation method is configured in 'dss/quantizer' tag. 
'MG_xxx_yyy' represents represents proposed *macro-grad* and `LSQ_XXX_YYY` represents LSQ. 
'xxx' represents the order of division and multiplication of Eq.(2), `divmul` represents division followed by multiplication (same as Eq.(2)), and `muldiv` does this in reverse order. 
'yyy' represents the parameterization of $s$, `log` parameterize $s$ in log-space, which is convenient to restrict the range of of $s$ in positive region. Available options are as follows:
    - `MG_divmul_lin`
    - `MG_muldiv_lin`
    - `MG_divmul_log` 
    - `MG_divmul_log`
    - `LSQ_divmul_lin`
    - `LSQ_muldiv_lin`
    - `LSQ_divmul_log` 
    - `LSQ_muldiv_log`

- The quantization step size can be per channel or shared within a layer. It is configured in  `dss/channel_wise_th` tag.
- DSS scheduling parameters are configured in  `dss/DSS_weight_step` ($\eta_{step}$) and `mlc_dss/dss_criteria` ($\eta_{thr}$)  tag.
- Network size is onfigured in  `dss/expantion_ratio` tag. `1` represents default size.
- The dataset could be configured in the `dataset/name` and  `dataset/name/dataset_path` tag.


## Installation
Install the dependencies with pip as:
```
pip install -r requirements.txt
```
Install CPP Bindings for the event representation tool with pip as:
```
cd install event_representation_tool
pip install event_representation_tool
```

## Datasets
The following datasets are supported:
* Regression (steering angle prediction) on [Nvidia Pilotnet](https://github.com/lhzlhz/PilotNet) (video) 
* Classification on [N-MNIST](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fsh%2Ftg2ljlbmtzygrag%2FAABrCc6FewNZSNsoObWJqY74a%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNErunHbfiO0S6DM_6iqNwEaPuU4VQ)   (event)  
* Classification on [N-Caltech101](http://rpg.ifi.uzh.ch/datasets/gehrig_et_al_iccv19/N-Caltech101.zip)   (event)
* Object Detection on [Prophesee Gen1 Automotive](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) (event)

## Citation
If you use the code in your research,  please use the following BibTeX entry.
```BibTeX
@inproceedings{sekikawa2021spr_sd,
  author    = {Yusuke Sekikawa and Keisuke Uto},
  title     = {Learning to Sparsify Differences of Synaptic Signal for Efficient Event Processing},
  booktitle = {Proceedings of the British Machine Vision Conference 2021, {BMVC}
               2021, Online,  November 22-25, 2021},
  publisher = {{BMVA} Press},
  year      = {2021},
  url       = {https://www.bmvc2021.com/},
}
```

## Author
Yusuke Sekikawa, Denso IT Laboratory, Inc.

## LICENSE

Copyright (C) 2020 Denso IT Laboratory, Inc.
All Rights Reserved

Denso IT Laboratory, Inc. retains sole and exclusive ownership of all
intellectual property rights including copyrights and patents related to this
Software.

Permission is hereby granted, free of charge, to any person obtaining a copy
of the Software and accompanying documentation to use, copy, modify, merge,
publish, or distribute the Software or software derived from it for
non-commercial purposes, such as academic study, education and personal use,
subject to the following conditions:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
