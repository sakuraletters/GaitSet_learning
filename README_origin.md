# GaitSet

[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

[GaitSet](https://arxiv.org/abs/1811.06186) is a **flexible**, **effective** and **fast** network for cross-view gait recognition.

#### Flexible 
The input of GaitSet is a set of silhouettes. 

- There are **NOT ANY constrains** on an input,
which means it can contain **any number** of **non-consecutive** silhouettes filmed under **different viewpoints**
with **different walking conditions**.

- As the input is a set, the **permutation** of the elements in the input
will **NOT change** the output at all.

#### Effective
It achieves **Rank@1=95.0%** on [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) 
and  **Rank@1=87.1%** on [OU-MVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html),
excluding  identical-view cases.

#### Fast
With 8 NVIDIA 1080TI GPUs, it only takes **7 minutes** to conduct an evaluation on
[OU-MVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html) which contains 133,780 sequences
and average 70 frames per sequence.

## What's new
The code and checkpoint for OUMVLP dataset have been released.
See [OUMVLP](#oumvlp) for details.

## Prerequisites

- Python 3.6
- PyTorch 0.4+
- GPU


## Getting started
### Installation

- (Not necessary) Install [Anaconda3](https://www.anaconda.com/download/)
- Install [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive)
- install [cuDNN7.0](https://developer.nvidia.com/cudnn)
- Install [PyTorch](http://pytorch.org/)

Noted that our code is tested based on [PyTorch 0.4](http://pytorch.org/)

### Dataset & Preparation
Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

Before training or test, please make sure you have prepared the dataset
by this two steps:
- **Step1:** Organize the directory as: 
`your_dataset_path/subject_ids/walking_conditions/views`.
E.g. `CASIA-B/001/nm-01/000/`.
- **Step2:** Cut and align the raw silhouettes with `pretreatment.py`.
(See [pretreatment](#pretreatment) for details.)
Welcome to try different ways of pretreatment but note that
the silhouettes after pretreatment **MUST have a size of 64x64**.

Futhermore, you also can test our code on [OU-MVLP Dataset](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html).
The number of channels and the training batchsize is slightly different for this dataset.
For more detail, please refer to [our paper](https://arxiv.org/abs/1811.06186).

#### Pretreatment
`pretreatment.py` uses the alignment method in
[this paper](https://ipsjcva.springeropen.com/articles/10.1186/s41074-018-0039-6).
Pretreatment your dataset by
```
python pretreatment.py --input_path='root_path_of_raw_dataset' --output_path='root_path_for_output'
```
- `--input_path` **(NECESSARY)** Root path of raw dataset.
- `--output_path` **(NECESSARY)** Root path for output.
- `--log_file` Log file path. #Default: './pretreatment.log'
- `--log` If set as True, all logs will be saved. 
Otherwise, only warnings and errors will be saved. #Default: False
- `--worker_num` How many subprocesses to use for data pretreatment. Default: 1

### Configuration 

In `config.py`, you might want to change the following settings:
- `dataset_path` **(NECESSARY)** root path of the dataset 
(for the above example, it is "gaitdata")
- `WORK_PATH` path to save/load checkpoints
- `CUDA_VISIBLE_DEVICES` indices of GPUs

### Train
Train a model by
```bash
python train.py
```
- `--cache` if set as TRUE all the training data will be loaded at once before the training start.
This will accelerate the training.
**Note that** if this arg is set as FALSE, samples will NOT be kept in the memory
even they have been used in the former iterations. #Default: TRUE

### Evaluation
Evaluate the trained model by
```bash
python test.py
```
- `--iter` iteration of the checkpoint to load. #Default: 80000
- `--batch_size` batch size of the parallel test. #Default: 1
- `--cache` if set as TRUE all the test data will be loaded at once before the transforming start.
This might accelerate the testing. #Default: FALSE

It will output Rank@1 of all three walking conditions. 
Note that the test is **parallelizable**. 
To conduct a faster evaluation, you could use `--batch_size` to change the batch size for test.

#### OUMVLP
Since the huge differences between OUMVLP and CASIA-B, the network setting on OUMVLP is slightly different.
- The alternated network's code can be found at `./work/OUMVLP_network`. Use them to replace the corresponding files in `./model/network`.
- The checkpoint can be found [here](https://1drv.ms/u/s!AurT2TsSKdxQuWN8drzIv_phTR5m?e=Gfbl3m).
- In `./config.py`, modify `'batch_size': (8, 16)` into `'batch_size': (32,16)`.
- Prepare your OUMVLP dataset according to the instructions in [Dataset & Preparation](#dataset--preparation).

## To Do List
- Transformation: The script for transforming a set of silhouettes into a discriminative representation.

## Authors & Contributors
GaitSet is authored by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/), 
[Yiwei He](https://www.linkedin.com/in/yiwei-he-4a6a6bbb/),
[Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/)
and JianFeng Feng from Fudan Universiy.
[Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/)
is the corresponding author.
The code is developed by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/)
and [Yiwei He](https://www.linkedin.com/in/yiwei-he-4a6a6bbb/).
Currently, it is being maintained by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/)
and Kun Wang.


## Citation
Please cite these papers in your publications if it helps your research:
```
@inproceedings{chao2019gaitset,
  author = {Chao, Hanqing and He, Yiwei and Zhang, Junping and Feng, Jianfeng},
  booktitle = {AAAI},
  title = {{GaitSet}: Regarding Gait as a Set for Cross-View Gait Recognition},
  year = {2019}
}
```
Link to paper:
- [GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition](https://arxiv.org/abs/1811.06186)


graph TD
    %% 定义样式
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef network fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef training fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,stroke-dasharray: 5 5;
    classDef operation fill:#ffffff,stroke:#616161,stroke-width:1px;

    %% --- 输入阶段 ---
    subgraph Input Data
        RGB["RGB Video Frames<br/>N x S x 3 x H x W"]:::input
        RawGT["Raw GT Labels<br/>N x S x 1 x H x W"]:::input
    end

    %% --- 网络前向传播阶段 ---
    subgraph E2E Segmentation Network
        RGB --> UNet["Shared Backbone<br/>(e.g., U-Net Encoder-Decoder)"]:::network
        UNet --> SharedFeat["Shared Features"]:::network
        
        %% 分支 1: 概率图
        SharedFeat --> HeadP["Head P<br/>(Conv + Sigmoid)"]:::network
        HeadP --> PMap["Probability Map (P)<br/>'Is this a person?'"]:::output
        
        %% 分支 2: 阈值图
        SharedFeat --> HeadT["Head T<br/>(Conv + Sigmoid)"]:::network
        HeadT --> TMap["Threshold Map (T)<br/>'How strict is the boundary here?'"]:::output
    end
    
    %% --- 可微分二值化模块 ---
    subgraph DB Module
        PMap --> DBCalc
        TMap --> DBCalc
        KParam["Amplification Factor k=50"]:::operation --> DBCalc
        
        DBCalc["Differentiable Binarization<br/>B_hat = 1 / (1 + exp(-k * (P - T)))"]:::operation
        DBCalc --> FinalSil["Final Refined Silhouettes<br/>(Soft Binary Output)"]:::output
    end

    %% --- 训练监督阶段 ---
    %% 修复点1：ID 从 "Training Supervision" 改为 "TrainingSupervision" (无空格)
    subgraph TrainingSupervision [Training Supervision & Online GT Gen]
        RawGT -.-> MorphOps["Morphological Operations<br/>(Erosion / Dilation)"]:::training
        
        MorphOps -.-> EdgeGT["Soft Edge Targets<br/>(Gaussian blurred edges)"]:::training
        MorphOps -.-> MaskGT["Boundary Mask (R_d)<br/>(Dilated area only, ignore background)"]:::training
        
        PMap -.-> LossP{Loss P<br/>BCE + Dice}:::training
        RawGT -.-> LossP
        
        TMap -.-> LossT_Pre["Masking Operation<br/>Select pixels only in Boundary Mask"]:::training
        EdgeGT -.-> LossT_Pre
        MaskGT -.-> LossT_Pre
        
        LossT_Pre -.-> LossT{Masked Loss T<br/>L1 Loss}:::training
    end

    FinalSil --> NextStage["To Gait Recognition Network<br/>(e.g., GaitSet)"]

    linkStyle default stroke-width:2px,fill:none,stroke:black;
    
    %% 修复点2：引用时也使用无空格的 ID
    class TrainingSupervision training

## License
GaitSet is freely available for free non-commercial use, and may be redistributed under these conditions.
For commercial queries, contact [Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/).
