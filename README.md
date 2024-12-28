# Principal Masked Autoencoders

Official PyTorch codebase for **P**rincipal **M**asked **A**uto-**E**ncoders (PMAE) presented in **Components Beat Patches: Eigenvector Masking for Visual Representation Learning** 
[\[arXiv\]](https://alicebizeul.github.io/assets/pdf/mae.pdf).

## Method
PMAE introduces an alternative approach to pixel masking for visual representation learning by masking principal components instead of pixel patches. This repository builds on top of the Masked Auto-Encoder (MAE, [\[arXiv\]](https://arxiv.org/pdf/2111.06377)) a prominent baseline for Masked Image Modelling (MIM) and replaces the masking of patches of pixels by the masking of principal components.

![pmae](https://github.com/alicebizeul/pmae/blob/main/assets/diagram.png)

## Code Structure

```
.
├── assets                    # assets for the README file
├── configs                   # directory in which all experiment '.yaml' configs are stored
├── scripts                   # bash scripts to launch training and evaluation
│   ├── train.sh              #   training script
│   └── eval.sh               #   evaluation script
├── src                       # the package
│   ├── plotting.py           #   plotting function to training tracking
│   ├── utils.py              #   helper functions for init of models & opt/loading checkpoint
│   ├── dataset               #   datasets, data loaders, ...
│   └── model                 #   models, training loops, ...
├── tools                     # scripts to compute PCA prior to training
├── main.py                   # entrypoint for launch PMAE pretraining locally on your machine
└── requirements.txt          # requirements file
```

**Config files:**
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [configs/](configs/) directory for example config files.


## Installation 

In your environment of choice, install the necessary requirements

    !pip install -r requirements.txt 

Alternatively, install individual packages as follows:

    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    !pip install pandas numpy pillow scikit-learn scikit-image plotly kaleido matplotlib submitit hydra-core pytorch-lightning imageio medmnist wandb transformers

Create a config file that suits your machine:

    cd ./config/user
    cp abizeul_biomed.yaml myusername_mymachine.yaml

Adjust the paths in ```myusername_mymachine.yaml``` to point to the directory you would like to use for storage of results and for fetching the data

## Launch Training
To launch experiments, you can find training and evaluation scripts in  ```scripts```. The following modifications should be made to these script to ensure a smooth training on your local machine:

    EXPERIMENT="mae_clevr"
    python main.py user=myusername_mymachine experiment=$EXPERIMENT run_tag=$EXPERIMENT"

The ```EXPERIMENT``` variable refers the experimental setting defined in ```./config/experiment/```. 

### Launch Evaluation
To evaluate a checkpoint, you can gain inspiration from ```./config/user/callen_euler.yaml``` where I stored my runs. Then the following command gives an overview of how to launch the evaluation

    EXPERIMENT="mae_clevr"
    EPOCH=800
    RUN_TAG="$EXPERIMENT_eval_$EPOCH"
    python main.py user=callen_euler experiment=$DATASET trainer=eval checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"

