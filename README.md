# Principal Masked Autoencoders

Official PyTorch codebase for **P**rincipal **M**asked **A**uto-**E**ncoders (PMAE) presented in **Components Beat Patches: Eigenvector Masking for Visual Representation Learning** 
[\[arXiv\]](https://alicebizeul.github.io/assets/pdf/mae.pdf).

## Method
PMAE introduces an alternative approach to pixel masking for visual representation learning by masking principal components instead of pixel patches. This repository builds on top of the Masked Auto-Encoder (MAE, [\[arXiv\]](https://arxiv.org/pdf/2111.06377)) a prominent baseline for Masked Image Modelling (MIM) and replaces the masking of patches of pixels by the masking of principal components.

![pmae](https://github.com/alicebizeul/pmae/blob/main/assets/diagram-larger.png)

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
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [config/](config/) directory for example config files.


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
To launch experiments, you can find training and evaluation scripts in  ```scripts```. The following modifications should be made to the ```train.sh``` script to ensure a smooth training on your local machine:

    USER_MACHINE="myusername_mymachine"  # the user which runs the experiment
    EXPERIMENT="pmae_tiny_pc"            # the experiment to run, defines the model, dataset and masking type
    MASK=0.2                             # the masking ratio to use, default: 0.2

Please find the whole set of pre-defined experiment to chose from in [config/experiment](config/experiment). 

**Baselines:** To run the MAE baseline in place of PMAE, adjust ```EXPERIMENT``` to ```mae_tiny``` or any other experiment which starts by ```mae```. 

**Random Masking:** To run PMAE with randomized masking ratios as presented in the [\[arXiv\]](https://alicebizeul.github.io/assets/pdf/mae.pdf), adjust ```EXPERIMENT``` to ```pmae_tiny_pcsampling``` or any other experiment which contains ```pcsampling```. 

## Launch Evaluation
To evaluate a checkpoint, the evaluation script for linear probe, MLP probe, k-nearest neighbors, and fine-tuning approaches can be found in the ```scripts``` directory. The following modifications should be made to the ```eval.sh``` script to ensure a smooth evaluation on your local machine:

    USER_MACHINE="myusername_mymachine"  # the user which runs the experiment
    EXPERIMENT="pmae_tiny_pc"            # the experiment to run, defines the model, dataset and masking type
    EPOCH=800                            # the epoch to be evaluated
    MASK=0.2                             # the masking ratio to use, default: 0.2

Additionally, ensure the path to the checkpoint you want to evaluate is correctly set in your [user configuration file](config/user/abizeul_euler.yaml). For reference, see config/user/abizeul_euler.yaml. The specified checkpoint (defined by its path and epoch) will then be evaluated.
