# Principal Masked Autoencoders

Official PyTorch codebase for **P**rincipal **M**asked **A**uto-**E**ncoders (PMAE) presented in **Components Beat Patches: Eigenvector Masking for Visual Representation Learning** 
[\[arXiv\]](https://alicebizeul.github.io/assets/pdf/mae.pdf).

## Method
PMAE introduces an alternative approach to pixel masking for visual representation learning by masking principal components instead of pixel patches. This repository builds on top of the Masked Auto-Encoder (MAE, [\[arXiv\]](https://arxiv.org/pdf/2111.06377)) a prominent baseline for Masked Image Modelling (MIM) and replaces the masking of patches of pixels by the masking of principal components.

![pmae](https://github.com/alicebizeul/pmae/blob/main/assets/diagram-20241228.png)

## Code Structure

```
.
├── configs                   # directory in which all experiment '.yaml' configs are stored
├── src                       # the package
│   ├── train.py              #   the I-JEPA training loop
│   ├── helper.py             #   helper functions for init of models & opt/loading checkpoint
│   ├── transforms.py         #   pre-train data transforms
│   ├── datasets              #   datasets, data loaders, ...
│   ├── models                #   model definitions
│   ├── masks                 #   mask collators, masking utilities, ...
│   └── utils                 #   shared utilities
├── main_distributed.py       # entrypoint for launch distributed I-JEPA pretraining on SLURM cluster
└── main.py                   # entrypoint for launch I-JEPA pretraining locally on your machine
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

Adjust the paths to point to the directory you would like to use for storage of results and for fetching the data

### Training
To launch experiments, you can find a good example for training at  ```./script/jobs_euler_pcmae_random.sh``` and ```./script/jobs_euler_eval_pcmae_random.sh``` for evaluation. Otherwise, you can run the following command to get started with training: 

    EXPERIMENT="mae_clevr"
    python main.py user=myusername_mymachine experiment=$EXPERIMENT run_tag=$EXPERIMENT"

The ```EXPERIMENT``` variable refers the experimental setting defined in ```./config/experiment/```. 

### Evaluation on linear probing
To evaluate a checkpoint, you can gain inspiration from ```./config/user/callen_euler.yaml``` where I stored my runs. Then the following command gives an overview of how to launch the evaluation

    EXPERIMENT="mae_clevr"
    EPOCH=800
    RUN_TAG="$EXPERIMENT_eval_$EPOCH"
    python main.py user=callen_euler experiment=$DATASET trainer=eval checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"


### Adding pointers to key aspect of the repo 

A change in the masking strategy should be reflected in ```./dataset/dataloader.py``` file which define the image-masking pairs. At the moment, the mask only refers to a threshold; The change should also be reflected in ```./model/module.py``` where each batch is masked and passed through the model
