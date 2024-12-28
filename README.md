# Principled Masked Autoencoders

In this repository we explore more principled methods for image masking 

## Installation 

In your environment of choice, install the necessary requirements

    !pip install -r requirements.txt 

Alternatively, install individual packages as follows:

    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    !pip install pandas numpy pillow scikit-learn scikit-image plotly kaleido matplotlib submitit hydra-core kornia pytorch-lightning imageio medmnist wandb transformers

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