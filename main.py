import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
import torch.nn as nn
import torchvision
import torchvision.datasets
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import logging
import numpy as np
import random
import matplotlib.pyplot as plt 
import csv
import medmnist
import numpy

import model
from model.module import ViTMAE
from model.module_lin import ViTMAE_lin
from model.module_knn import ViTMAE_knn
from model.vit_mae import ViTMAEForPreTraining
from dataset.dataloader import DataModule
from dataset.CLEVRCustomDataset import CLEVRCustomDataset
import transformers
from transformers import ViTMAEConfig
from utils import (
    print_config,
    setup_wandb,
    get_git_hash,
    load_checkpoints,
    Normalize
)

# Configure logging
log = logging.getLogger(__name__)
git_hash = get_git_hash()
def create_lambda_transform(mean, std):
    return torchvision.transforms.Lambda(lambda sample: (sample - mean) / std)
OmegaConf.register_new_resolver('divide', lambda a, b: int(int(a)/b))
OmegaConf.register_new_resolver('multiply', lambda a, b: int(int(a)*b))
OmegaConf.register_new_resolver("compute_lr", lambda base_lr, batch_size: base_lr * (batch_size / 256))
OmegaConf.register_new_resolver("decimal_2_percent", lambda decimal: int(100*decimal) if decimal is not None else decimal)
OmegaConf.register_new_resolver("convert_str", lambda number: "_"+str(number))
OmegaConf.register_new_resolver("substract_one", lambda number: number-1)
OmegaConf.register_new_resolver('to_tuple', lambda a, b, c: (a,b,c))
OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))

# Main function
@hydra.main(version_base="1.2", config_path="config", config_name="train_defaults.yaml")
def main(config: DictConfig) -> None:
    
    # Setup 
    print_config(config)
    pl.seed_everything(config.seed)
    hydra_core_config = HydraConfig.get()
    wandb_logger = setup_wandb(
        config, log, git_hash, {"job_id": hydra_core_config.job.name}
    )

    # Creating data 
    datamodule = instantiate(
        config.datamodule,
        data = config.datasets,
        masking = config.masking,
        extra_data = config.extradata,
    )
    
    # Creating model
    vit_config = instantiate(config.module_config)
    vit = instantiate(config.module,vit_config)
    model_train = instantiate(
        config.pl_module, 
        model=vit,
        datamodule = datamodule,
        save_dir=config.local_dir
        )
    model_train = load_checkpoints(model_train, config.checkpoint_fn)

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,  # Directory where to save the checkpoints
        filename='{epoch:02d}-{train_loss:.2f}',  # Filename format
        save_top_k=-1,  # Save all checkpoints
        save_weights_only=False,  # Save the full model (True for weights only)
        every_n_epochs=100  # Save every epoch
    )

    # Runing training (with eval on masked data to track behavior/convergence)
    trainer_configs = OmegaConf.to_container(config.trainer, resolve=True)
    print(trainer_configs)
    trainer = pl.Trainer(
            **trainer_configs,
            logger=wandb_logger,
            enable_checkpointing = True,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=config.pl_module.eval_freq,
        )
    print("------------------------- Start Training")
    trainer.fit(model_train, datamodule=datamodule)
    print("------------------------- End Training")


    # Final evaluation: original data, no pixel or pc masking, MAE eval protocol
    eval_configs = OmegaConf.to_container(config.evaluator, resolve=True)
    datamodule = instantiate(
        config.datamodule_eval,
        masking = {"type":"pixel","strategy":"pixel"},
        data = config.datasets,
    )
    
    del trainer, vit
    for i in range(config.data.task):
        model_eval = instantiate(
            config=config.pl_module_eval,
            model=model_train.model,
            datamodule=datamodule,
            save_dir=config.local_dir,
            task=i
        )
        evaluator = pl.Trainer(
                **eval_configs,
                logger=wandb_logger,
                enable_checkpointing = False,
                num_sanity_val_steps=0,
                check_val_every_n_epoch=1
            )
        print(f"------------------------- Start Evaluation: lin probe for task {i}")
        evaluator.fit(model_eval, datamodule=datamodule)
        print(f"------------------------- End Evaluation: lin probe for task {i}")


if __name__ == "__main__":
    main()

