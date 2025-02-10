"""
Module Name: main.py
Author: Alice Bizeul, Some portions of the code below were taken from prior codebases written by Mark Ibrahim and Randall Balestreiro
Ownership: ETH Zürich - ETH AI Center
"""
# Standard library imports
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import yaml

# Third-party library imports
import git
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import submitit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.decomposition import PCA
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

# Hydra imports
from hydra.utils import instantiate

def setup_wandb(
    config: DictConfig,
    log: logging.Logger,
    git_hash: str = "",
    extra_configs: dict = dict(),
) -> WandbLogger:
    
    log_job_info(log)
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    job_logs_dir = os.getcwd()

    # increase timeout per wandb folks' suggestion
    os.environ["WANDB_INIT_TIMEOUT"] = "60"
    os.environ["WANDB_DIR"] = config.wandb_dir
    os.environ["WANDB_DATA_DIR"] = config.wandb_datadir
    os.environ["WANDB_CACHE_DIR"] = config.wandb_cachedir
    os.environ["WANDB_CONFIG_DIR"] = config.wandb_configdir

    config_dict["job_logs_dir"] = job_logs_dir
    config_dict["git_hash"] = git_hash

    name = (
        config.wandb.tags 
        + "_"
        + config.module._target_.split(".")[-1]
        + "_"
        + config.datamodule._target_.split(".")[-1]
    )
    config_dict.update(extra_configs)

    try:
        wandb_logger = WandbLogger(
            name=name,
            config=config_dict,
            settings={"start_method": "fork"},
            **config.wandb,
        )
    except Exception as e:
        print(f"exception: {e}")
        print("starting wandb in offline mode. To sync logs run")
        print(f"wandb sync {job_logs_dir}")
        os.environ["WANDB_MODE"] = "offline"
        wandb_logger = WandbLogger(
            name=name,
            config=config_dict,
            settings={"start_method": "fork"},
            **config.wandb,
        )
    return wandb_logger

def get_git_hash() -> Optional[str]:
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except:
        print("not able to find git hash")


@rank_zero_only
def print_config(
    config: DictConfig,
    resolve: bool = True,
) -> None:
    """Saves and prints content of DictConfig
    Args:
        config (DictConfig): Configuration composed by Hydra.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    run_configs = OmegaConf.to_yaml(config, resolve=resolve)
    with open("run_configs.yaml", "w") as f:
        OmegaConf.save(config=config, f=f)


def log_job_info(log: logging.Logger):
    """Logs info about the job directory and SLURM job id"""
    job_logs_dir = os.getcwd()
    log.info(f"Logging to {job_logs_dir}")
    job_id = "local"

    try:
        job_env = submitit.JobEnvironment()
        job_id = job_env.job_id
    except RuntimeError:
        pass
    log.info(f"job id {job_id}")


def find_existing_checkpoint(dirpath: str) -> Optional[str]:
    """Searches dirpath for an existing model checkpoint.
    If found, returns its path.
    """
    ckpts = list(Path(dirpath).rglob("*.ckpt"))
    if ckpts:
        ckpt = str(ckpts[-1])
        print(f"resuming from existing checkpoint: {ckpt}")
        return ckpt
    return None

def load_checkpoints(model, config):
    if config.f is not None: 
        print("------------------ Trying to load checkpoint from",config.f) 
        try:
            model.load_state_dict(instantiate(config)["state_dict"],strict=False)
            attempt=1
        except:
            try:
                model.load_state_dict(instantiate(config)["model_state_dict"],strict=False)
                attempt = 2
            except:
                attempt=3
        print('------------------ Loaded checkpoint following attempt',attempt," - model is ",type(model))
    return model 

# Define the function to save images and their reconstructions
def save_reconstructed_images(input, target, reconstructed, epoch, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    input_grid = torchvision.utils.make_grid(input[:8].cpu(), nrow=4, normalize=True)
    target_grid = torchvision.utils.make_grid(target[:8].cpu(), nrow=4, normalize=True)
    reconstructed_grid = torchvision.utils.make_grid(reconstructed[:8].cpu(), nrow=4, normalize=True)
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_grid.permute(1, 2, 0))
    axes[0].set_title('Input Images')
    axes[0].axis('off')

    axes[1].imshow(target_grid.permute(1, 2, 0))
    axes[1].set_title('Target Images')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_grid.permute(1, 2, 0))
    axes[2].set_title('Reconstructed Images')
    axes[2].axis('off')
    
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_{name}.png'))
    plt.close()

# Define the function to save images and their reconstructions
def save_attention_maps(input, attention_cls, attention_spatial, epoch, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    input_grid = torchvision.utils.make_grid(input[:8].cpu(), nrow=4, normalize=True)
    cls_grid = torchvision.utils.make_grid(attention_cls[:8].cpu(), nrow=4, normalize=True)
    spatial_grid = torchvision.utils.make_grid(attention_spatial[:8].cpu(), nrow=4, normalize=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_grid.permute(1, 2, 0))
    axes[0].set_title('Input Images')
    axes[0].axis('off')

    im1= axes[1].imshow(cls_grid.permute(1, 2, 0),cmap='gray')
    axes[1].set_title('CLS Attention Maps')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)  # Add color bar for CLS Attention Maps

    im2 = axes[2].imshow(spatial_grid.permute(1, 2, 0),cmap='gray')
    axes[2].set_title('Average Spatial Attention Maps')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)  # Add color bar for CLS Attention Maps

    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_{name}_attention.png'))
    plt.close()

def save_attention_maps_batch(att_map_cls, att_map_spatial, epoch, output_dir, name):
    # average over batch
    att_map_cls = torch.mean(att_map_cls.detach().cpu(),dim=0)
    att_map_spatial = torch.mean(att_map_spatial.detach().cpu(),dim=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    im1 = axes[0].imshow(att_map_cls.unsqueeze(0).permute(1, 2, 0),cmap='viridis')
    axes[0].set_title('CLS Attention Maps')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Add color bar for CLS Attention Maps

    im2 = axes[1].imshow(att_map_spatial.unsqueeze(0).permute(1, 2, 0),cmap='viridis')
    axes[1].set_title('Average Spatial Attention Maps')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Add color bar for CLS Attention Maps

    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_{name}_attention_batchavg.png'))
    plt.close()

class PCImageDataset(Dataset):
    def __init__(self, folder, pc_path, eigen_path, transform=None, ):
        """
        Initialize the dataset with two root directories and an optional transform.

        :param root1: Root directory for the first dataset.
        :param root2: Root directory for the second dataset.
        :param transform: Transformations to apply to the images.
        """
        self.dataset1 = ImageFolder(root=folder)
        try:
            self.pc_matrix = np.load(pc_path)
            self.eigenvalues = np.load(eigen_path)
        except:
            print(f"The path {pc_path} does not exist")
        self.transform = transform

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):

        # Load the images
        img1, _ = self.dataset1[idx]

        # Apply transformations if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return (tensor - self.mean)/self.std


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def get_eigenvalues(data):
    pca = PCA()  # You can adjust the number of components

    if len(data.shape)!=2:
        data = data.reshape(data.shape[0],*data.shape[1:])
    pca.fit(data)

    return pca.explained_variance_

class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.base_lr = 0.0
        self.annealing_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, total_epochs - warmup_epochs, eta_min=0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.annealing_scheduler.step(epoch - self.warmup_epochs)
    
class Lars(Optimizer):
    r"""Implements the LARS optimizer from `"Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>`_.
    Code taken from: https://github.com/NUS-HPC-AI-Lab/InfoBatch/blob/master/examples/lars.py 
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eeta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr=1e-3,
            momentum=0,
            eeta=1e-3,
            weight_decay=0,
            epsilon=0.0
    ) -> None:
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eeta <= 0:
            raise ValueError("Invalid eeta value: {}".format(eeta))
        if epsilon < 0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eeta=eeta, epsilon=epsilon, lars=True)

        super().__init__(params, defaults)

    def set_decay(self,weight_decay):
        for group in self.param_groups:
            group['weight_decay'] = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eeta = group['eeta']
            lr = group['lr']
            lars = group['lars']
            eps = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                decayed_grad = p.grad
                scaled_lr = lr
                if lars:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(p.grad)
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm)
                    )
                    trust_ratio.clamp_(0.0, 50)
                    scaled_lr *= trust_ratio.item()
                    if weight_decay != 0:
                        decayed_grad = decayed_grad.add(p, alpha=weight_decay)
                decayed_grad = torch.clamp(decayed_grad, -10.0, 10.0)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            decayed_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf

                p.add_(decayed_grad, alpha=-scaled_lr)

        return loss
