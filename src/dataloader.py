"""
Module Name: dataloader.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
"""

# Standard library imports
import os
import random
import time
from typing import Optional

# Third-party library imports
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision

# Hydra imports
from hydra.utils import instantiate

USER_NAME = os.environ.get("USER")

class PairedDataset(Dataset):
    def __init__(self, dataset, masking, extra_data):

        self.dataset = dataset
        self.masking = masking

        self.is_pc_mask = self.masking.type == "pc"

        if self.is_pc_mask:
            assert "eigenratiomodule" in extra_data and "pcamodule" in extra_data
            self.eigenvalues = np.array(extra_data['eigenratiomodule'])
            self.cum_eigenvalues = np.cumsum(self.eigenvalues)
            self.pc_mask = None
            self.find_threshold = lambda eigenvalues ,ratio: np.argmin(np.abs(np.cumsum(eigenvalues) - ratio))
            self.get_pcs_index  = np.arange
        else:
            self.pc_mask = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # Load the images
        img, y = self.dataset[idx]
        pc_mask = self.pc_mask

        if self.masking.type == "pc":

            if self.masking.strategy == "sampling_pc":
                index = torch.randperm(self.eigenvalues.shape[0]).numpy()
                pc_ratio = random.uniform(0.1, 0.9)
                threshold = self.find_threshold(self.eigenvalues[index],pc_ratio)
                pc_mask = index[:threshold]

            elif self.masking.strategy == "pc":
                index = np.random.permutation(self.eigenvalues.shape[0])
                threshold = self.find_threshold(self.eigenvalues[index],self.masking.pc_ratio)
                pc_mask = index[:threshold]
                
        elif self.masking.type == "pixel":
            if self.masking.strategy == "sampling":
                pc_mask = random.uniform(0.1, 0.9)

        return img, y, pc_mask

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data,
        masking, 
        extra_data =None,
        batch_size: int = 512,
        num_workers: int = 8,
        classes: int =10,
        channels: int =3,
        resolution: int =32,
        name: str =None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = classes
        self.input_channels = channels
        self.image_size = resolution
        self.masking = masking
        self.extra_data = extra_data
        self.datasets = data
        self.name = name

    def setup(self, stage):
        self.train_dataset = PairedDataset(
            dataset=self.datasets["train"],
            masking=self.masking,
            extra_data=self.extra_data
            )

        self.val_dataset = self.datasets["val"]
        self.num_val_samples = len(self.val_dataset)
        self.test_dataset = self.datasets["test"]

    def collate_fn(self,batch):
        """
        Custom collate function to handle variable-sized pc_mask.
        Pads the pc_mask to the size of the largest pc_mask in the batch.
        """

        imgs, labels, pc_masks = zip(*batch)

        imgs = torch.stack(imgs) 
        labels = torch.tensor(labels) 
        pc_masks = torch.tensor(pc_masks[0])
        return imgs, labels, pc_masks

    def train_dataloader(self) -> DataLoader:
        training_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, collate_fn=self.collate_fn if (self.masking.type == "pc" and self.masking.strategy in ["sampling_pc","pc"]) else None
        )
        return training_loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers
        )
        return loader
