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
        if self.masking.type == "pixel":
            self.pc_mask = 0

        elif self.masking.type == "pc":
            assert "eigenratiomodule" in list(extra_data.keys())
            assert "pcamodule" in list(extra_data.keys())

            self.eigenvalues = torch.Tensor(extra_data.eigenratiomodule)

            self.find_threshold = lambda eigenvalues ,ratio: np.argmin(np.abs(np.cumsum(eigenvalues) - ratio))
            self.get_pcs_index  = np.arange
            self.pc_mask = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # Load the images
        img1, y = self.dataset[idx]
        pc_mask = self.pc_mask

        if isinstance(y,list) and len(y)==2:
            pc_mask = y[1]
            y = y[0]
        if self.masking.type == "pc":
            if self.masking.strategy == "sampling_pc":
                index = torch.randperm(self.eigenvalues.shape[0]).numpy()
                pc_ratio = np.random.randint(10,90,1)[0]/100
                threshold = self.find_threshold(self.eigenvalues[index],pc_ratio)
                pc_mask = index[:threshold]
            elif self.masking.strategy == "pc":
                index = torch.randperm(self.eigenvalues.shape[0]).numpy()
                threshold = self.find_threshold(self.eigenvalues[index],self.masking.pc_ratio)
                pc_mask = index[:threshold]
        elif self.masking.type == "pixel":
            if self.masking.strategy == "sampling":
                pc_mask = float(np.random.randint(10,90,1)[0]/100)            
        return img1, y, pc_mask

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
        max_len = max([pc_mask.size for pc_mask in pc_masks])

        padded_pc_masks = [torch.nn.functional.pad(torch.tensor(pc_mask), (0, max_len - pc_mask.size),value=-1) for pc_mask in pc_masks]
        imgs = torch.stack(imgs)  # Assuming images are tensors and can be stacked directly
        labels = torch.tensor(labels)  # Convert labels to tensor
        padded_pc_masks = torch.stack(padded_pc_masks)  # Stack the padded pc_masks

        return imgs, labels, padded_pc_masks

    def train_dataloader(self) -> DataLoader:
        training_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, collate_fn=self.collate_fn if (self.masking.type == "pc" and self.masking.strategy in ["sampling_pc","sampling_rest_pc","sampling_ratio","sampling_pc_block","pc"]) else None
        )
        return training_loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers
        )
        return loader
