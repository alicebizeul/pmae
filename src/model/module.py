"""
Module Name: module.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
"""

# Standard library imports
import os
import time
from typing import Any, Dict, List, Optional

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import Tensor
from torch.nn.parameter import Parameter
from torchvision.models import resnet18
import wandb

# Local imports
from ..plotting import plot_loss, plot_performance
from ..utils import save_attention_maps, save_attention_maps_batch, save_reconstructed_images

class ViTMAE(pl.LightningModule):

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        base_learning_rate: float =1e-3,
        weight_decay: float = 0.05,
        betas: list =[0.9,0.95],
        optimizer_name: str = "adamw",
        warmup: int =10,
        datamodule: Optional[pl.LightningDataModule] = None,
        eval_freq: int =100,
        eval_type ="multiclass",
        eval_fn =nn.CrossEntropyLoss(),
        eval_logit_fn = nn.Softmax(),
        save_dir: str =None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.optimizer_name = optimizer_name
        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes
        self.image_size = datamodule.image_size
        self.classifierlr = learning_rate
        self.warm_up = warmup
        self.eval_freq = eval_freq
        self.masking = datamodule.masking

        self.model = model

        if self.masking.type == "pc":
            self.register_buffer("masking_fn_",torch.Tensor(self.datamodule.extra_data.pcamodule.T))
        elif self.masking.type == "random":
            self.register_buffer("masking_fn",nn.Linear())

        self.classifier = nn.Linear(model.config.hidden_size, self.num_classes)
        self.online_classifier_loss = eval_fn
        self.online_logit_fn= eval_logit_fn
        self.online_train_accuracy = torchmetrics.Accuracy(
                    task=eval_type, num_classes=self.num_classes, top_k=1
        )
        self.online_val_accuracy = torchmetrics.Accuracy(
                    task=eval_type, num_classes=self.num_classes, top_k=1
        ) 
        self.save_dir = save_dir
        self.train_losses = []
        self.avg_train_losses = []
        self.online_losses = []
        self.avg_online_losses = []
        self.performance = {}

    def forward(self, x):
        return self.model(self.transformation(x))

    def shared_step(self, batch: Tensor, stage: str = "train", batch_idx: int = None):
        if stage == "train":
            img, y, pc_mask = batch

            # mae training
            if self.masking.type == "pc":
                target  = (img.reshape([img.shape[0],-1]) @ self.masking_fn_[:,pc_mask])

                if self.masking.strategy in ["sampling_pc","pc"]:
                    indexes = self.indexes.to(self.device)
                    pc_mask_input = indexes[~torch.isin(indexes,pc_mask)]

                img     = ((img.reshape([img.shape[0],-1]) @ self.masking_fn_[:,pc_mask_input])@ self.masking_fn_[:,pc_mask_input].T).reshape(img.shape)

            elif self.masking.type == "pixel":
                if self.masking.strategy == "sampling":
                    self.model.config.mask_ratio = pc_mask
                    self.model.vit.embeddings.config.mask_ratio=pc_mask
                target = img

            outputs, cls = self.model(img,return_rep=False)
            reconstruction = self.model.unpatchify(outputs.logits)
            mask = outputs.mask.unsqueeze(-1).repeat(1, 1, self.model.config.patch_size**2 *3) 
            mask = self.model.unpatchify(mask)

            if self.masking.type == "pc":
                outputs.logits = reconstruction.reshape([img.shape[0],-1]) @ self.masking_fn_[:,pc_mask]
                outputs.mask = torch.zeros_like(mask.reshape([mask.shape[0],-1]),device=self.device)

            loss_mae = self.model.forward_loss(target,outputs.logits,outputs.mask,patchify=False if self.masking.type == "pc" else True)
            
            if (self.current_epoch+1)%self.eval_freq==0 and batch_idx==0:
                self.log(
                    f"{stage}_mae_loss", 
                    loss_mae, 
                    prog_bar=True,
                    sync_dist=False,
                    on_step=True,
                    on_epoch=False
                    )
                self.train_losses.append(loss_mae.item())
                self.avg_train_losses.append(np.mean(self.train_losses))
                plot_loss(self.avg_train_losses,name_loss="MSE",save_dir=self.save_dir,name_file="_train")
                plot_loss(self.avg_online_losses,name_loss="X-Ent",save_dir=self.save_dir,name_file="_train_online_cls")

                if (
                    self.model.config.mask_ratio is None
                    or self.model.config.mask_ratio > 0
                ):
                    save_reconstructed_images((-1*(mask[:10]-1))*img[:10],mask[:10]*img[:10], reconstruction[:10], self.current_epoch+1, self.save_dir,"train")
                else:
                    save_reconstructed_images(img[:10], target[:10], reconstruction[:10], self.current_epoch+1, self.save_dir,"train")


            # online classifier
            logits_cls = self.classifier(cls.detach())
            loss_ce = self.online_classifier_loss(logits_cls.squeeze(),y.squeeze())

            if (self.current_epoch+1)%self.eval_freq==0 and batch_idx==0:
                self.log(f"{stage}_classifier_loss", loss_ce, sync_dist=False, on_step=True, on_epoch=False)

                accuracy_metric = getattr(self, f"online_{stage}_accuracy")
                accuracy_metric(self.online_logit_fn(logits_cls.squeeze()), y.squeeze())
                self.log(
                    f"online_{stage}_accuracy",
                    accuracy_metric,
                    prog_bar=False,
                    sync_dist=True,
                )
                del logits_cls 

                self.online_losses.append(loss_ce.item())
                self.avg_online_losses.append(np.mean(self.online_losses))

                plot_loss(self.avg_online_losses,name_loss="X-Ent",save_dir=self.save_dir,name_file="_train_online_cls")

            return loss_mae + loss_ce

        else:
            img, y = batch
            cls, _ = self.model(img,return_rep=True)
            logits = self.classifier(cls.detach())

            accuracy_metric = getattr(self, f"online_{stage}_accuracy")
            accuracy_metric(self.online_logit_fn(logits.squeeze()), y.squeeze())
            self.log(
                f"online_{stage}_accuracy",
                accuracy_metric,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
            )

            if batch_idx == 0:
                if self.current_epoch+1 not in list(self.performance.keys()): 
                    self.performance[self.current_epoch+1]=[]
                    
            if len(y.squeeze().shape) > 1:
                self.performance[self.current_epoch+1].append(sum(sum(1*((self.online_logit_fn(logits.squeeze())>0.5)==y.squeeze()))).item())  
            else: 
                self.performance[self.current_epoch+1].append(sum(1*(torch.argmax(self.online_logit_fn(logits.squeeze()), dim=-1)==y.squeeze())).item())  

            return None

    def on_validation_epoch_end(self):
        self.performance[self.current_epoch+1] = sum(self.performance[self.current_epoch+1])/self.datamodule.num_val_samples
        plot_performance(list(self.performance.keys()),list(self.performance.values()),self.save_dir,name="val")

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="train", batch_idx=batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="val", batch_idx=batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="test", batch_idx=batch_idx)
        return loss

    def configure_optimizers(self):
        def warmup(current_step: int):
            return 1 / (10 ** (float(num_warmup_epochs - current_step)))

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.trainer.max_epochs, verbose=False
            )
        elif self.optimizer_name == "adamw_warmup":
            num_warmup_epochs = self.warm_up
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.betas
            )

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=warmup
            )

            train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.trainer.max_epochs, verbose=False
            )

            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, train_scheduler], [num_warmup_epochs]
            )

        else:
            raise ValueError(f"{self.optimizer_name} not supported")

        return [optimizer], [lr_scheduler]
