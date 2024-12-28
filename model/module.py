import pytorch_lightning as pl
import torchmetrics
from torch import Tensor
import torch
from torch import nn
import torch.nn as nn
from typing import Optional, Dict, List, Any
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from model_zoo.scattering_network import Scattering2dResNet
from torchvision.models import resnet18
from torch import Tensor
import wandb
import os 
import time
import matplotlib.pyplot as plt
import numpy as np
from utils import save_reconstructed_images, save_attention_maps, save_attention_maps_batch
from plotting import plot_loss, plot_performance
# import kornia.augmentation as K_transformations
# from kornia.constants import Resample
from dataset.CLEVRCustomDataset import CLEVRCustomDataset

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
            
            # size = self.image_size
            # if isinstance(size, int):
            #     size = (size, size)
            # self.random_resized_crop = K_transformations.RandomResizedCrop(
            #     size=tuple(size),
            #     scale=(0.2,1.0),
            #     resample=Resample.BICUBIC.name
            # )
        
        elif self.masking.type == "random":
            self.register_buffer("masking_fn",nn.Linear())
        elif self.masking.type == "segmentation":
            patch_size = self.model.config.patch_size
            self.mask_patch_pool = torch.nn.MaxPool2d(
                (patch_size, patch_size), stride=(patch_size, patch_size)
            )

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

    def get_current_masking_function(self, seg_mask):
        if self.masking.strategy == 'complete':
            seg_mask = seg_mask.max(dim=1).values.float()
            patched_seg_mask = self.mask_patch_pool(seg_mask[:, 0]).flatten(1)
        else:
            # Patch each individual segmentation mask
            patched_seg_mask = (
                self.mask_patch_pool(seg_mask[:, :, 0].float()).flatten(2)
            )
            batch_size, n_masks, n_patches = patched_seg_mask.shape
            # Get maximum number of patches out of any segmentation mask
            max_patches_per_element = patched_seg_mask.sum(dim=-1).max().int()
            # Create random order overlapping with the segmentation mask
            segmentation_noise = torch.rand_like(patched_seg_mask) * patched_seg_mask
            sorted_order = torch.argsort(segmentation_noise, dim=-1, descending=True).to(seg_mask.device)
            # Randomly select patches from the segmentation mask to mask out
            num_patches_per_mask = torch.randint(
                low=0,
                high=max_patches_per_element,
                size=(batch_size, n_masks, int(self.datamodule.masking.pixel_ratio*max_patches_per_element)),
            ).to(seg_mask.device)
            # Gather indices to and set them to zero
            index_to_mask = torch.gather(sorted_order, dim=-1, index=num_patches_per_mask).long()
            patched_seg_mask = patched_seg_mask.scatter(
                dim=-1, index=index_to_mask, value=0
            )
            if self.masking.strategy == "partial":
                # Select random index to keep for each object
                idx_to_keep = sorted_order[:, :, 0, None]
                patched_seg_mask = patched_seg_mask.scatter(
                    dim=-1, index=idx_to_keep, value=1
                )

            # Collapse segmentation mask to get a single mask per sample
            patched_seg_mask = patched_seg_mask.max(dim=1).values

        # Invert mask because we keep patches with 0
        patched_seg_mask = 1 - patched_seg_mask
        
        # Get indices such that, per batch, the first indices correspond
        # to '0', i.e., keep, and the last indices to '1', i.e., mask within the patched_seg_mask
        ids_sort = torch.argsort(patched_seg_mask, dim=1).to(
            patched_seg_mask.device
        ) 
        # Keep as many patches as the maximum selected by any mask
        len_keep = patched_seg_mask.sum(dim=1).max().int()
        ids_keep = ids_sort[:, :len_keep] 

        # Set correct masking ratio
        ratio = patched_seg_mask.sum()/(patched_seg_mask.shape[0]*patched_seg_mask.shape[1])
        self.model.config.mask_ratio = ratio
        self.model.vit.embeddings.config.mask_ratio=ratio

        def masking_fn(sequence, noise=None):
            sequence_unmasked = torch.gather(
                sequence,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, sequence.shape[-1]),
            )
            # Restore original mask from sorted mask/sequence
            ids_restore = torch.argsort(ids_sort, dim=1).to(patched_seg_mask.device)
            return sequence_unmasked, patched_seg_mask, ids_restore
        # Get all patch indices that are kept but should not be masked
        batch_idx, patch_idx = torch.where(
            torch.gather(patched_seg_mask, dim=1, index=ids_keep) == 1
        )
        # Construct a head mask to mask out cross attention with invalid patches
        head_mask = torch.ones(seg_mask.shape[0], len_keep+1, len_keep+1)
        head_mask[batch_idx, patch_idx+1, :] = 0
        head_mask[batch_idx, :, patch_idx+1] = 0
        head_mask = head_mask[None].repeat(self.model.vit.config.num_hidden_layers, 1, 1, 1)
        head_mask = head_mask[:, :, None]
        head_mask = head_mask.to(seg_mask.device)

        return masking_fn, head_mask

    def shared_step(self, batch: Tensor, stage: str = "train", batch_idx: int = None):
        if stage == "train":
            img, y, pc_mask = batch

            # mae training
            head_mask = None # Masking sequence after self attention heads
            if self.masking.type == "pc":
                pc_mask = pc_mask[0]
                target  = (img.reshape([img.shape[0],-1]) @ self.masking_fn_[:,pc_mask])

                if self.masking.strategy in ["sampling_pc","sampling_rest_pc","pc"]:
                    indexes = torch.arange(self.masking_fn_.shape[1],device=self.device)
                    pc_mask_input = indexes[~torch.isin(indexes,pc_mask[pc_mask!=-1])]
                img     = (img.reshape([img.shape[0],-1]) @ self.masking_fn_[:,pc_mask_input] @ self.masking_fn_[:,pc_mask_input].T).reshape(img.shape)

            elif self.masking.type == "pixel":
                if self.masking.strategy == "sampling":
                    self.model.config.mask_ratio = pc_mask[0]
                    self.model.vit.embeddings.config.mask_ratio=pc_mask[0]
                target = img
            elif self.masking.type == "segmentation":
                original_masking_fn = self.model.vit.embeddings.random_masking
                self.model.vit.embeddings.random_masking, head_mask = (
                    self.get_current_masking_function(pc_mask)
                )
                target = img

            outputs, cls = self.model(img,return_rep=False, head_mask=head_mask)
            if self.masking.type == "segmentation":
                self.model.vit.embeddings.random_masking = original_masking_fn

            reconstruction = self.model.unpatchify(outputs.logits)
            mask = outputs.mask.unsqueeze(-1).repeat(1, 1, self.model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
            mask = self.model.unpatchify(mask)

            if self.masking.type == "pc":
                outputs.logits = reconstruction.reshape([img.shape[0],-1]) @ self.masking_fn_[:,pc_mask]
                outputs.mask = torch.zeros_like(mask.reshape([mask.shape[0],-1]),device=self.device)

            loss_mae = self.model.forward_loss(target,outputs.logits,outputs.mask,patchify=False if self.masking.type == "pc" else True)

            self.log(
                f"{stage}_mae_loss", 
                loss_mae, 
                prog_bar=True,
                sync_dist=True,
                on_step=False,
                on_epoch=True
                )

            self.train_losses.append(loss_mae.item())
            self.avg_train_losses.append(np.mean(self.train_losses))

            if (self.current_epoch+1)%self.eval_freq==0 and batch_idx==0:
                plot_loss(self.avg_train_losses,name_loss="MSE",save_dir=self.save_dir,name_file="_train")
                plot_loss(self.avg_online_losses,name_loss="X-Ent",save_dir=self.save_dir,name_file="_train_online_cls")

                if (
                    self.model.config.mask_ratio is None
                    or self.model.config.mask_ratio > 0
                ):
                    save_reconstructed_images((-1*(mask[:10]-1))*img[:10],mask[:10]*img[:10], reconstruction[:10], self.current_epoch+1, self.save_dir,"train")
                else:
                    save_reconstructed_images(img[:10], target[:10], reconstruction[:10], self.current_epoch+1, self.save_dir,"train")

            del mask, reconstruction

            # online classifier
            logits_cls = self.classifier(cls.detach())
            loss_ce = self.online_classifier_loss(logits_cls.squeeze(),y.squeeze())

            self.log(f"{stage}_classifier_loss", loss_ce, sync_dist=True)
            self.online_losses.append(loss_ce.item())
            self.avg_online_losses.append(np.mean(self.online_losses))

            accuracy_metric = getattr(self, f"online_{stage}_accuracy")
            accuracy_metric(self.online_logit_fn(logits_cls.squeeze()), y.squeeze())
            self.log(
                f"online_{stage}_accuracy",
                accuracy_metric,
                prog_bar=False,
                sync_dist=True,
            )
            del logits_cls 

            if (self.current_epoch+1)%self.eval_freq==0 and batch_idx==0:
                plot_loss(self.avg_online_losses,name_loss="X-Ent",save_dir=self.save_dir,name_file="_train_online_cls")

            return loss_mae + loss_ce

        else:
            img, y = batch
            # CLEVR
            # if y.shape[1] > 1:
            #     y = y[:,:1]
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
