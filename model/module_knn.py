import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
import torchmetrics
from torch import Tensor
from typing import Optional, Dict, List, Any
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.models import resnet18
from torch import Tensor
import wandb
import os 
import matplotlib.pyplot as plt
import numpy as np
from utils import save_reconstructed_images, save_attention_maps, save_attention_maps_batch
from plotting import plot_loss, plot_performance
import csv

# Lightning Module definition
class ViTMAE_knn(pl.LightningModule):
    def __init__(
        self, 
        model, 
        k=[3,4,5,6,7,8,9,10,11,12,4,16,18,20],
        save_dir: str =None,
        evaluated_epoch: int =800,
        datamodule: Optional[pl.LightningDataModule] = None,
        task :int =0,
        ):
        super().__init__()
        self.model = model  # Your base model (e.g., ResNet or any other embedding model)
        self.model.config.mask_ratio = 0.0
        self.model.vit.embeddings.config.mask_ratio=0.0
        self.task=task

        self.k = k  # Number of nearest neighbors

        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes  # Number of classes for classification

        self.online_val_accuracy = torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_classes, top_k=1
        )
        self.classifier = nn.Linear(model.config.hidden_size, self.num_classes)

        self.data_embeddings = []
        self.data_labels = []

        self.save_dir = save_dir
        self.performance = {}
        self.evaluated_epoch = evaluated_epoch

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch: Tensor, stage: str = "train", batch_idx: int =None):
        if stage == "train":
            img, y, _ = batch
            cls, _ = self.model(img,return_rep=True)
            self.data_embeddings.append(cls.detach())
            self.data_labels.append(y)
        
            return None
        else:
            # Validation logic
            img, y = batch
            cls, _ = self.model(img,return_rep=True)
            distances = torch.cdist(cls.detach(), self.data_embeddings, p=2)  # L2 distance
            accuracy_metric = getattr(self, f"online_val_accuracy")

            # Get the indices of the k nearest neighbors
            for k in self.k:
                _, indices = torch.topk(distances, k=k, dim=1, largest=False)
                pred_labels, _ = torch.mode(self.data_labels[indices].squeeze())
                accuracy_metric(pred_labels.squeeze(), y.squeeze())

                self.log(
                        f"final_val_accuracy_{self.evaluated_epoch}_knn_k_{k}",
                        accuracy_metric,
                        prog_bar=True,
                        sync_dist=True,
                        on_epoch=True,
                        on_step=False,
                    )
                
                if batch_idx == 0 and (k not in list(self.performance.keys())): 
                    self.performance[k]=[]
                self.performance[k].append(sum(pred_labels.squeeze()==y.squeeze()).item())

            return None
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="train", batch_idx=batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="val", batch_idx=batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="test", batch_idx=batch_idx)
        return loss
    
    def on_validation_epoch_start(self,):
        self.data_embeddings = torch.cat(self.data_embeddings, dim=0)
        self.data_labels     = torch.cat(self.data_labels, dim=0)
        return 

    def on_validation_epoch_end(self):
        for k in self.k:
            self.performance[k] = sum(self.performance[k])/self.datamodule.num_val_samples
        if (self.current_epoch+1)%10 == 0:
            plot_performance(list(self.performance.keys()),list(self.performance.values()),self.save_dir,name=f"val_final_{self.evaluated_epoch}_knn")

    def on_fit_end(self):
        # Write to a CSV file
        with open(os.path.join(self.save_dir,f'performance_final_{self.evaluated_epoch}_knn_task_{self.task}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Eval Epoch', 'Test Accuracy'])
            for epoch in list(self.performance.keys()):
                # Assuming you have the accuracy for each epoch stored in a list
                writer.writerow([epoch, round(100*self.performance[epoch],2)])  
        return 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.classifier.parameters())
        return [optimizer]
    

    
