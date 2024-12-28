import pytorch_lightning as pl
import torchmetrics
from torch import Tensor
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from model_zoo.scattering_network import Scattering2dResNet
from torchvision.models import resnet18
from torch import Tensor
import wandb
import os 
import matplotlib.pyplot as plt
import numpy as np
from utils import save_reconstructed_images, save_attention_maps, save_attention_maps_batch
from plotting import plot_loss, plot_performance
import csv

class ViTMAE_mlp(pl.LightningModule):

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
        save_dir: str =None,
        evaluated_epoch: int =800,
        eval_type ="multiclass",
        eval_fn =nn.CrossEntropyLoss(),
        eval_logit_fn = nn.Softmax(),
        task :int =0,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.optimizer_name = optimizer_name
        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes
        self.image_size = datamodule.image_size
        self.warm_up = warmup
        self.evaluated_epoch = evaluated_epoch
        self.task = task

        model.delete_decoder()
        self.model = model
        self.model.config.mask_ratio = 0.0
        self.model.vit.embeddings.config.mask_ratio=0.0

        self.classifier = nn.Sequential(
            nn.Linear(model.config.hidden_size, model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(model.config.hidden_size, model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(model.config.hidden_size, self.num_classes)
        )

        self.online_classifier_loss = eval_fn
        self.online_logit_fn= eval_logit_fn
        self.online_train_accuracy = torchmetrics.Accuracy(
                    task=eval_type, num_classes=self.num_classes, top_k=1
        )
        self.online_val_accuracy = torchmetrics.Accuracy(
                    task=eval_type, num_classes=self.num_classes, top_k=1
        )
        self.online_val_f1 = torchmetrics.F1Score(
                    task=eval_type, num_classes=self.num_classes, top_k=1, average = "macro"
        ) 

        self.save_dir = save_dir
        self.train_losses = []
        self.avg_train_losses = []
        self.performance = {}
        self.f1scores = {}

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch: Tensor, stage: str = "train", batch_idx: int =None):
        if stage == "train":
            img, y, _ = batch
            if len(y.shape)>1:
                y = y[:,self.task]
            cls, _ = self.model(img,return_rep=True)
            logits = self.classifier(cls.detach())

            loss_ce = self.online_classifier_loss(logits.squeeze(),y.squeeze())
            self.log(f"final_{stage}_classifier_loss_{self.evaluated_epoch}_mlp", loss_ce, sync_dist=True)

            accuracy_metric = getattr(self, f"online_{stage}_accuracy")
            accuracy_metric(self.online_logit_fn(logits.squeeze()), y.squeeze())
            self.log(
                f"final_{stage}_accuracy_{self.evaluated_epoch}_mlp",
                accuracy_metric,
                prog_bar=False,
                sync_dist=True,
            )

            self.train_losses.append(loss_ce.item())
            self.avg_train_losses.append(np.mean(self.train_losses))

            if (self.current_epoch+1)%10==0 and batch_idx==0:
                plot_loss(self.avg_train_losses,name_loss="X-Entropy",save_dir=self.save_dir,name_file=f"_eval_train_{self.evaluated_epoch}_mlp")
            return  loss_ce

        else:
            img, y = batch

            if len(y.shape)>1:
                y = y[:,self.task]

            cls, attentions = self.model(img,return_rep=True,output_attentions=True)
            logits = self.classifier(cls.detach())

            accuracy_metric = getattr(self, f"online_{stage}_accuracy")
            accuracy_metric(self.online_logit_fn(logits.squeeze()), y.squeeze())
            self.log(
                f"final_{stage}_accuracy_{self.evaluated_epoch}_mlp",
                accuracy_metric,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
            )

            if batch_idx == 0 and (self.current_epoch+1) not in list(self.performance.keys()): 
                self.performance[self.current_epoch+1]=[]
                if len(y.squeeze().shape) >1:
                    self.f1scores[self.current_epoch+1]=[]

            if len(y.squeeze().shape) > 1:
                f1_metric = getattr(self, f"online_{stage}_f1")
                f1_score = f1_metric(self.online_logit_fn(logits.squeeze()), y.squeeze())
                self.performance[self.current_epoch+1].append(sum(1*((self.online_logit_fn(logits.squeeze())>0.5)==y.squeeze())).detach().cpu().numpy())  
                self.f1scores[self.current_epoch+1].append(f1_score.detach().cpu().numpy())  

            else:
                self.performance[self.current_epoch+1].append(sum(1*(torch.argmax(logits.squeeze(), dim=-1)==y.squeeze())).item())  

            # check the attention we get at final
            if self.current_epoch==0 and batch_idx==0:
                attentions = attentions[-1].mean(1)
                att_map_cls = attentions[:,0,1:]
                att_map_spatial = torch.mean(attentions[:,1:,1:],dim=-1)
                att_map_cls = att_map_cls.reshape([img.shape[0],int(np.sqrt(att_map_cls.shape[-1])),int(np.sqrt(att_map_cls.shape[-1]))])
                att_map_spatial = att_map_spatial.reshape([img.shape[0],int(np.sqrt(att_map_spatial.shape[-1])),int(np.sqrt(att_map_spatial.shape[-1]))])
                save_attention_maps(img[:10],att_map_cls[:10].unsqueeze(1),att_map_spatial[:10].unsqueeze(1),self.current_epoch+1, self.save_dir,f"eval_{self.evaluated_epoch}_mlp")
                save_attention_maps_batch(att_map_cls=att_map_cls,att_map_spatial=att_map_spatial,epoch=self.current_epoch+1, output_dir=self.save_dir,name=f"eval_{self.evaluated_epoch}_mlp")

            return None    

    def on_validation_epoch_end(self):
        if isinstance(self.performance[self.current_epoch+1][0],np.ndarray):
            self.performance[self.current_epoch+1] = np.array(self.performance[self.current_epoch+1])
            self.f1scores[self.current_epoch+1] = np.array(self.f1scores[self.current_epoch+1])

        self.performance[self.current_epoch+1] = sum(self.performance[self.current_epoch+1])/self.datamodule.num_val_samples

        if isinstance(self.performance[self.current_epoch+1],np.ndarray):
           self.performance[self.current_epoch+1] = np.mean(self.performance[self.current_epoch+1]) 
           self.f1scores[self.current_epoch+1] = np.mean(self.f1scores[self.current_epoch+1])

        if (self.current_epoch+1)%10 == 0 and not isinstance(self.performance[self.current_epoch+1],np.ndarray):
            plot_performance(list(self.performance.keys()),list(self.performance.values()),self.save_dir,name=f"val_final_{self.evaluated_epoch}_mlp")

    def on_fit_end(self):
        # Write to a CSV file
        with open(os.path.join(self.save_dir,f'performance_final_{self.evaluated_epoch}_mlp_task_{self.task}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Eval Epoch', 'Test Accuracy'])
            for epoch in list(self.performance.keys()):
                # Assuming you have the accuracy for each epoch stored in a list
                writer.writerow([epoch, round(100*self.performance[epoch],2)])  

        if len(self.f1scores.keys()) > 1 :
            with open(os.path.join(self.save_dir,f'f1scores_final_{self.evaluated_epoch}_mlp_task_{self.task}.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Eval Epoch', 'Test Accuracy'])
                for epoch in list(self.f1scores.keys()):
                    # Assuming you have the accuracy for each epoch stored in a list
                    writer.writerow([epoch, round(100*self.f1scores[epoch],2)])    
        return 

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
                self.classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.trainer.max_epochs, verbose=False
            )
        elif self.optimizer_name == "adamw_warmup":
            num_warmup_epochs = self.warm_up
            optimizer = torch.optim.AdamW(
                self.classifier.parameters(),
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
        elif self.optimizer_name == "sgd":
            num_warmup_epochs = self.warm_up
            optimizer = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.betas
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

