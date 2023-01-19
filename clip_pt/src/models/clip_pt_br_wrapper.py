import torch
import numpy as np
from torchmetrics import Accuracy
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl

from models.model import CLIPTBR
from utils.loss import clip_loss


class CLIPPTBRWrapper(pl.LightningModule):
    def __init__(
        self,
        config: dict
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model = CLIPTBR()
        self.config = config

        self.image_train_acc = Accuracy(task="multiclass",num_classes=self.config["batch_size"]*self.config["accumulate_grad_batches"])
        self.image_val_acc = Accuracy(task="multiclass",num_classes=self.config["batch_size"]*self.config["accumulate_grad_batches"])
        self.text_train_acc = Accuracy(task="multiclass",num_classes=self.config["batch_size"]*self.config["accumulate_grad_batches"])
        self.text_val_acc = Accuracy(task="multiclass",num_classes=self.config["batch_size"]*self.config["accumulate_grad_batches"])

        self.image_feature_list = []
        self.text_feature_list = []
        self.valid_image_feature_list = []
        self.valid_text_feature_list = []

        self.complete_training = False
        self.complete_validation = False

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def configure_optimizers(self):
        optimizer = Adam(
            [
                {
                    'params': self.model.parameters(),
                    'lr': self.config["learning_rate"]
                }
            ],
                eps=self.config["eps"],
                betas=self.config["betas"],
                weight_decay=self.config["weight_decay"]
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=self.config["scheduler_patience"],
            factor=0.9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


    def training_step(self, train_batch, batch_idx):

        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        mb_image_features, mb_text_features = self.model(train_batch)

        self.image_feature_list.append(mb_image_features)
        self.text_feature_list.append(mb_text_features)

        # print("train", batch_idx + 1, self.config["accumulate_grad_batches"], (batch_idx + 1) % self.config["accumulate_grad_batches"])

        if (batch_idx + 1) % self.config["accumulate_grad_batches"] == 0:
            image_features = torch.concat(self.image_feature_list, dim=0)
            text_features = torch.concat(self.text_feature_list, dim=0)
            logits_per_image, logits_per_text = self.model.compute_logits(
                image_features,
                text_features,
                fixed_logit=-1
            )

            loss = clip_loss(logits_per_text)

            ground_truth = torch.arange(
                len(logits_per_image),
                dtype=torch.long,
                device=logits_per_image.device
            )

            preds_image = logits_per_image.argmax(dim=1)
            preds_text = logits_per_text.argmax(dim=1)

            # # check parameters with no grad
            # for n, p in model.named_parameters():
            #     if p.grad is None and p.requires_grad is True:
            #         print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model 

            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            lr_scheduler.step(loss)

            batch_image_accuracy = self.image_train_acc(preds_image, ground_truth)
            batch_text_accuracy = self.text_train_acc(preds_text, ground_truth)
            batch_text_image = (batch_image_accuracy + batch_text_accuracy)/2.0

            self.log("train/loss", loss)
            self.log("train/batch_image_accuracy", batch_image_accuracy)
            self.log("train/batch_text_accuracy", batch_text_accuracy)
            self.log("train/batch_text_image_accuracy", batch_text_image)

            self.image_feature_list = []
            self.text_feature_list = []

            self.complete_training = True


    def training_epoch_end(self, batch_parts):
        self.image_feature_list = []
        self.text_feature_list = []

        if self.complete_training:
            epoch_image_accuracy = self.image_train_acc.compute()
            epoch_text_accuracy = self.text_train_acc.compute()
            epoch_text_image_accuracy = (epoch_image_accuracy + epoch_text_accuracy) / 2.0

            # print(epoch_image_accuracy, epoch_text_accuracy, epoch_text_image_accuracy)

            self.log("train/epoch_image_accuracy", epoch_image_accuracy)
            self.log("train/epoch_text_accuracy", epoch_text_accuracy)
            self.log("train/epoch_text_image_accuracy", epoch_text_image_accuracy)

            self.image_train_acc.reset()
            self.text_train_acc.reset()


    def validation_step(self, val_batch, batch_idx):
        mb_image_features, mb_text_features = self.model(val_batch)

        self.valid_image_feature_list.append(mb_image_features)
        self.valid_text_feature_list.append(mb_text_features)

        # print("Val", batch_idx + 1, self.config["accumulate_grad_batches"], (batch_idx + 1) % self.config["accumulate_grad_batches"])

        if (batch_idx + 1) % self.config["accumulate_grad_batches"] == 0:
            image_features = torch.concat(self.valid_image_feature_list, dim=0)
            text_features = torch.concat(self.valid_text_feature_list, dim=0)
            logits_per_image, logits_per_text = self.model.compute_logits(
                image_features,
                text_features,
                fixed_logit=-1
            )

            loss = clip_loss(logits_per_text)

            ground_truth = torch.arange(
                len(logits_per_image),
                dtype=torch.long,
                device=logits_per_image.device
            )

            preds_image = logits_per_image.argmax(dim=1)
            preds_text = logits_per_text.argmax(dim=1)

            batch_image_accuracy = self.image_val_acc(preds_image, ground_truth)
            batch_text_accuracy = self.text_val_acc(preds_text, ground_truth)
            batch_text_image = (batch_image_accuracy + batch_text_accuracy)/2.0

            # print("Val", batch_image_accuracy, batch_text_accuracy, batch_text_image)

            self.log("val/loss", loss)
            self.log("val/batch_image_accuracy", batch_image_accuracy)
            self.log("val/batch_text_accuracy", batch_text_accuracy)
            self.log("val/batch_text_image_accuracy", batch_text_image)

            self.valid_image_feature_list = []
            self.valid_text_feature_list = []

            self.complete_validation = True


    def validation_epoch_end(self, batch_parts):
        self.valid_image_feature_list = []
        self.valid_text_feature_list = []

        if self.complete_validation:
            epoch_image_accuracy = self.image_val_acc.compute()
            epoch_text_accuracy = self.text_val_acc.compute()
            epoch_text_image_accuracy = (epoch_image_accuracy + epoch_text_accuracy) / 2.0

            self.log("val/epoch_image_accuracy", epoch_image_accuracy)
            self.log("val/epoch_text_accuracy", epoch_text_accuracy)
            self.log("val/epoch_text_image_accuracy", epoch_text_image_accuracy)

            self.image_val_acc.reset()
            self.text_val_acc.reset()