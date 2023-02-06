import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from models.model import CLIPTBR
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.loss import clip_loss


class CLIPPTBRWrapper(pl.LightningModule):
    def __init__(
            self,
            config: DictConfig
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = CLIPTBR(vision_encoder_version=config.model.image_encoder,
                             text_encoder_version=config.model.text_encoder,
                             pretraining=config.model.pretraining)
        self.config = config

        n_classes = self.config["batch_size"] * self.config["accumulate_grad_batches"]
        self.image_train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.image_val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.text_train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.text_val_acc = Accuracy(task="multiclass", num_classes=n_classes)

        self.image_feature_list = []
        self.text_feature_list = []
        self.valid_image_feature_list = []
        self.valid_text_feature_list = []

        self.complete_training = False
        self.complete_validation = False
        self.unfreeze = config.model["warmup_steps"] > 0
        if self.unfreeze:
            print("Freezing model!!")
            self.model.freeze()

    def configure_optimizers(self):
        opt_params = self.config.optimizer["params"]
        optimizer = Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": opt_params["learning_rate"]
                }
            ],
            eps=opt_params["eps"],
            betas=opt_params["betas"],
            weight_decay=opt_params["weight_decay"]
        )

        if not self.config['scheduler']:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
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
        # warmup
        if self.trainer.global_step >= self.config.model["warmup_steps"] and self.unfreeze:
            print(f"Epoch {self.current_epoch}: unfreezing model")
            self.model.unfreeze()
            self.unfreeze = False

        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        mb_image_features, mb_text_features = self.model(train_batch)

        self.image_feature_list.append(mb_image_features)
        self.text_feature_list.append(mb_text_features)

        if (batch_idx + 1) % self.config["accumulate_grad_batches"] == 0:
            image_features = torch.concat(self.image_feature_list, dim=0)
            text_features = torch.concat(self.text_feature_list, dim=0)
            logits_per_image, logits_per_text = self.model.compute_logits(image_features,
                                                                          text_features,
                                                                          fixed_logit=True)
            loss = clip_loss(logits_per_text)
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step(loss)

            preds_image = logits_per_image.argmax(dim=1)
            preds_text = logits_per_text.argmax(dim=1)
            ground_truth = torch.arange(len(logits_per_image), dtype=torch.long,
                                        device=logits_per_image.device)

            batch_image_accuracy = self.image_train_acc(preds_image, ground_truth)
            batch_text_accuracy = self.text_train_acc(preds_text, ground_truth)
            batch_text_image = (batch_image_accuracy + batch_text_accuracy) / 2.0

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

            self.log("train/epoch_image_accuracy", epoch_image_accuracy)
            self.log("train/epoch_text_accuracy", epoch_text_accuracy)
            self.log("train/epoch_text_image_accuracy", epoch_text_image_accuracy)

            self.image_train_acc.reset()
            self.text_train_acc.reset()

    def validation_step(self, val_batch, batch_idx):
        mb_image_features, mb_text_features = self.model(val_batch)

        self.valid_image_feature_list.append(mb_image_features)
        self.valid_text_feature_list.append(mb_text_features)

        if (batch_idx + 1) % self.config["accumulate_grad_batches"] == 0:
            image_features = torch.concat(self.valid_image_feature_list, dim=0)
            text_features = torch.concat(self.valid_text_feature_list, dim=0)
            logits_per_image, logits_per_text = self.model.compute_logits(image_features,
                                                                          text_features,
                                                                          fixed_logit=True)

            loss = clip_loss(logits_per_text)

            ground_truth = torch.arange(len(logits_per_image), dtype=torch.long,
                                        device=logits_per_image.device)

            preds_image = logits_per_image.argmax(dim=1)
            preds_text = logits_per_text.argmax(dim=1)

            batch_image_accuracy = self.image_val_acc(preds_image, ground_truth)
            batch_text_accuracy = self.text_val_acc(preds_text, ground_truth)
            batch_text_image = (batch_image_accuracy + batch_text_accuracy) / 2.0

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
