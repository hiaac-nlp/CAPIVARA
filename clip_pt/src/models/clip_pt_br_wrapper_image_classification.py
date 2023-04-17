import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy

from models.model import CLIPTBR
from utils.loss import clip_loss
from utils.scheduler import CosineWarmupLR


class CLIPPTBRWrapperImageClassification(pl.LightningModule):
    def __init__(
            self,
            config: DictConfig,
            train_size: int,
            val_labels,
            carbon_tracker
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False
        self.model = CLIPTBR(vision_encoder_version=config.model.image_encoder,
                             text_encoder_version=config.model.text_encoder,
                             pretraining=config.model.pretraining,
                             adapter=config.model.get("adapter", None))
        self.config = config
        self.train_size = train_size
        self.val_labels = val_labels
        self.carbon_tracker = carbon_tracker

        n_classes = self.config["batch_size"] * self.config["accumulate_grad_batches"]
        self.image_train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.text_train_acc = Accuracy(task="multiclass", num_classes=n_classes)

        self.retrieval_val_acc = Accuracy(task="multiclass", num_classes=self.config["batch_size"])
        self.classification_val_acc = None if val_labels is None \
            else Accuracy(task="multiclass", num_classes=val_labels['input_ids'].shape[0])

        self.image_feature_list = []
        self.text_feature_list = []

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

        scheduler = None
        if self.config.scheduler.name.lower() == 'reducelronplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=self.config.scheduler.params["patience"],
                factor=0.9,
                min_lr=1.0e-6
            )

        if self.config.scheduler.name.lower() == "cosinewarmuplr":
            scheduler = CosineWarmupLR(
                optimizer,
                lr_min=1.0e-6,
                lr_max=opt_params["learning_rate"],
                warmup=self.config.scheduler.params["warmup_lr"],
                T_max=self.train_size * self.trainer.max_epochs
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.model.logit_scale.data.clamp_(0, np.log(100))

    def training_step(self, train_batch, batch_idx):
        # warmup
        if self.trainer.global_step >= self.config.model["warmup_steps"] and self.unfreeze:
            print(f"Epoch {self.current_epoch}: unfreezing model!!")
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

    def on_train_epoch_end(self):
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

            our_emission = self.carbon_tracker.flush()
            our_energy = self.carbon_tracker._total_energy.__float__()
            self.log('carbon/Carbon Emission(CodeCarbon)', our_emission)
            self.log('carbon/Carbon Emission', self.config.carbon["brazil_carbon_intensity"] * our_energy)
            self.log('carbon/Spent energy', our_energy)

    def validation_step(self, val_batch, batch_idx, dataset_idx, dataloader_idx=0):
        if dataset_idx == 0:
            self.val_retrieval(val_batch)
        else:
            self.val_classification(val_batch)

    def val_retrieval(self, batch):
        image_features, text_features = self.model(batch)
        logits_per_image, logits_per_text = self.model.compute_logits(image_features,
                                                                      text_features)
        loss = clip_loss(logits_per_image)

        ground_truth = torch.arange(len(logits_per_image), dtype=torch.long,
                                    device=logits_per_image.device)
        preds_image = logits_per_image.argmax(dim=1)
        batch_image_accuracy = self.retrieval_val_acc(preds_image, ground_truth)
        self.log("val/retrieval_loss", loss)
        self.log("val/batch_retrieval_acc", batch_image_accuracy)

    def val_classification(self, batch):
        image_input, label_idx = batch
        text_input = self.val_labels
        text_input = text_input.to(self.device)
        label_idx = label_idx.to(self.device)
        image_input["pixel_values"] = image_input["pixel_values"].squeeze(1)
        batch = image_input, text_input
        image_features, text_features = self.model(batch)
        logits_per_image, _ = self.model.compute_logits(image_features, text_features,
                                                        fixed_logit=False)
        loss = nn.functional.cross_entropy(logits_per_image, label_idx)
        preds_image = logits_per_image.argmax(dim=1)
        image_accuracy = self.classification_val_acc(preds_image, label_idx)
        self.log("val/batch_classification_acc", image_accuracy)
        self.log("val/classification_loss", loss)

    def on_validation_epoch_end(self):
        epoch_retrieval_acc = self.retrieval_val_acc.compute()
        epoch_classification_acc = self.classification_val_acc.compute()

        self.log("val/epoch_retrieval_acc", epoch_retrieval_acc)
        self.log("val/epoch_classification_acc", epoch_classification_acc)

        self.retrieval_val_acc.reset()
        self.classification_val_acc.reset()

        our_emission = self.carbon_tracker.flush()
        our_energy = self.carbon_tracker._total_energy.__float__()
        self.log('carbon/Carbon Emission(CodeCarbon)', our_emission)
        self.log('carbon/Carbon Emission', self.config.carbon["brazil_carbon_intensity"] * our_energy)
        self.log('carbon/Spent energy', our_energy)

