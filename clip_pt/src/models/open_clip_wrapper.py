import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy

from models.open_CLIP import OpenCLIP
from models.open_CLIP_adapter import OpenCLIPAdapter
from utils import utils
from utils.loss import clip_loss
from utils.scheduler import CosineWarmupLR, LinearLR


class OpenCLIPWrapper(pl.LightningModule):
    def __init__(
            self,
            config: DictConfig,
            train_size: int = 0,
            val_labels=None,
            carbon_tracker=None,
            model=None
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False
        if model is None:
            if config.get("model", None) is None:
                # model doesn't have adapters
                self.model = OpenCLIP()
            else:
                # model has adapters
                self.model = OpenCLIPAdapter(adapter=config.model.adapter)
        else:
            self.model = model

        self.carbon_tracker = carbon_tracker

        self.config = config
        self.train_size = train_size
        self.val_labels = val_labels

        n_classes = self.config["batch_size"] * self.config["accumulate_grad_batches"]
        self.image_train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.text_train_acc = Accuracy(task="multiclass", num_classes=n_classes)

        self.retrieval_val_acc = Accuracy(task="multiclass", num_classes=self.config["batch_size"])
        self.classification_val_acc = None if val_labels is None \
            else Accuracy(task="multiclass", num_classes=val_labels.shape[0])

        self.complete_training = False
        self.complete_validation = False

        # freezing image encoder
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
                T_max=utils.compute_n_batches(self.train_size,
                                              self.config.batch_size) * self.trainer.max_epochs
            )
        
        if self.config.scheduler.name.lower() == 'linearlr':
            scheduler = LinearLR(
                optimizer, 
                start_factor=self.config.scheduler.params["start_factor"], 
                end_factor=self.config.scheduler.params["end_factor"], 
                total_iters=self.config.scheduler.params["total_iters"], 
                last_epoch=-1, 
                verbose=False
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

        image_features, text_features = self.model.encode(train_batch)
        logits_per_image, logits_per_text = self.model.compute_logits(image_features,
                                                                      text_features,
                                                                      fixed_logit=False)
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

        self.log("train/loss", loss)
        self.log("train/batch_image_accuracy", batch_image_accuracy)
        self.log("train/batch_text_accuracy", batch_text_accuracy)

    def on_train_epoch_end(self):
        epoch_image_accuracy = self.image_train_acc.compute()
        epoch_text_accuracy = self.text_train_acc.compute()

        self.log("train/epoch_image_accuracy", epoch_image_accuracy)
        self.log("train/epoch_text_accuracy", epoch_text_accuracy)

        self.image_train_acc.reset()
        self.text_train_acc.reset()

    def validation_step(self, val_batch, batch_idx, dataset_idx, dataloader_idx=0):
        if dataset_idx == 0:
            self.val_retrieval(val_batch)
        else:
            self.val_classification(val_batch)

    def val_retrieval(self, batch):
        image_features, text_features = self.model.encode(batch)
        logits_per_image, logits_per_text = self.model.compute_logits(image_features,
                                                                      text_features,
                                                                      fixed_logit=False)
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
        batch = image_input, text_input

        image_features, text_features = self.model.encode(batch)
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
