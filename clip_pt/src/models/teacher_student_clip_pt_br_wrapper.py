import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam

from models.teacher_student_model import TeacherStudentCLIPTBR
from utils.scheduler import CosineWarmupLR


class TeacherStudentCLIPPTBRWrapper(pl.LightningModule):
    def __init__(
            self,
            config: DictConfig,
            train_size: int,
            carbon_tracker
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False
        self.model = TeacherStudentCLIPTBR(teacher_version=config.model.teacher,
                                           student_version=config.model.student)
        self.config = config
        self.train_size = train_size
        self.carbon_tracker = carbon_tracker
        self.loss = nn.MSELoss()

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
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        student_output, teacher_output = self.model(train_batch)
        loss = self.loss(student_output, teacher_output)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step(loss)

        self.log("train/loss", loss)

    def validation_step(self, val_batch, batch_idx):
        student_output, teacher_output = self.model(val_batch)
        loss = self.loss(student_output, teacher_output)
        self.log("val/loss", loss)
