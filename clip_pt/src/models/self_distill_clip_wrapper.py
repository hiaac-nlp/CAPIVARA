import copy

import torch
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
from models.open_clip_wrapper import OpenCLIPWrapper
from utils.loss import clip_loss


class SelfDistillCLIPWrapper(OpenCLIPWrapper):
    def __init__(
            self,
            config: DictConfig,
            train_size: int = 0,
            val_labels=None,
            carbon_tracker=None,
            model=None
    ):
        super().__init__(config, train_size=train_size, val_labels=val_labels, model=model,
                         carbon_tracker=carbon_tracker)
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(log_target=True, reduction="batchmean")
        if isinstance(config.alpha, DictConfig):
            self.alpha_constant = False
            self.max_alpha = config.alpha.max_alpha
            self.min_alpha = config.alpha.min_alpha
            self.warmup_start = config.alpha.warmup_start
            self.warmup_end = config.alpha.warmup_end
        else:
            self.alpha_constant = True
            self.alpha = config.alpha
            assert 0 <= self.alpha <= 1, "alpha must be in the range [0,1]"

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        image_features, text_pt_features, text_en_features = self.extract_features(train_batch)
        loss, logits_per_image_pt, logits_per_text_pt = self.compute_loss(image_features=image_features,
                                                                          text_pt_features=text_pt_features,
                                                                          text_en_features=text_en_features)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step(loss)

        preds_image = logits_per_image_pt.argmax(dim=1)
        preds_text = logits_per_text_pt.argmax(dim=1)
        ground_truth = torch.arange(len(logits_per_image_pt), dtype=torch.long,
                                    device=logits_per_image_pt.device)

        batch_image_accuracy = self.image_train_acc(preds_image, ground_truth)
        batch_text_accuracy = self.text_train_acc(preds_text, ground_truth)

        self.log("train/loss", loss)
        self.log("train/batch_image_accuracy", batch_image_accuracy)
        self.log("train/batch_text_accuracy", batch_text_accuracy)

    def extract_features(self, train_batch):
        image_input, text_pt_input, text_en_input = train_batch
        image_features, text_pt_features = self.model.encode((image_input, text_pt_input))
        if self.config.self_distill == "complete":
            text_en_features = self.model.encode_text(text_en_input)
        else:
            with torch.no_grad():
                text_en_features = self.model.encode_text(text_en_input)
        return image_features, text_pt_features, text_en_features

    def compute_alpha(self):
        if self.alpha_constant:
            return self.alpha

        if self.trainer.global_step <= self.warmup_start:
            return self.min_alpha

        if self.warmup_start < self.trainer.global_step < self.warmup_end:
            return self.min_alpha + (self.trainer.global_step - self.warmup_start) * \
                (self.max_alpha - self.min_alpha) / (self.warmup_end - self.warmup_start)

        return self.max_alpha

    def compute_loss(self, image_features, text_pt_features, text_en_features):
        logits_per_image_pt, logits_per_text_pt = self.model.compute_logits(image_features,
                                                                            text_pt_features,
                                                                            fixed_logit=False)
        contrastive_loss = clip_loss(logits_per_text_pt)

        alpha = self.compute_alpha()

        if self.config.self_distill == "kl":
            _, logits_per_text_en = self.model.compute_logits(image_features,
                                                              text_en_features,
                                                              fixed_logit=False)

            distillation_loss = self.kl(input=F.log_softmax(logits_per_text_pt / 3.0, dim=1),
                                        target=F.log_softmax(logits_per_text_en / 3.0, dim=1))

            loss = alpha * contrastive_loss + (1 - alpha) * distillation_loss
            self.log("train/KL pt-en", distillation_loss)
        else:
            if self.config.self_distill == "complete":
                _, logits_per_text_en = self.model.compute_logits(image_features,
                                                                  text_en_features,
                                                                  fixed_logit=False)
                contrastive_loss_en = clip_loss(logits_per_text_en)
                self.log("train/infoNCE_en", contrastive_loss_en)
                self.log("train/infoNCE_pt", contrastive_loss)

                contrastive_loss = (contrastive_loss_en + contrastive_loss) / 2

            mse_loss = self.mse(input=text_pt_features, target=text_en_features)
            loss = alpha * contrastive_loss + (1 - alpha) * mse_loss
            self.log("train/mse_loss", mse_loss)

        self.log("train/alpha", alpha)
        self.log("train/infoNCE", contrastive_loss)

        return loss, logits_per_image_pt, logits_per_text_pt


class TeacherStudentSelfDistillCLIPWrapper(SelfDistillCLIPWrapper):
    def __init__(
            self,
            config: DictConfig,
            train_size: int = 0,
            val_labels=None,
            carbon_tracker=None,
            model=None
    ):
        super().__init__(config, train_size=train_size, val_labels=val_labels, model=model,
                         carbon_tracker=carbon_tracker)
        self.teacher = copy.deepcopy(self.model.model.text)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def training_step(self, train_batch, batch_idx):
        self.config.self_distill = self.config.self_distill.replace("teacher-", "")
        super().training_step(train_batch, batch_idx)

    def extract_features(self, train_batch):
        image_input, text_pt_input, text_en_input = train_batch
        image_features, text_pt_features = self.model.encode((image_input, text_pt_input))
        if self.config.self_distill == "complete":
            text_en_features = self.teacher(text_en_input)
        else:
            with torch.no_grad():
                text_en_features = self.teacher(text_en_input)
        return image_features, text_pt_features, text_en_features
