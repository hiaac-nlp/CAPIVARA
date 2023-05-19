import torch
from omegaconf import DictConfig
from torch import nn

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

        image_input, text_pt_input, text_en_input = train_batch

        image_features, text_pt_features = self.model.encode((image_input, text_pt_input))
        logits_per_image, logits_per_text = self.model.compute_logits(image_features,
                                                                      text_pt_features,
                                                                      fixed_logit=False)
        with torch.no_grad():
            text_en_features = self.model.encode_text(text_en_input)

        contrastive_loss = clip_loss(logits_per_text)
        mse_loss = self.mse(input=text_pt_features, target=text_en_features)

        if self.alpha_constant:
            alpha = self.alpha
        else:
            alpha = self.compute_alpha()

        loss = alpha * contrastive_loss + (1 - alpha) * mse_loss

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
        self.log("train/infoNCE", contrastive_loss)
        self.log("train/mse_loss", mse_loss)
        self.log("train/batch_image_accuracy", batch_image_accuracy)
        self.log("train/batch_text_accuracy", batch_text_accuracy)

    def compute_alpha(self):
        if self.trainer.global_step <= self.warmup_start:
            return self.min_alpha

        if self.warmup_start < self.trainer.global_step < self.warmup_end:
            return self.min_alpha + (self.trainer.global_step - self.warmup_start) * \
                (self.max_alpha - self.min_alpha) / (self.warmup_end - self.warmup_start)

        return self.max_alpha
