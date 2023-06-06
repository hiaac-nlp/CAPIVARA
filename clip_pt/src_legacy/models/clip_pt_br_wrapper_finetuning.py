import torch
from omegaconf import DictConfig

from models.clip_pt_br_wrapper_image_classification import CLIPPTBRWrapperImageClassification
from models.model import CLIPTBRFinetuning, CLIPTBRZeroshot


class CLIPPTBRWrapperFinetuning(CLIPPTBRWrapperImageClassification):
    def __init__(
            self,
            config: DictConfig,
            text_encoder_checkpoint=None,
            train_size: int = 0,
            val_labels=None,
            carbon_tracker=None
    ):
        super().__init__(config, train_size=train_size, val_labels=val_labels,
                         carbon_tracker=carbon_tracker)
        if text_encoder_checkpoint is None:
            text_encoder_checkpoint = torch.load(config.model.text_encoder)
            self.model = CLIPTBRFinetuning(vision_encoder_version=config.model.image_encoder,
                                           text_encoder_checkpoint=text_encoder_checkpoint,
                                           inference=True)
        else:
            self.model = CLIPTBRFinetuning(vision_encoder_version=config.model.image_encoder,
                                           text_encoder_checkpoint=text_encoder_checkpoint,
                                           inference=False)

        self.unfreeze = config.model["warmup_steps"] > 0
        if self.unfreeze:
            print("Freezing model!!")
            self.model.freeze()


class CLIPPTBRZeroshotWrapper(CLIPPTBRWrapperImageClassification):
    def __init__(
            self,
            config: DictConfig,
            checkpoint_path=None,
            train_size: int = 0,
            val_labels=None,
            carbon_tracker=None
    ):
        super().__init__(config, train_size=train_size, val_labels=val_labels,
                         carbon_tracker=carbon_tracker)

        self.model = CLIPTBRZeroshot(checkpoint_path=checkpoint_path)

        self.unfreeze = config.model["warmup_steps"] > 0
        if self.unfreeze:
            print("Freezing model!!")
            self.model.freeze()
