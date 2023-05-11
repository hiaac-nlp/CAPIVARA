from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModelWithProjection
from transformers import CLIPVisionModel

from models.teacher_student_model import Student


class CLIPTBR(nn.Module):
    def __init__(
            self,
            projection_dim: int = 512,
            vision_encoder_version: str = "openai/clip-vit-base-patch32",
            text_encoder_version: str = "neuralmind/bert-base-portuguese-cased",
            pretraining: str = "LiT",
            adapter: str = None
    ):
        super().__init__()
        self.pretraining = pretraining
        self.projection_dim = projection_dim
        self.image_encoder = CLIPVisionModel.from_pretrained(vision_encoder_version,
                                                             cache_dir='/hahomes/gabriel.santos')
        self.image_encoder.gradient_checkpointing_enable()

        self.visual_projection = nn.Linear(
            self.image_encoder.vision_model.post_layernorm.normalized_shape[0],
            self.projection_dim,
            bias=False
        )

        if self.__class__.__name__ == 'CLIPTBR':
            self.text_encoder = AutoModel.from_pretrained(text_encoder_version,
                                                          cache_dir='/hahomes/gabriel.santos')
            self.text_encoder.gradient_checkpointing_enable()

            self.text_projection = nn.Linear(
                self.text_encoder.pooler.dense.in_features,
                self.projection_dim,
                bias=False
            )

        # value extracted from original CLIP proposal
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_visual(self, visual_inputs):
        outputs = self.image_encoder(visual_inputs)
        hidden_states = outputs.pooler_output

        return self.visual_projection(hidden_states)

    def encode_text(self, text_inputs):
        outputs = self.text_encoder(**text_inputs)

        return self.text_projection(outputs.pooler_output)

    def forward(self, data):
        image_input, text_input = data
        image_features = self.encode_visual(image_input["pixel_values"])
        text_features = self.encode_text(text_input)

        return image_features, text_features

    def compute_logits(
            self,
            image_features,
            text_features,
            fixed_logit: bool = False
    ):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        if fixed_logit:
            logit_scale = 20
        else:
            logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape: [batch_size, batch_size]
        return logits_per_image, logits_per_text

    def model_requires_grad(self, status=True):
        for param in self.image_encoder.parameters():
            param.requires_grad = status
        for param in self.text_encoder.parameters():
            param.requires_grad = status

    def freeze(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        if self.pretraining.lower() != "lit":
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = True

        if self.pretraining.lower() != "lit":
            for param in self.text_encoder.parameters():
                param.requires_grad = True


class CLIPTBRFinetuning(CLIPTBR):
    def __init__(self,
                 text_encoder_checkpoint,
                 vision_encoder_version: str = "openai/clip-vit-base-patch32",
                 projection_dim: int = 512,
                 inference: bool = False):
        super().__init__(projection_dim=projection_dim,
                         vision_encoder_version=vision_encoder_version)
        self.projection_dim = projection_dim
        self.inference = inference
        self.text_encoder = self.load_student(text_encoder_checkpoint)
        self.text_projection = nn.Linear(
            512,
            self.projection_dim,
            bias=False
        )

    def encode_text(self, text_inputs):
        outputs = self.text_encoder(text_inputs)
        return self.text_projection(outputs)

    def load_student(self, checkpoint):
        new_checkpoint = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            if "student" in k:
                new_key = k[14:]
                new_checkpoint[new_key] = checkpoint["state_dict"][k]

        student_version = checkpoint["hyper_parameters"]["model"]["student"]
        print("Text encoder:", student_version)
        text_encoder = Student(student_version=student_version)
        if not self.inference:
            text_encoder.load_state_dict(new_checkpoint)

        return text_encoder

class CLIPTBRZeroshot(nn.Module):
    def __init__(
            self,
            checkpoint_path
    ):
        super().__init__()
        model_checkpoint = torch.load(checkpoint_path)
        vision_encoder_version = model_checkpoint["hyper_parameters"]["model"]["teacher"]
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(vision_encoder_version,
                                                             cache_dir='/hahomes/gabriel.santos')
        self.text_encoder = self.load_student(model_checkpoint)
        # value extracted from original CLIP proposal
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def load_student(self, checkpoint):
        new_checkpoint = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            if "student" in k:
                new_key = k[14:]
                new_checkpoint[new_key] = checkpoint["state_dict"][k]

        student_version = checkpoint["hyper_parameters"]["model"]["student"]
        print("Text encoder:", student_version)
        text_encoder = Student(student_version=student_version)
        text_encoder.load_state_dict(new_checkpoint)

        return text_encoder

    def encode_visual(self, visual_inputs):
        outputs = self.image_encoder(visual_inputs)
        return outputs.image_embeds

    def encode_text(self, text_inputs):
        return self.text_encoder(text_inputs)

    def forward(self, data):
        image_input, text_input = data
        image_features = self.encode_visual(image_input["pixel_values"])
        text_features = self.encode_text(text_input)

        return image_features, text_features

    def compute_logits(
            self,
            image_features,
            text_features,
            fixed_logit: bool = False
    ):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        if fixed_logit:
            logit_scale = 20
        else:
            logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape: [batch_size, batch_size]
        return logits_per_image, logits_per_text

    def freeze(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False        

