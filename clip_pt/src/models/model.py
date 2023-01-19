import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import CLIPVisionModel


class CLIPTBR(nn.Module):
    def __init__(
            self,
            projection_dim: int = 512,
            vision_encoder_version: str = "openai/clip-vit-base-patch32",
            text_encoder_version: str = "neuralmind/bert-base-portuguese-cased"
    ):
        super().__init__()

        self.projection_dim = projection_dim
        self.model_clip = CLIPVisionModel.from_pretrained(vision_encoder_version)
        self.model_bertimbau = AutoModel.from_pretrained(text_encoder_version)

        self.model_clip.gradient_checkpointing_enable()
        self.model_bertimbau.gradient_checkpointing_enable()

        self.visual_projection = nn.Linear(
            self.model_clip.vision_model.post_layernorm.normalized_shape[0],
            self.projection_dim,
            bias=False
        )

        self.text_projection = nn.Linear(
            self.model_bertimbau.pooler.dense.in_features,
            self.projection_dim,
            bias=False
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_visual(self, visual_inputs):
        outputs = self.model_clip(visual_inputs)
        hidden_states = outputs.pooler_output

        return self.visual_projection(hidden_states)

    def encode_text(self, text_inputs):
        outputs = self.model_bertimbau(**text_inputs)

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
            fixed_logit
    ):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        if (fixed_logit) >= 0:
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = 20
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape: [batch_size, batch_size]
        return logits_per_image, logits_per_text

    def model_requires_grad(self, status=True):
        for param in self.model_clip.parameters():
            param.requires_grad = status
        for param in self.model_bertimbau.parameters():
            param.requires_grad = status