import open_clip
import torch


class OpenCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _, self.image_preprocessor = open_clip.create_model_and_transforms(
            "xlm-roberta-base-ViT-B-32",
            pretrained="laion5b_s13b_b90k"
        )
        self.text_tokenizer = open_clip.get_tokenizer("xlm-roberta-base-ViT-B-32")
        self.model.text.set_grad_checkpointing(True)

    def forward(self, batch):
        return self.encode(batch)

    def encode(self, batch):
        image_input, text_input = batch

        image_features = self.encode_visual(image_input)
        text_features = self.encode_text(text_input)

        return image_features, text_features

    def encode_visual(self, visual_inputs):
        return self.model.encode_image(visual_inputs)

    def encode_text(self, text_inputs):
        return self.model.encode_text(text_inputs)

    def compute_logits(
            self,
            image_features,
            text_features,
            fixed_logit: bool
    ):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        if fixed_logit:
            logit_scale = 20
        else:
            logit_scale = self.model.logit_scale.exp().float()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape: [batch_size, batch_size]
        return logits_per_image, logits_per_text

    def freeze(self):
        for param in self.model.visual.parameters():
            param.requires_grad = False