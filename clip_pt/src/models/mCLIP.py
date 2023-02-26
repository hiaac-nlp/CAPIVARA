import clip
import torch
from multilingual_clip import pt_multilingual_clip


class mCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
        self.text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained(self.text_model_name,
                                                                                  cache_dir='/work/gabriel.santos/cache')
        self.vision_model_name = "ViT-B/32"
        self.image_encoder, self.image_preprocessor = clip.load(self.vision_model_name,
                                                              download_root='/work/gabriel.santos/cache')

    def forward(self, batch):
        return self.encode(batch)

    def encode(self, batch):
        image_input, text_input = batch

        image_features = self.encode_visual(image_input)
        text_features = self.encode_text(text_input)

        return image_features, text_features

    def encode_text(self, text_input):
        embeddings = self.text_encoder.transformer(**text_input)[0]
        att = text_input['attention_mask']
        embeddings = (embeddings * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.text_encoder.LinearTransformation(embeddings)

    def encode_visual(self, visual_inputs):
        outputs = self.image_encoder(**visual_inputs)
        return outputs.image_embeds

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
            logit_scale = self.image_encoder.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape: [batch_size, batch_size]
        return logits_per_image, logits_per_text
