import torch
from multilingual_clip import pt_multilingual_clip
from transformers import CLIPVisionModelWithProjection


class mCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
        self.text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained(self.text_model_name,
                                                             cache_dir='/work/gabriel.santos/cache')
        self.vision_model_name = "openai/clip-vit-base-patch32"
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.vision_model_name,
                                                             cache_dir='/work/gabriel.santos/cache')

    def forward(self, batch):
        return self.encode(batch)


    def encode(self, batch):
        image_input, text_input = batch

        image_features = self.encode_visual(image_input)
        text_features = self.encode_text(text_input)
        #print("img shape:", image_features.shape, "text shape:", text_features.shape)

        return image_features, text_features

    def encode_text(self, text_input):
        embeddings = self.text_encoder.transformer(**text_input)[0]
        att = text_input['attention_mask']
        embeddings = (embeddings * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.text_encoder.LinearTransformation(embeddings)

    def encode_visual(self, visual_inputs):
        outputs = self.image_encoder(**visual_inputs)
        return outputs.image_embeds        

