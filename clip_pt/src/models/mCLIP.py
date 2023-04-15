import clip
import torch
from multilingual_clip import pt_multilingual_clip
from transformers.adapters import XLMRobertaAdapterModel


class mCLIP(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vision_model_name = "ViT-B/32"
        self.image_encoder, self.image_preprocessor = clip.load(self.vision_model_name,
                                                                download_root='/work/gabriel.santos/cache',
                                                                device=device)
        del self.image_encoder.transformer  # delete original text encoder

        self.text_model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
        self.text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained(self.text_model_name,
                                                                                  cache_dir='/work/gabriel.santos/cache')

    def add_adapter(self, model, adapter_name):
       config = None
       if adapter_name.lower() == "lora":
           config = LoRAConfig()
       elif adapter_name.lower() == "unipelt":
           config = UniPELTConfig()


       if config is not None:
           xlm_roberta_adapter = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-large")
            xlm_roberta_adapter.add_adapter(adapter_name, config=config)
            # Add projection layer to resemble the original mCLIP model
            xlm_roberta_adapter.add_module("LinearTransformation", nn.Linear(in_features=1024, out_features=512, bias=True))

            state_dict = model.state_dict()
            for key in list(state_dict.keys()):
                # There is a difference in key names between the original mCLIP model and the XLM-Roberta model
                state_dict[key.replace('transformer', 'roberta')] = state_dict.pop(key)
            xlm_roberta_adapter.load_state_dict(state_dict, strict=False)

            xlm_roberta_adapter.train_adapter(adapter_name)

            return xlm_roberta_adapter 
       return model

    def forward(self, batch):
        return self.encode(batch)

    def encode(self, batch):
        image_input, text_input = batch

        image_features = self.encode_visual(image_input)
        text_features = self.encode_text(text_input)

        return image_features, text_features

    def encode_text(self, text_input):
        if list(xlm_roberta_adapter.state_dict().keys())[0].split(".")[0] == "transformer":
            embeddings = text_encoder.transformer(**text_input)[0]
        elif list(xlm_roberta_adapter.state_dict().keys())[0].split(".")[0] == "roberta":
            embeddings = text_encoder.roberta(**text_input)[0]

        att = text_input['attention_mask']
        embeddings = (embeddings * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.text_encoder.LinearTransformation(embeddings)

    def encode_visual(self, visual_inputs):
        outputs = self.image_encoder.encode_image(visual_inputs).float()
        return outputs

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
            logit_scale = self.image_encoder.logit_scale.exp().float()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape: [batch_size, batch_size]
        return logits_per_image, logits_per_text

