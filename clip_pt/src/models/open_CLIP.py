import open_clip
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoConfig
from transformers import LoRAConfig, UniPELTConfig
from transformers.adapters import XLMRobertaAdapterModel



class OpenCLIP(torch.nn.Module):
    def __init__(
            self,
            adapter: str = None
    ):
        super().__init__()
        self.model, _, self.image_preprocessor = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32',
                                                                                       pretrained='laion5b_s13b_b90k')
        self.text_tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')
        self.model.text.set_grad_checkpointing(True)
        self.xlm_config = AutoConfig.from_pretrained("xlm-roberta-base")
        self.pooler_xlm = MeanPooler()
        
        self.text_encoder = self.add_adapter(self.model, adapter_name=adapter)
    
    def add_adapter(self, model, adapter_name):
       open_clip_model_state = model.text.state_dict()
       config = None
       
       if adapter_name.lower() == "lora":
           config = LoRAConfig()
       elif adapter_name.lower() == "unipelt":
           config = UniPELTConfig()
       if config is not None:
            xlm_roberta_adapter = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
            xlm_roberta_adapter.add_adapter(adapter_name, config=config)
            xlm_roberta_adapter.train_adapter(adapter_name)
            xlm_roberta_adapter.set_active_adapters(adapter_name)
            xlm_roberta_adapter.add_module("proj", nn.Sequential(
                nn.Linear(768, 640, bias=False),
                nn.GELU(),
                nn.Linear(640, 512, bias=False),
            ))
            
            #freeze the Linear layer
            for name, param in xlm_roberta_adapter.named_parameters():
                if name in ['proj.0.weight','proj.2.weight']:
                    param.requires_grad = False

            for key in list(open_clip_model_state.keys()):
                # There is a difference in key names between the original mCLIP model and the XLM-Roberta model
                open_clip_model_state[key.replace('transformer', 'roberta')] = open_clip_model_state.pop(key)
            xlm_roberta_adapter.load_state_dict(open_clip_model_state, strict=False)
            del model.text

            return xlm_roberta_adapter 
       return model.text

    def forward(self, batch):
        return self.encode(batch)

    def encode(self, batch):
        image_input, text_input = batch

        image_features = self.model.encode_image(image_input)
        #text_features = self.model.encode_text(text_input)
        text_features = self.encode_text_adapters(text_input,self.xlm_config)
        return image_features, text_features

    def encode_visual(self, visual_inputs):
        return self.model.encode_image(visual_inputs).float()

    def encode_text(self, text_inputs):
        return self.text_encoder(text_inputs)
    
    def encode_text_adapters(self, text_input, config, normalize=True):
        attn_mask = (text_input != config.pad_token_id).long()
        text_latent = self.text_encoder(text_input)
        text_latent = self.pooler_xlm(text_latent, attn_mask)
        text_latent = self.text_encoder.proj(text_latent)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent

        return text_latent

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

    
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x, attention_mask):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)
    
def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
