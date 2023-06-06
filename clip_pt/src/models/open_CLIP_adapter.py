import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, XLMRobertaAdapterModel, LoRAConfig, UniPELTConfig

from models.open_CLIP import OpenCLIP


class OpenCLIPAdapter(OpenCLIP):
    def __init__(self,
                 adapter: str = None,
                 devices=None,
                 inference=False):
        super().__init__()
        self.xlm_config = AutoConfig.from_pretrained("xlm-roberta-base")
        self.pooler_xlm = MeanPooler()
        self.devices = devices
        self.adapter = adapter

        if not inference:
            self.text_encoder = self.add_adapter(self.model, adapter_name=adapter)

    def add_adapter(self, model, adapter_name):
        # get the state from OpenCLIP
        open_clip_model_state = model.text.state_dict()
        del model.text

        if adapter_name.lower() == "lora":
            config = LoRAConfig()
        elif adapter_name.lower() == "unipelt":
            config = UniPELTConfig()
        else:
            raise NotImplementedError

        # create text model in AdapterTransformer framework
        xlm_roberta = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
        xlm_roberta.add_adapter(adapter_name, config=config)
        xlm_roberta.train_adapter(adapter_name)
        return self.transfer_weights(adapter_name, open_clip_model_state, xlm_roberta)

    def load_adapters(self, pretrained_adapter, adapter_name="LoRA"):
        open_clip_model_state = self.model.text.state_dict()
        del self.model.text
        xlm_roberta = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
        model_path = f"./CLIP-PT/adapter_checkpoints/{pretrained_adapter}/{adapter_name}"
        xlm_roberta.load_adapter(model_path)
        self.text_encoder = self.transfer_weights(adapter_name, open_clip_model_state, xlm_roberta)

    def encode_visual(self, visual_inputs):
        return self.model.encode_image(visual_inputs).float()

    def encode_text(self, text_inputs):
        attn_mask = (text_inputs != self.xlm_config.pad_token_id).long()
        text_latent = self.text_encoder(text_inputs, attn_mask)
        text_latent = self.pooler_xlm(text_latent, attn_mask)
        text_latent = self.text_encoder.proj(text_latent)
        return F.normalize(text_latent, dim=-1)

    def transfer_weights(self, adapter_name, open_clip_model_state, xlm_roberta):
        """
        Transfer weights from model in OpenCLIP to AdapterTransformers model

        :param adapter_name:
        :param open_clip_model_state:
        :param xlm_roberta:
        :return:
        """
        xlm_roberta.set_active_adapters(adapter_name)
        xlm_roberta.add_module("proj", nn.Sequential(
            nn.Linear(768, 640, bias=False),
            nn.GELU(),
            nn.Linear(640, 512, bias=False),
        ))

        # freeze the last Linear layer
        for name, param in xlm_roberta.named_parameters():
            if name in ['proj.0.weight', 'proj.2.weight']:
                param.requires_grad = False

        # map the layer names from OpenCLIP to AdapterTransformers model
        for key in list(open_clip_model_state.keys()):
            # There is a difference in key names between the original OpenCLIP model and the
            # XLM-Roberta model
            open_clip_model_state[
                key.replace('transformer', 'roberta')] = open_clip_model_state.pop(key)

        # transfer the weights from OpenCLIP model to AdapterTransformers one
        m, u = xlm_roberta.load_state_dict(open_clip_model_state, strict=False)
        print("Missing keys:")
        print(m)
        print("Unexpected keys:")
        print(u)

        return xlm_roberta.to(self.device)



class MeanPooler(torch.nn.Module):
    """Mean pooling"""

    def forward(self, x, attention_mask):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
