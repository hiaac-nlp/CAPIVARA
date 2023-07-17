import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from models.open_CLIP import OpenCLIP
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
import copy

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
            self.model.text = self.add_adapter(self.model, adapter_name=adapter)
            list_trainable_params(self.model.text)
            count_parameters(self.model)
            count_parameters(self.model.text)

    def add_adapter(self, model, adapter_name):

        if adapter_name.lower() == "lora":
            config = LoraConfig(
                    r=8,
                    lora_alpha=8,
                    target_modules=["query", "value"],
                    lora_dropout=0.0,
                    bias="none",
                )
        else:
            raise NotImplementedError
        
        lora_model_text = get_peft_model(copy.deepcopy(model.text), config)

        return lora_model_text


    def load_adapters(self, pretrained_adapter):
        if pretrained_adapter != 'none':
            model_path = f"/work/diego.moreira/CLIP-PtBr/clip_pt/src/CLIP-PT/adapter_PEFT_checkpoints/{pretrained_adapter}"
            config = PeftConfig.from_pretrained(model_path)
            self.model.text = PeftModel.from_pretrained(self.model.text, model_path, config=config)
        else:
            print('No adapter configuration defined')
            config = LoraConfig(
                    r=8,
                    lora_alpha=8,
                    target_modules=["query", "value"],
                    lora_dropout=0.0,
                    bias="none",
                )
            self.model.text = get_peft_model(copy.deepcopy(self.model.text), config)

    def encode_text(self, text_inputs):
        text_latent = self.model.text(text_inputs)
        return F.normalize(text_latent, dim=-1)

class MeanPooler(torch.nn.Module):
    """Mean pooling"""

    def forward(self, x, attention_mask):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)

def count_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def list_trainable_params(model):
    for name, param in model.named_parameters():
        print(f'{name}: {param.requires_grad}')

def freeze(in_model):
        for param in in_model.parameters():
            param.requires_grad = False


def gpu_status():
    import nvidia_smi
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(7)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(7, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))