import os
from typing import Union

import torch
from torch import nn
from huggingface_hub import hf_hub_download

def download_pretrained_from_hf(
    model_id: str = "hiaac-nlp/CAPIVARA",
    filename: str = "2kmj76gi.ckpt",
    revision = None,
    cache_dir: Union[str, None] = None
) -> str:
    """Download the pretrained weights from the Hugging Face Hub"""
    cached_file = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
    return cached_file


def load_pretrained_weights_from_capivara(model: nn.Module = None, path_to_pretrained_weights: str = None):
    """Adapts the keys of the pretrained weights to the model's state_dict keys.

    Removes the prefix 'model.model.' or 'model.' from the keys because the model is loaded
    from a checkpoint which is based on a pytorch-lightning class.
    """
    if os.path.exists(path_to_pretrained_weights):
        model_path = path_to_pretrained_weights
    else:
        try:
            model_path = download_pretrained_from_hf(model_id=path_to_pretrained_weights)
        except Exception as e:
            print(f"Error downloading the model from Hugging Face Hub: {e}")
            return

    # Removes the prefix 'model.model.' from the keys because the model is loaded
    # from a checkpoint which is based on a pytorch-lightning class
    original_state_dict = torch.load(model_path).get("state_dict", {})

    # Get the model's state_dict keys to determine if they have a "model." prefix
    model_keys = model.state_dict().keys()

    # Prepare a new state dict with adjusted keys
    adjusted_state_dict = {}
    for key, value in original_state_dict.items():
        adjusted_key = key
        # Remove "model.model." prefix if it's redundant
        if key.startswith("model.model."):
            adjusted_key = key.replace("model.model.", "", 1)
        # Adjust keys by adding or removing "model." prefix based on the model's keys
        if adjusted_key not in model_keys and 'model.' + adjusted_key in model_keys:
            adjusted_key = 'model.' + adjusted_key
        elif adjusted_key.startswith("model.") and adjusted_key.replace("model.", "", 1) in model_keys:
            adjusted_key = adjusted_key.replace("model.", "", 1)

        adjusted_state_dict[adjusted_key] = value

    # Removes the position_ids key if it is present in the state_dict and not in the model
    # The openclip repo do the same when loading weights:
    # https://github.com/mlfoundations/open_clip/blob/9eaf2424e74a4e34f5041e640e5e69bac5eb41aa/src/open_clip/factory.py#L142
    for specific_key in [
        "text.transformer.embeddings.position_ids",
        "model.text.transformer.embeddings.position_ids"
    ]:
        if specific_key in adjusted_state_dict and specific_key not in model_keys:
            del adjusted_state_dict[specific_key]

    # Load the adjusted state dict into the model
    model.load_state_dict(adjusted_state_dict)

    return model