from typing import Union

from huggingface_hub import hf_hub_download

def download_pretrained_from_hf(
    model_id: str = "hiaac-nlp/CAPIVARA",
    filename: str = "2kmj76gi.ckpt",
    revision=None,
    cache_dir: Union[str, None] = None
) -> str:
    cached_file = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
    return cached_file