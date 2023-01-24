import os

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers import CLIPProcessor

from clip_pt.src.utils.dataset.load_datasets import load_datasets

batch_size = 5
home_directory = os.path.expanduser('~')
config = OmegaConf.load('/home/gabriel/CLIP-PtBr/clip_pt/experiment_setup/baseline.yaml')

vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

dataloaders = load_datasets(config=config, vision_processor=vision_processor, text_tokenizer=text_tokenizer)

i = 1
k = 0
for dataloader in dataloaders['train']:
    for data in dataloader:
        images, texts = data
        print(i, '---')
        print(images)
        print(texts)
        i += 1
        # if k == 1:
        break
    k += 1