import os

import torch
from torch.utils.data import Subset, ConcatDataset
from transformers import AutoTokenizer
from transformers import CLIPProcessor
import webdataset as wds

from utils.dataset.custom_dataset import CustomDataset
from utils.dataset.data_collator import ImageMultipleTextDataCollator

home_directory = os.path.expanduser('~')

dataset_path = os.path.join(home_directory, "dataset_flickr30k.json")
image_base_dir = os.path.join(home_directory, "flickr30k/flickr30k_images")
translation_path = os.path.join(home_directory, "dataset_clip_pt/flickr30k/results_20130124_ptbr.token")
dataset_flickr30k = Subset(CustomDataset(dataset_name='flickr30k', dataset_path=dataset_path,
                                    image_base_dir=image_base_dir, translation_path=translation_path),
                           range(5))


dataset_path = os.path.join(home_directory, "dataset_clip_pt/cc3m-w/{00000..00001}.tar")
dataset_cc3m = wds.WebDataset(dataset_path).decode("torchrgb").to_tuple("jpg;png", "json").batched(1)


datasets = [dataset_flickr30k, dataset_cc3m]

batch_size = 5
vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
dataloaders = [torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         collate_fn=ImageMultipleTextDataCollator(vision_processor=vision_processor,
                                                                                   text_tokenizer=text_tokenizer,
                                                                                   text_padding_size=50),
                                         num_workers=4) for dataset in datasets]
i = 1
k = 0
for dataloader in dataloaders:
    for data in dataloader:
        images, texts = data
        print(i, '---')
        print(images)
        print(texts)
        i += 1
        if k == 1:
            break
    k += 1