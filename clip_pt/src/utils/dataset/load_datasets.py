from typing import List, Dict

import webdataset as wds
from torch.utils.data import DataLoader

from utils.dataset.custom_dataset import CustomDataset
from utils.dataset.data_collator import ImageMultipleTextDataCollator


def __load_dataloader(config, info, text_tokenizer, vision_processor, split):
    custom_datasets = ["pracegover", "mscoco", "flickr30k"]

    if info.name.lower() in custom_datasets:
        name = info.pop('name')
        path = info.pop('path')
        image_base_dir = info.pop('image_base_dir')
        dataset = CustomDataset(**info, dataset_name=name, dataset_path=path,
                                image_base_dir=image_base_dir, split=split)
    else:
        dataset = wds.WebDataset(info.path).decode("torchrgb") \
            .to_tuple("jpg;png", "json") \
            .batched(1)

    collate_fn = ImageMultipleTextDataCollator(vision_processor=vision_processor,
                                               text_tokenizer=text_tokenizer,
                                               text_padding_size=config.text_padding_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                            num_workers=4)
    return dataloader


def load_datasets(config, vision_processor, text_tokenizer) -> Dict[str, List[DataLoader]]:
    dataloaders = {"train": [], "val": []}
    for info in config.datasets.train:
        dataloader = __load_dataloader(config, info, text_tokenizer, vision_processor,
                                       split="train")
        dataloaders["train"].append(dataloader)

    for info in config.datasets.validation:
        dataloader = __load_dataloader(config, info, text_tokenizer, vision_processor,
                                       split="val")
        dataloaders["val"].append(dataloader)

    return dataloaders
