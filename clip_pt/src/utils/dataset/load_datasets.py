from typing import Tuple, Any

import braceexpand
import webdataset as wds
from torch.utils.data import DataLoader

from utils.dataset.data_collator import ImageMultipleTextDataCollator


def load_datasets(config, vision_processor, text_tokenizer) -> \
        Tuple[DataLoader[Any], DataLoader[Any]]:
    train = []
    for dataset in config.datasets.train:
        train += list(braceexpand.braceexpand(dataset['path']))

    val = []
    for dataset in config.datasets.validation:
        val += list(braceexpand.braceexpand(dataset['path']))

    train_dataset = wds.WebDataset(train, shardshuffle=True).shuffle(10000)\
                                                            .decode("torchrgb") \
                                                            .to_tuple("jpg;png", "json") \
                                                            .batched(config.batch_size)
    val_dataset = wds.WebDataset(val, shardshuffle=True).shuffle(10000) \
                                                        .decode("torchrgb") \
                                                        .to_tuple("jpg;png", "json") \
                                                        .batched(config.batch_size)

    collate_fn = ImageMultipleTextDataCollator(vision_processor=vision_processor,
                                               text_tokenizer=text_tokenizer,
                                               text_padding_size=config.text_padding_size)

    train_dataloader = DataLoader(train_dataset, batch_size=None, collate_fn=collate_fn,
                                  num_workers=10)

    val_dataloader = DataLoader(val_dataset, batch_size=None, collate_fn=collate_fn, num_workers=10)

    return train_dataloader, val_dataloader
