import json
import os
import random
from typing import Tuple, Any, Dict

import braceexpand
import torch
import webdataset as wds
from torch.utils.data import DataLoader


def tokenize(example, vision_processor, text_tokenizer, max_length):
    image_input = vision_processor(
        images=example[0],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    text_input = text_tokenizer(
        random.choice(example[1]["captions-pt"]),  # take a random caption
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    return image_input, text_input


def format_batch(batch):
    pixel_values = []
    input_ids = []
    attention_mask = []

    for img, txt in zip(batch[0], batch[1]):
        pixel_values.append(img["pixel_values"])
        input_ids.append(txt["input_ids"])
        attention_mask.append(txt["attention_mask"])

    image_input = {"pixel_values": torch.cat(pixel_values, dim=0)}
    text_input = {"input_ids": torch.cat(input_ids, dim=0),
                  "attention_mask": torch.cat(attention_mask, dim=0)}

    return image_input, text_input


def load_datasets(config, vision_processor, text_tokenizer) -> Dict:
    """
        previously computed dataset sizes. This is necessary because __len__ method in WebDataset
        returns an inaccurate value, so we have to set it manually.
        Reference: https://webdataset.github.io/webdataset/sharding/
    """
    current_path = os.path.dirname(__file__)
    with open(os.path.join(current_path, "datasets_size.json")) as file:
        datasets_sizes = json.load(file)

    print(">>>>> Train datasets:", [dataset['path'] for dataset in config.datasets.train])
    print(">>>>> Validation datasets:", [dataset['path'] for dataset in config.datasets.validation])

    train = []
    train_size = 0
    for dataset in config.datasets.train:
        train_size += datasets_sizes["train"][dataset['name']]
        train += list(braceexpand.braceexpand(dataset['path']))

    val = []
    val_size = 0
    for dataset in config.datasets.validation:
        val_size += datasets_sizes["validation"][dataset['name']]
        val += list(braceexpand.braceexpand(dataset['path']))

    max_length = config.model.text_padding_size

    train_dataset = wds.WebDataset(train, shardshuffle=True) \
        .shuffle(10000) \
        .decode("torchrgb") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, vision_processor, text_tokenizer, max_length)) \
        .batched(config.batch_size) \
        .map(format_batch)

    val_dataset = wds.WebDataset(val, shardshuffle=True) \
        .shuffle(10000) \
        .decode("torchrgb") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, vision_processor, text_tokenizer, max_length)) \
        .batched(config.batch_size) \
        .map(format_batch)

    # dataset size correctly according to the number of batches
    train_size = train_size // config.batch_size
    val_size = val_size // config.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=10)

    return {"train_dataloader": train_dataloader,
            "train_size": train_size,
            "val_dataloader": val_dataloader,
            "val_size": val_size}
