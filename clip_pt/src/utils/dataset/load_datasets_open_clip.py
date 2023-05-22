import json
import os
import random
from typing import Dict

import braceexpand
import webdataset as wds
from torch.utils.data import DataLoader

from utils.dataset.grocery_store_dataset import GroceryStoreDataset


def tokenize(example, vision_processor, text_tokenizer, self_distill=False):
    img = example[0]
    image_input = vision_processor(img)

    if self_distill:
        n_captions = len(example[1]["captions-en"])
        caption_index = random.randint(0, n_captions - 1)

        # Google translation (w/ even indices)
        sample_pt = example[1]["captions-pt"][2 * caption_index + 1]
        sample_en = example[1]["captions-en"][caption_index]

        text_pt_input = text_tokenizer(sample_pt)
        text_en_input = text_tokenizer(sample_en)
        return image_input, text_pt_input, text_en_input
    else:
        # take a random caption
        text_input = text_tokenizer(random.choice(example[1]["captions-pt"]))
        return image_input, text_input


def format_batch(batch, self_distill=False):
    image_input = batch[0]
    if self_distill:
        text_pt_input = batch[1].reshape((-1, 77))
        text_en_input = batch[2].reshape((-1, 77))

        return image_input, text_pt_input, text_en_input
    else:
        text_input = batch[1].reshape((-1, 77))
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

    train_dataset = wds.WebDataset(train, shardshuffle=True) \
        .shuffle(10000) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, vision_processor, text_tokenizer, self_distill=config.get("self_distill", False))) \
        .batched(config.batch_size) \
        .map(lambda x: format_batch(x, self_distill=config.get("self_distill", False)))

    val_dataset = wds.WebDataset(val, shardshuffle=True) \
        .shuffle(10000) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, vision_processor, text_tokenizer)) \
        .batched(config.batch_size) \
        .map(lambda x: format_batch(x))

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=10)

    output = {"train_dataloader": train_dataloader,
              "train_size": train_size,
              "val_dataloader": val_dataloader,
              "val_size": val_size}

    if config.datasets.get("img_classification", False):
        img_classif_dataset = GroceryStoreDataset(
            dataset_path=config.datasets.img_classification.path,
            annotation_path=config.datasets.img_classification.annotation_path,
            vision_processor=vision_processor,
            text_tokenizer=text_tokenizer,
            max_length=77,
            open_clip=True)

        img_classif_dataloader = DataLoader(img_classif_dataset, batch_size=config.batch_size,
                                            num_workers=10)

        output["img_classification"] = img_classif_dataloader
        output["img_classif_labels"] = img_classif_dataset.get_labels()

    return output


