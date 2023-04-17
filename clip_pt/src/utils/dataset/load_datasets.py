import json
import math
import os
import random
from typing import Dict

import braceexpand
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPFeatureExtractor

from utils.dataset.clip_pt_br_transform import CliptPTTransform
from utils.dataset.grocery_store_dataset import GroceryStoreDataset


def image_augmentation(image, augment):
    augmentation = [transforms.Resize(250),
                    transforms.RandomResizedCrop(224, scale=(0.9, 1.0))]

    if augment.lower() == "augmix":
        augmentation.append(transforms.AugMix(severity=2))
    elif augment.lower() == "trivialaugmentwide":
        augmentation.append(transforms.TrivialAugmentWide())
    elif augment.lower() == "autoaug":
        augmentation.append(transforms.AutoAugment())
    elif augment.lower() == "clipptbr":
        augmentation.append(CliptPTTransform())
    elif augment.lower() == "clipptbr2":
        augmentation.append(CliptPTTransform(n_transformations=2))

    augmentation = transforms.Compose(augmentation)
    return augmentation(image)


def tokenize(example, vision_processor, text_tokenizer, max_length, augment=None):
    img = example[0]
    if augment:
        img = image_augmentation(img, augment)

    if isinstance(vision_processor, CLIPFeatureExtractor):
        image_input = vision_processor(
            images=img,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
    else:
        image_input = vision_processor(img)

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

    augment = config.get("augment", False)
    decode_format = "torchrgb8" if augment else "torchrgb"
    train_dataset = wds.WebDataset(train, shardshuffle=True) \
        .shuffle(10000) \
        .decode(decode_format) \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, vision_processor, text_tokenizer, max_length, augment)) \
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
    train_size = math.ceil(train_size // config.batch_size)
    val_size = val_size // config.batch_size

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
            text_tokenizer=text_tokenizer)

        img_classif_dataloader = DataLoader(img_classif_dataset, batch_size=config.batch_size,
                                            num_workers=10)

        output["img_classification"] = img_classif_dataloader
        output["img_classif_labels"] = img_classif_dataset.get_labels()

    return output


def tokenize_teacher_student(example, teacher_tokenizer, student_tokenizer, max_length):
    teacher_input = teacher_tokenizer(
        random.choice(example[1]["captions-en"]),  # take a random caption
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    student_input = student_tokenizer(
        random.choice(example[1]["captions-pt"]),  # take a random caption
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    return teacher_input, student_input


def format_batch_teacher_student(batch):
    teacher_input_ids = []
    teacher_attention_mask = []
    student_input_ids = []
    student_attention_mask = []

    for teacher, student in zip(batch[0], batch[1]):
        teacher_input_ids.append(teacher["input_ids"])
        teacher_attention_mask.append(teacher["attention_mask"])
        student_input_ids.append(student["input_ids"])
        student_attention_mask.append(student["attention_mask"])

    teacher_input = {"input_ids": torch.cat(teacher_input_ids, dim=0),
                     "attention_mask": torch.cat(teacher_attention_mask, dim=0)}

    student_input = {"input_ids": torch.cat(student_input_ids, dim=0),
                     "attention_mask": torch.cat(student_attention_mask, dim=0)}

    return teacher_input, student_input


def load_datasets_teacher_student(config, teacher_tokenizer, student_tokenizer) -> Dict:
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
        .map(lambda x: tokenize_teacher_student(x, teacher_tokenizer, student_tokenizer, max_length)) \
        .batched(config.batch_size) \
        .map(format_batch_teacher_student)

    val_dataset = wds.WebDataset(val, shardshuffle=True) \
        .shuffle(10000) \
        .decode("torchrgb") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize_teacher_student(x, teacher_tokenizer, student_tokenizer, max_length)) \
        .batched(config.batch_size) \
        .map(format_batch_teacher_student)

    # dataset size correctly according to the number of batches
    train_size = math.ceil(train_size // config.batch_size)
    val_size = val_size // config.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=10)

    output = {"train_dataloader": train_dataloader,
              "train_size": train_size,
              "val_dataloader": val_dataloader,
              "val_size": val_size}

    return output

