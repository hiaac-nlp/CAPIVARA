import os
import math
import json
import random
from typing import Tuple

import torch
import tqdm
import numpy as np
import pandas as pd
import webdataset as wds
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer 
from utils.data_generator import ImageDataset, CollateFuncTextImg



def seed_init_fn(x: int) -> None:
    seed = x
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def prepare_pracegover(
    train_metadata_path: str,
    val_metadata_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    ### Training Data
    f = open(train_metadata_path)
    dataset = json.load(f)
    f.close()

    train_pracegover = []
    for data in tqdm.tqdm(dataset['images']):
        d = {
            'image_name' : data['filename'],  # some formula for obtaining values
            'filepath' : data['filepath'],
            'comment' : data['sentences'][0]['raw'],
        }
        train_pracegover.append(d)

    df_train = pd.DataFrame(train_pracegover)

    ### Validation Data
    f = open(val_metadata_path)
    dataset_valid = json.load(f)
    f.close()

    valid_annotations = []
    for data in dataset_valid['annotations']:
        d = {
            'image_id' : data['image_id'],  # some formula for obtaining values
            'comment' : data['caption'].replace('\n'," "),
        }
        valid_annotations.append(d)

    valid_annotations = pd.DataFrame(valid_annotations)

    valid_file_names = []
    for data in dataset_valid['images']:
        d = {
            'image_id' : data['id'],  # some formula for obtaining values
            'image_name' : data['file_name'],
        }
        valid_file_names.append(d)

    valid_file_names = pd.DataFrame(valid_file_names)

    df_pracegover_valid = pd.merge(valid_annotations, valid_file_names, how='inner')

    return df_train[:-20000], df_train[-20000:]


def prepare_image_text_dataloader(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    cfg: dict
):

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
    
    train_image_text_dataset = ImageDataset(
        df=df_train,
        image_base_dir=cfg.image_text_dataset.base_image_path,
        text_column=cfg.image_text_dataset.text_column,
        image_path_column=cfg.image_text_dataset.image_path_column
    )

    val_image_text_dataset = ImageDataset(
        df=df_val,
        image_base_dir=cfg.image_text_dataset.base_image_path,
        text_column=cfg.image_text_dataset.text_column,
        image_path_column=cfg.image_text_dataset.image_path_column
    )

    train_image_text_dataloader = DataLoader(
        dataset=train_image_text_dataset,
        batch_size=cfg.image_text_dataset.batch_size,
        collate_fn=CollateFuncTextImg(clip_processor, tokenizer, cfg.image_text_dataset.padding_size),
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    val_image_text_dataloader = DataLoader(
        dataset=val_image_text_dataset,
        batch_size=cfg.image_text_dataset.batch_size,
        collate_fn=CollateFuncTextImg(clip_processor, tokenizer, cfg.image_text_dataset.padding_size),
        shuffle=False,
        drop_last=True,
        num_workers=8
    )

    return train_image_text_dataloader, val_image_text_dataloader

def compute_n_batches(dataset_size, batch_size, shard_size=10000):
    batches_per_shards = math.ceil(shard_size / batch_size)
    batches_in_last_shard = math.ceil((dataset_size % shard_size) / batch_size)
    n_shards = dataset_size // shard_size
    print("n_shards", n_shards)
    print("batches_per_shards", batches_per_shards)
    print("batches_in_last_shard", batches_in_last_shard)
    return n_shards * batches_per_shards + batches_in_last_shard