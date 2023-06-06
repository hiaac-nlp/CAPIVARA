import os
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_base_dir: str,
        text_column: str,
        image_path_column: str
    ):
        self.dataset = df
        self.image_base_dir = image_base_dir
        self.text_column = text_column
        self.image_path_column = image_path_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset.iloc[idx][self.text_column], Image.open(os.path.join(self.image_base_dir, self.dataset.iloc[idx][self.image_path_column]))


class CollateFuncTextImg:
    def __init__(
        self,
        vision_processor,
        text_tokenizer,
        text_padding_size
    ):
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.text_padding_size = text_padding_size

    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        text = []
        image = []

        for element in batch:
            text_value, image_value = element
            text.append(text_value)
            image.append(image_value)

        image_input = self.vision_processor(
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        text_input = self.text_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.text_padding_size
        )

        return image_input, text_input

