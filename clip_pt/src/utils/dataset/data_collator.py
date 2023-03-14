import random
from typing import List, Tuple

import torch


class ImageMultipleTextDataCollator:
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
        texts = []
        images = []

        for image, annotations in zip(batch[0], batch[1]):
            # select a random caption
            texts.append(random.choice(annotations["captions-pt"]))
            images.append(image)

        image_input = self.vision_processor(
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        text_input = self.text_tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.text_padding_size
        )

        return image_input, text_input
