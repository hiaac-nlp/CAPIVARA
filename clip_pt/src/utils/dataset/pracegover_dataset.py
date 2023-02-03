import json
import os

import torch
from PIL import Image
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PraCegoVerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, image_base_dir: str,
                 vision_processor, text_tokenizer, split='train'):
        self.image_base_dir = image_base_dir
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer

        with open(dataset_path) as file:
            dataset = json.load(file)

        self.dataset = dataset[split if split.lower() != "val" else "validation"]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        img_path = os.path.join(self.image_base_dir, example["filename"])
        img = Image.open(img_path)

        image_input = self.vision_processor(
            images=img,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        text_input = self.text_tokenizer(
            example["caption"],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=30
        )

        return image_input, text_input

