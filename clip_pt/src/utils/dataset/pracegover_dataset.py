import json
import os

import torch
from PIL import Image
from torchvision import transforms


class PraCegoVerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, image_base_dir: str, split='train'):
        self.image_base_dir = image_base_dir

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
        img = self.transform(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img, {"image": example["filename"],
                                     "captions-pt": [example["caption"]],
                                     "captions-en": []}