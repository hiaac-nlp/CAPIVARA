import json
import os

import torch
from PIL import Image
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name: str, dataset_path: str, image_base_dir: str, split='train'):
        self.dataset_name = dataset_name
        self.image_base_dir = image_base_dir

        if dataset_name.lower() == 'pracegover':
            self.dataset = self.read_pracegover(dataset_path, split)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset_name.lower() == 'pracegover':
            example = self.get_example_pracegover(index)

        return example

    def read_pracegover(self, path, split):
        with open(path) as file:
            dataset = json.load(file)
        return dataset[split]

    def get_example_pracegover(self, index):
        example = self.dataset[index]
        img_path = os.path.join(self.image_base_dir, example['filename'])
        img = Image.open(img_path)
        to_tensor = transforms.ToTensor()

        return to_tensor(img), {"image": example['filename'],
                                "captions-PT": [example['caption']],
                                "captions-EN": []}
