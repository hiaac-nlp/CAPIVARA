import os
import json

import torch
from PIL import Image
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name: str, dataset_path: str, image_base_dir: str, split='train'):
        self.dataset_name = dataset_name
        self.image_base_dir = image_base_dir

        if dataset_name.lower() == 'pracegover':
            self.dataset = self.read_pracegover(dataset_path, split)

        if dataset_name.lower() == 'mscoco':
            self.dataset = self._read_mscoco(dataset_path, split)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)


    def __getitem__(self, index):
        if self.dataset_name.lower() == 'pracegover':
            example = self.get_example_pracegover(index)

        if self.dataset_name.lower() == 'mscoco':
            example = self._get_example_mscoco(index)

        return example


    def _read_mscoco(self, path):
        id2image = {}
        with open(path) as j:
            data = json.load(j)

        for annotation in data["annotations"]:
            if annotation["image_id"] in id2image:
                id2image[annotation["image_id"]]["captions"].append(annotation["caption"])
            else:
                id2image[annotation["image_id"]] = {}
                id2image[annotation["image_id"]]["captions"] = []
                id2image[annotation["image_id"]]["captions"].append(annotation["caption"])

        return id2image


    def _get_example_mscoco(self, index):
        index2key = list(self.dataset)[index]
        
        img_path = os.path.join(self.image_base_dir, f"{str(index2key).zfill(12)}.jpg")
        img = Image.open(img_path)
        to_tensor = transforms.ToTensor()

        return to_tensor(img), {"image": f"{str(index2key).zfill(12)}.jpg",
                                "captions-PT": self.dataset[index2key]["captions"],
                                "captions-EN": []}


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
