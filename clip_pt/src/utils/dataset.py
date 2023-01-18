import os
import json
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name: str, dataset_path: str, image_base_dir: str, 
                 split="train", **kwargs):
        """

        :param dataset_name: the name of the dataset (pracegover, mscoco, flickr30k)
        :param dataset_path: the path to the dataset
        :param image_base_dir: the path to the directory with the images
        :param split: possible options - train, val or test
        """
        self.dataset_name = dataset_name
        self.image_base_dir = image_base_dir

        if dataset_name.lower() == "pracegover":
            self.dataset = self.__read_pracegover(dataset_path, split)
        elif dataset_name.lower() == "mscoco":
            self.dataset = self.__read_mscoco(dataset_path, split)
        elif dataset_name.lower() == "flickr30k":
            translation_path = kwargs["translation_path"]
            self.dataset = self.__read_flickr30k(dataset_path, translation_path, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset_name.lower() == "pracegover":
            example = self.__get_example_pracegover(index)
        elif self.dataset_name.lower() == "mscoco":
            example = self.__get_example_mscoco(index)
        elif self.dataset_name.lower() == "flickr30k":
            example = self.__get_example_flickr30k(index)
            
        return example

    def __read_mscoco(self, path, split):
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

    def __get_example_mscoco(self, index):
        index2key = list(self.dataset)[index]

        img_path = os.path.join(self.image_base_dir, f"{str(index2key).zfill(12)}.jpg")
        img = Image.open(img_path)
        to_tensor = transforms.ToTensor()

        return to_tensor(img), {"image": f"{str(index2key).zfill(12)}.jpg",
                                "captions-PT": self.dataset[index2key]["captions"],
                                "captions-EN": []}

    def __read_pracegover(self, path, split):
        """

        :param path: path to pracegover 400k
        :param split: train/val/test
        :return:
        """
        with open(path) as file:
            dataset = json.load(file)            
        return dataset[split if split.lower() != "val" else "validation"]

    def __get_example_pracegover(self, index):
        example = self.dataset[index]
        img_path = os.path.join(self.image_base_dir, example["filename"])
        img = Image.open(img_path)
        to_tensor = transforms.ToTensor()

        return to_tensor(img), {"image": example["filename"],
                                "captions-PT": [example["caption"]],
                                "captions-EN": []}

    def __read_flickr30k(self, path, translation_path, split):
        """

        :param path: path to karpathy dataset splits
        :param translation_path: path to file with translations
        :param split: train/val/test
        :return:
        """
        with open(path) as file:
            karpathy_dataset = json.load(file)

        df_trad = pd.read_csv(translation_path, sep="\t", names=["image", "caption"])
        df_trad.loc[:, "image"] = df_trad["image"].apply(lambda image: image[:-2])

        dataset = defaultdict(lambda: {"image": None,
                                       "captions-PT": [],
                                       "captions-EN": []})

        for example in karpathy_dataset["images"]:
            if example["split"] == split:
                img = example["filename"]
                en_captions = [caption["raw"] for caption in example["sentences"]]
                dataset[img]["image"] = img
                dataset[img]["captions-EN"] = en_captions

        for row in df_trad.iterrows():
            img = row[1]["image"]
            pt_caption = row[1]["caption"]
            # take only the images from karpathy set
            if img in dataset:
                dataset[img]["captions-PT"].append(pt_caption)

        return list(dataset.values())

    def __get_example_flickr30k(self, index):
        example = self.dataset[index]
        img_path = os.path.join(self.image_base_dir, example["image"])
        img = Image.open(img_path)
        to_tensor = transforms.ToTensor()

        return to_tensor(img), example
