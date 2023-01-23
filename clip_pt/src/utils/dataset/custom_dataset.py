import os
import json
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _convert_image_to_rgb(image):
    return image.convert("RGB")


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

        # same size used in original CLIP
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

        if dataset_name.lower() == "pracegover":
            self.dataset = self.__read_pracegover(dataset_path, split)
        elif dataset_name.lower() == "mscoco":
            train_translation_path = kwargs["train_translation_path"]
            val_orig_path = kwargs["val_orig_path"]
            val_translation_path = kwargs["val_translation_path"]
            val_img_base_dir = kwargs["val_img_base_dir"]
            self.dataset = self.__read_mscoco(
                path=dataset_path,
                train_translation_path=train_translation_path,
                val_orig_path=val_orig_path,
                val_translation_path=val_translation_path,
                val_img_base_dir=val_img_base_dir,
                split=split
            )
        elif dataset_name.lower() == "flickr30k":
            translation_path = kwargs["translation_path"]
            self.dataset = self.__read_flickr30k(dataset_path, translation_path, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset_name.lower() == "pracegover":
            img, annotations = self.__get_example_pracegover(index)
        elif self.dataset_name.lower() == "mscoco":
            img, annotations = self.__get_example_mscoco(index)
        elif self.dataset_name.lower() == "flickr30k":
            img, annotations = self.__get_example_flickr30k(index)

        return self.transform(img), annotations


    def __read_files_mscoco(self, path: str, translation_path: str):
        """

        :param path: path to the original dataset metadata file
        :param translation_path: path to the translated dataset metadata file
        :return:
        """
        id2image = {}
        with open(path) as orig_file:
            orig_data = json.load(orig_file)

        with open(translation_path) as translation_file:
            translation_data = json.load(translation_file)

        for orig_annotation, translation_annotation in zip(orig_data["annotations"], translation_data["annotations"]):
            assert orig_annotation["image_id"] == translation_annotation["image_id"], "Not synced"
            if orig_annotation["image_id"] in id2image:
                id2image[orig_annotation["image_id"]]["en_captions"].append(orig_annotation["caption"])
                id2image[orig_annotation["image_id"]]["pt_captions"].append(translation_annotation["caption"])
            else:
                id2image[orig_annotation["image_id"]] = {}
                id2image[orig_annotation["image_id"]]["en_captions"] = []
                id2image[orig_annotation["image_id"]]["pt_captions"] = []
                id2image[orig_annotation["image_id"]]["en_captions"].append(orig_annotation["caption"])
                id2image[orig_annotation["image_id"]]["pt_captions"].append(translation_annotation["caption"])

        return id2image

    def __read_mscoco(
        self,
        path: str,
        train_translation_path: str,
        val_orig_path: str,
        val_translation_path: str,
        val_img_base_dir: str,
        split: str
    ):
        """

        :param path: path to the original dataset metadata file (train)
        :param train_translation_path: path to the translated dataset metadata file (train)
        :param val_orig_path: path to the original dataset metadata file (val)
        :param val_translation_path: path to the translated dataset metadata file (val)
        :param val_img_base_dir: path to the directory with the images of the val split
        :param split: train/val/test
        :return:
        """
        if split == "train":
            return self.__read_files_mscoco(path=path, translation_path=train_translation_path)

        elif split == "val":
            self.image_base_dir = val_img_base_dir
            return self.__read_files_mscoco(path=val_orig_path, translation_path=val_translation_path)

        else:
            raise NotImplementedError(
                f"split '{split}' not implemented"
            )

    def __get_example_mscoco(self, index):
        index2key = list(self.dataset)[index]

        img_path = os.path.join(self.image_base_dir, f"{str(index2key).zfill(12)}.jpg")
        img = Image.open(img_path)

        return img, {"image": f"{str(index2key).zfill(12)}.jpg",
                                "captions-pt": self.dataset[index2key]["pt_captions"],
                                "captions-en": self.dataset[index2key]["en_captions"]}

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

        return img, {"image": example["filename"],
                                "captions-pt": [example["caption"]],
                                "captions-en": []}

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
                                       "captions-pt": [],
                                       "captions-en": []})

        for example in karpathy_dataset["images"]:
            if example["split"] == split:
                img = example["filename"]
                en_captions = [caption["raw"] for caption in example["sentences"]]
                dataset[img]["image"] = img
                dataset[img]["captions-en"] = en_captions

        for row in df_trad.iterrows():
            img = row[1]["image"]
            pt_caption = row[1]["caption"]
            # take only the images from karpathy set
            if img in dataset:
                dataset[img]["captions-pt"].append(pt_caption)

        return list(dataset.values())

    def __get_example_flickr30k(self, index):
        example = self.dataset[index]
        img_path = os.path.join(self.image_base_dir, example["image"])
        img = Image.open(img_path)

        return img, example
