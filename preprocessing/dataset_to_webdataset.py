import os
import sys
import time
import tqdm
import torch
import argparse
import pandas as pd
import webdataset as wds
from collections import defaultdict
import torchvision.transforms as T

import json
from PIL import Image
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name: str, dataset_path: str, image_base_dir: str, translation_path: str, split="train"):
        """
        :param dataset_name: the name of the dataset (pracegover, mscoco, flickr30k)
        :param dataset_path: the path to the dataset
        :param image_base_dir: the path to the directory with the images
        :param split: possible options - train, val or test
        """
        self.dataset_name = dataset_name
        self.image_base_dir = image_base_dir

        self.transform = transforms.ToTensor()

        if dataset_name.lower() == "pracegover":
            self.dataset = self.__read_pracegover(dataset_path, split)
        elif dataset_name.lower() == "mscoco":
            self.dataset = self.__read_mscoco(path=dataset_path, translation_path=translation_path)
        elif dataset_name.lower() == "flickr30k":
            self.dataset = self.__read_flickr30k(path=dataset_path,
                                                 translation_path=translation_path,
                                                 split=split)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset_name.lower() == "pracegover":
            img, annotations = self.__get_example_pracegover(index)
        elif self.dataset_name.lower() == "mscoco":
            img, annotations = self.__get_example_mscoco(index)
        elif self.dataset_name.lower() == "flickr30k":
            img, annotations = self.__get_example_flickr30k(index)

        try:
            return self.transform(img), annotations
        except:
            print("Image error, index: " + str(index))
            return False, False

    def __read_mscoco(self, path: str, translation_path: str):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--dataset_path",
        type=str,
        help="Path to the directory of the dataset in the format: e.g: http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
    )

    parser.add_argument(
        "-i_p",
        "--image_dataset_path",
        type=str,
    )

    parser.add_argument(
        "-t",
        "--translate_path",
        type=str,
        help="Path to the csv translation file"
    )

    parser.add_argument(
        "-s_p_l",
        "--split",
        default="train",
        type=str
    )

    parser.add_argument(
        "-s",
        "--save_path",
        default="/hadatasets/clip_pt/final_webdatasets",
        type=str
    )

    parser.add_argument(
        "-d_n",
        "--dataset_name",
        default="wit",
        type=str
    )

    parser.add_argument(
        "-s_p",
        "--separator",
        default='\t',
        type=str,
        help="Separator used to read the csv: default = '\t' option =''"
    )

    parser.add_argument(
        "-r",
        "--repetition",
        default='True',
        type=str,
    )


    args = parser.parse_args()

    dataset = CustomDataset(args.dataset_name, args.dataset_path, args.image_dataset_path, args.translate_path)
    
    # check and create storage directory
    if not os.path.isdir(args.save_path+'/'+args.dataset_name):
        os.makedirs(args.save_path+'/'+args.dataset_name)

    # iterates and saves the data in webdataset
    sink = wds.ShardWriter(args.save_path + '/' + args.dataset_name + '/%05d.tar', maxcount=10000)
    if not bool(args.repetition):
        dictionary = defaultdict(lambda: 0)
        print("dictionary created")

    transform = T.ToPILImage()
    caption_type = "captions-pt" if args.dataset_name == "pracegover" else "captions-en"

    for index, (input, output) in enumerate(dataset):
        if torch.is_tensor(input):
            if bool(args.repetition) or dictionary[output[caption_type][0]] == 0:
                try:
                    if index%1000==0:
                        print(f"{index:5d}", end="\r", flush=True, file=sys.stderr)
                    sample = {
                        "__key__": "sample%05d" % index,
                        "png": transform(input),
                        "json": output,
                    }
                    sink.write(sample)
                    if not bool(args.repetition):
                        dictionary[output[caption_type][0]] = dictionary[output[caption_type][0]] + 1
                except:
                    print('Something went wrong in the index' + str(index))
                    pass

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))