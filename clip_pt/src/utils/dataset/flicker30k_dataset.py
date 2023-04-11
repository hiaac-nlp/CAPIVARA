import json
import os
import itertools

import tqdm
from PIL import Image
from transformers import PreTrainedTokenizer
from torch.utils.data.dataset import Dataset

class Flicker30kDataset(Dataset):
    def __init__(
            self,
            root_dir,
            translation_path,
            vision_processor,
            text_tokenizer
        ):
        """
        :param root_dir: root directory
        :param translation_path:
        :param vision_processor:
        :param text_tokenizer:
        """
        self.root_dir = root_dir
        self.dataset = self.__read_metadata_flicker30k(
            translation_path=translation_path
        )
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __read_metadata_flicker30k(self, translation_path: str):
        """
        :param translation_path: path to the translated dataset metadata file
        :return:
        """
        with open(translation_path, "r") as translation_file:
            translation_data = [line.rstrip() for line in translation_file.readlines()]

        chunks = []
        N = 5 # Number of captions per sample

        for i in range(0, len(translation_data), N):
            # Extract each group of N elements as a sublist
            chunk = translation_data[i:i + N]
            # Append the sublist to the list of chunks
            chunks.append(chunk)

        metadata_dict = {}
        for chunk in chunks:
            filename = chunk[0].split("#")[0]
            captions = [element.split("\t")[1] for element in chunk]
            metadata_dict[filename] = captions

        return metadata_dict

    def __getitem__(self, index: int):
        file_name = list(self.dataset.keys())[index]

        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path)

        if isinstance(self.vision_processor, PreTrainedTokenizer):
            image_input = self.vision_processor(
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        else:
            image_input = self.vision_processor(image)

        captions = list(self.dataset.values())[index]

        return image_input, captions