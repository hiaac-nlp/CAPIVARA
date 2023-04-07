import os
import json

import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

class MsCocoDataset(Dataset):
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
        self.dataset = self.__read_files_mscoco(
            translation_path=translation_path
        )
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __read_files_mscoco(self, translation_path: str):
        """
        :param translation_path: path to the translated dataset metadata file
        :return:
        """
        id2image = {}

        with open(translation_path) as translation_file:
            translation_data = json.load(translation_file)

        for translation_annotation in translation_data["annotations"]:
            if translation_annotation["image_id"] in id2image:
                id2image[translation_annotation["image_id"]]["pt_captions"].append(translation_annotation["caption"])
            else:
                id2image[translation_annotation["image_id"]] = {}
                id2image[translation_annotation["image_id"]]["pt_captions"] = []
                id2image[translation_annotation["image_id"]]["pt_captions"].append(translation_annotation["caption"])

        return id2image

    def __getitem__(self, index: int):
        index2key = list(self.dataset)[index]

        img_path = os.path.join(self.root_dir, f"{str(index2key).zfill(12)}.jpg")
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

        # Limit the number of captions to 5 because there are some images with more than 5 captions
        captions = self.dataset[index2key]["pt_captions"][:5]

        return image_input, captions