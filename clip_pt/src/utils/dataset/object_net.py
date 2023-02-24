import json
import os

import tqdm
from PIL import Image

from utils.dataset.evaluation_dataset import EvaluationDataset


class ObjectNetDataset(EvaluationDataset):
    def __init__(self, root_dir, translation_path, vision_processor, text_tokenizer,
                 template="Uma imagem de [CLASS]"):
        """

        :param root_dir: root directory
        :param translation_path:
        :param vision_processor:
        :param text_tokenizer:
        :param template: a text template that follows the format "uma imagem de [CLASS]"
        """
        self.root_dir = root_dir
        self.template = template
        with open(translation_path) as file:
            self.translations = json.load(file)

        self.labels = os.listdir(root_dir)
        self.label_to_idx = {cls_name: i for i, cls_name in enumerate(self.labels)}

        self.images = []
        for cls_name in tqdm.tqdm(self.labels):
            cls_dir = os.path.join(root_dir, cls_name)
            for filename in os.listdir(cls_dir):
                image_path = os.path.join(cls_dir, filename)
                self.images.append((image_path, self.label_to_idx[cls_name]))

        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label_id = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        image_input = self.vision_processor(
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        return image_input, label_id

    def get_labels(self):
        # replace the occurrences of [CLASS] to the translated label
        texts = [self.template.replace("[CLASS]", self.translations[label])
                 for label in self.labels]

        return self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=95
        )
