import os.path

import pandas as pd
from PIL import Image
from transformers import CLIPFeatureExtractor

from utils.dataset.evaluation_dataset import EvaluationDataset


class GroceryStoreDataset(EvaluationDataset):

    def __init__(self, dataset_path, annotation_path, vision_processor, text_tokenizer,
                 template="Uma foto de [CLASS].", lang="pt", max_length=95, open_clip=False):
        self.template = template
        self.max_length = max_length
        self.open_clip = open_clip
        self.root_dir = os.path.dirname(dataset_path)
        self.df_dataset = pd.read_csv(dataset_path, names=["filepath", "fine_id", "coarse_id"])

        df_annotations = pd.read_csv(annotation_path)
        if lang == "pt":
            idx_to_label = {row["Coarse Class ID (int)"]: row["Translated Coarse Class Name (str)"]
                            for _, row in df_annotations.iterrows()}
        else:
            idx_to_label = {row["Coarse Class ID (int)"]: row["Coarse Class Name (str)"]
                            for _, row in df_annotations.iterrows()}
        self.labels = idx_to_label.values()

        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, index):
        image_path, label_id = self.df_dataset.loc[index, ["filepath", "coarse_id"]]
        image_path = os.path.join(self.root_dir, image_path)

        image = Image.open(image_path)

        if isinstance(self.vision_processor, CLIPFeatureExtractor):
            image_input = self.vision_processor(
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        else:
            image_input = self.vision_processor(image)

        return image_input, label_id

    def get_labels(self):
        # replace the occurrences of [CLASS] to the translated label
        texts = [self.template.replace("[CLASS]", label) for label in self.labels]
        print(texts)

        if self.open_clip:
            return self.text_tokenizer(texts)

        return self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )


