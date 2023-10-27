import os.path

import pandas as pd
from PIL import Image
from transformers import CLIPFeatureExtractor

from utils.dataset.evaluation_dataset import EvaluationDataset


class ImageNetDataset(EvaluationDataset):

    def __init__(
        self,
        dataset_path,
        annotation_path,
        vision_processor,
        text_tokenizer,
        template="Uma foto de [CLASS].",
        lang="pt",
        label_type="fine", # coarse or fine
        max_length=95
    ):
        self.template = template
        self.max_length = max_length
        self.dataset_path = dataset_path
        self.lang = lang
        self.label_type = label_type

        self.df_dataset = pd.read_csv(annotation_path)

        labels_df = self.df_dataset[
            [
                "id",
                "translated_fine_grained_label",
                "translated_coarse_grained_label",
                "fine_grained_label"
            ]
        ].value_counts().reset_index(name='count')
        labels_df["id"] = pd.to_numeric(labels_df["id"])
        labels_df = labels_df.sort_values(by=["id"], ascending=True)

        if lang == "pt":
            if label_type=="coarse":
                self.labels = labels_df["translated_coarse_grained_label"].values.tolist()
            else:
                self.labels = labels_df["translated_fine_grained_label"].values.tolist()
        else:
            self.labels = labels_df["fine_grained_label"].values.tolist()

        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        # return len(self.df_dataset)
        return 5000

    def __getitem__(self, index):
        image_filename = self.df_dataset.iloc[index]["filename"]
        label_id = int(self.df_dataset.iloc[index]["id"])-1 # The Id starts at 1
        image_path = os.path.join(self.dataset_path, image_filename)

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

        return self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=95
        )