import sys
# add previous and current path
sys.path.append('./')
sys.path.append('../')

import os
import json
import argparse

import clip
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor

from utils.model import Clip_Multimodal
from utils.clip_multimodal_wrapper import ClipMultimodalWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform, value_image_list):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = value_image_list

    def __len__(self):
        return len(self.total_imgs)

    def get_image_name(self, idx):
        return self.total_imgs[idx]

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image


class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def text_encoder(text, clip_processor, model_clip):
    text_data = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    with torch.no_grad():
        text_features = model_clip.get_text_features(
            input_ids=text_data['input_ids'],
            attention_mask=text_data['attention_mask']
        )
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def precompute_text_features(loader, clip_processor, model_clip):
    text_features_arr = []

    for texts in tqdm(loader):
        text_data = clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_data.to(device)
        with torch.no_grad():
            text_features = model_clip.get_text_features(
                input_ids=text_data['input_ids'],
                attention_mask=text_data['attention_mask']
            )
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_features_arr.extend(text_features.cpu().detach().numpy())

    return np.array(text_features_arr)


def precompute_image_features(loader, model_clip):
    image_features_arr = []
    for images in tqdm(loader):

        with torch.no_grad():
            image_features = model_clip.get_image_features(images.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features_arr.extend(image_features.cpu().detach().numpy())

    return np.array(image_features_arr)


def find_image(text_query, dataset, image_features, model_clip, clip_processor, n=1):
    text_data = clip_processor(
            text=text_query,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
    text_data.to(device)
    with torch.no_grad():
        text_features = model_clip.get_text_features(
            input_ids=text_data['input_ids'],
            attention_mask=text_data['attention_mask']
        )
    text_features /= text_features.norm(dim=-1, keepdim=True)

    zeroshot_weights = text_features.cpu().detach().numpy()
    distances = np.dot(image_features, zeroshot_weights.reshape(-1, 1))
    file_paths = []
    for i in range(1, n+1):
        idx = np.argsort(distances, axis=0)[-i, 0]
        file_paths.append("/content/val2017/"+get_path_image(idx))
    return file_paths


def show_images(image_list):
    for im_path in image_list:
        display(Image(filename=im_path))


def get_path_image(index, df):
    return df.iloc[index]["image"]


def compute_mrr(data, dataset, n, image_features, text_features, df):
    collect_rr = []

    pbar = tqdm(total=len(data), position=0, leave=True)

    found = np.matmul(image_features, text_features.T)
    for index, distances in enumerate(found):
        # print(distances, index, np.max(distances), np.argmax(distances))
        # exit()
        pbar.update(1)
        image_path = get_path_image(index, df)
        collect_rr.append(new_rr(distances, image_path, dataset, n, df))

    pbar.close()
    return np.average(collect_rr)


def new_rr(distances, target_image, dataset, n, df):
    image_paths = []
    idxs = distances.argsort()[-n:][::-1]

    for idx in idxs:
        image_paths.append(get_path_image(idx, df))

    if target_image in image_paths:
        return 1/(image_paths.index(target_image) + 1)
    else:
        return 0


def internal_hits(distances, target_image, dataset, n, df):
    image_paths = []
    idxs = distances.argsort()[-n:][::-1]
    for idx in idxs:
        image_paths.append(get_path_image(idx, df))

    if target_image in image_paths:
        return 1
    else:
        return 0


def compute_hits(data, dataset, n, text_features, image_features, df):
    collect_rr = []

    pbar = tqdm(total=len(data), position=0, leave=True)

    found = np.matmul(text_features, image_features.T)
    for index, distances in enumerate(found):
        pbar.update(1)
        image_path = get_path_image(index, df)
        collect_rr.append(internal_hits(distances, image_path, dataset, n, df))

    pbar.close()
    return np.average(collect_rr)


def get_mrr(img_loader, text_loader, model_clip, clip_processor, df):
    image_features = precompute_image_features(
        img_loader,
        model_clip
    )

    text_features = precompute_text_features(
        loader=text_loader,
        clip_processor=clip_processor,
        model_clip=model_clip
    )

    print('MRR@1:', compute_mrr(df['caption'].values.tolist(), df["image"].values.tolist(), 1, image_features, text_features, df))
    print('MRR@5:', compute_mrr(df['caption'].values.tolist(), df["image"].values.tolist(), 5, image_features, text_features, df))
    print('MRR@10:', compute_mrr(df['caption'].values.tolist(), df["image"].values.tolist(), 10, image_features, text_features, df))

    print(compute_hits(df['caption'].values.tolist(), df["image"].values.tolist(), 100, text_features, image_features, df))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='imagenet',
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default="checkpoints/batch_8_acc_4_last.ckpt",
        help="path of checkpoint pt file, for continue training"
    )
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint_path}")

    pl_model = ClipMultimodalWrapper().load_from_checkpoint(args.checkpoint_path)
    pl_model.to(device)

    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model_clip.to(device)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _, preprocess = clip.load("ViT-B/32")

    dataset = json.load(open("evaluate/data/annotations/captions_val2017.json", 'r'))

    images_name = []
    captions = []

    for ann in dataset['annotations']:
        image_id = str(ann["image_id"])
        image_name = (12-len(image_id))*"0" + image_id + ".jpg"
        caption = ann["caption"]

        assert os.path.exists("evaluate/data/val2017/"+image_name)

        images_name.append(image_name)
        captions.append(caption)

    df = pd.DataFrame(
        list(
            zip(
                images_name,
                captions
            )
        ),
        columns=['image', 'caption']
    )

    # Drops other captions
    df.drop_duplicates(subset=['image'], inplace=True)

    img_dataset = CustomDataSet(
        main_dir="evaluate/data/val2017",
        transform=preprocess,
        value_image_list=df["image"].values.tolist()
    )

    img_loader = DataLoader(
        dataset=img_dataset,
        batch_size=24,
        shuffle=False,
        drop_last=False
    )

    text_dataset = SimpleTextDataset(df['caption'].values.tolist())

    text_loader = DataLoader(
        dataset=text_dataset,
        batch_size=24,
        shuffle=False,
        drop_last=False
    )

    get_mrr(
        img_loader=img_loader,
        text_loader=text_loader,
        # model_clip=model_clip,
        clip_processor=clip_processor,
        model_clip=pl_model.model.model_clip,
        df=df
    )



if __name__ == "__main__":
    main()