import argparse
import json

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
import tqdm
from transformers import CLIPFeatureExtractor, AutoTokenizer

from models.clip_pt_br_wrapper import CLIPPTBRWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", )
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--translation", choices=['marian', 'google'], required=False)
    parser.add_argument("--batch", type=int, help="Batch size", )

    return parser.parse_args()


def tokenize(example, args):
    image_input = vision_processor(
        images=example[0],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    text_input = None
    if len(example[1]["captions-pt"]) == 1:
        text_input = text_tokenizer(
            example[1]["captions-pt"][0],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=95
        )
    else:
        if args.translation == 'marian':
            text_input = text_tokenizer(
                example[1]["captions-pt"][1::2],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=95
            )
        elif args.translation == 'google':
            text_input = text_tokenizer(
                example[1]["captions-pt"][0::2],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=95
            )

    return image_input, text_input


def format_batch(batch):
    pixel_values = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for img, txt in zip(batch[0], batch[1]):
        pixel_values.append(img['pixel_values'])
        input_ids.append(txt['input_ids'])
        attention_mask.append(txt['attention_mask'])
        token_type_ids.append(txt['token_type_ids'])

    image_input = {'pixel_values': torch.cat(pixel_values, dim=0)}
    text_input = {'input_ids': torch.cat(input_ids, dim=0),
                  'attention_mask': torch.cat(attention_mask, dim=0),
                  'token_type_ids': torch.cat(token_type_ids, dim=0), }

    return image_input, text_input


def feature_extraction(model, dataloader):
    image_features = []
    text_features = []
    for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
        img_features, txt_features = model(batch)
        norm_img_features = img_features / img_features.norm(dim=1, keepdim=True)
        norm_txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)
        image_features.append(norm_img_features)
        text_features.append(norm_txt_features)
    del model  # free memory

    return image_features, text_features


def text_to_image_retrieval(image_features, text_features):
    top_k = 10
    top_k_predictions = []
    for text_feature in text_features:
        similarities = []
        for image_feature in image_features:
            scores = text_feature @ image_feature.t()  # shape: [batch_size, batch_size]
            similarities.append(scores)

        similarities = torch.cat(similarities, dim=1)  # shape: [batch_size, #images]
        top_k_pred = torch.argsort(similarities, descending=True)[:, : top_k]  # shape: [batch_size, top_k]
        top_k_predictions.append(top_k_pred)

    top_k_predictions = torch.cat(top_k_predictions, dim=0)  # shape: [#texts, top_k]
    return top_k_predictions


def compute_recall_k(predictions, ground_truth, k, n_relevants):
    top_k_preds = predictions[:, :k]
    corrects = (top_k_preds == ground_truth).any(dim=1)
    return sum(corrects) / n_relevants


def get_ground_truth(args):
    dataset = wds.WebDataset(args.dataset_path) \
        .decode("torchrgb") \
        .to_tuple("jpg;png", "json")
    gt = []
    for i, example in tqdm.tqdm(enumerate(dataset), desc="Computing ground truth"):
        n_captions = len(example[1]["captions-pt"])
        if n_captions > 1:
            gt += [i] * (n_captions//2)
        else:
            gt.append(i)

    n_relevants = gt[-1] + 1
    return torch.Tensor(gt).reshape(-1, 1), n_relevants


if __name__ == "__main__":
    args = parse_args()

    dataset = wds.WebDataset(args.dataset_path) \
        .decode("torchrgb") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, args=args)) \
        .batched(args.batch) \
        .map(format_batch)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=10)

    vision_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
    text_tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                   do_lower_case=False)
    model = CLIPPTBRWrapper.load_from_checkpoint(args.model_path)
    image_features, text_features = feature_extraction(model, dataloader)

    top_k_predictions = text_to_image_retrieval(image_features, text_features)
    ground_truth, n_relevants = get_ground_truth(args)

    recall_1 = compute_recall_k(top_k_predictions, ground_truth, k=1, n_relevants=n_relevants)
    recall_5 = compute_recall_k(top_k_predictions, ground_truth, k=5, n_relevants=n_relevants)
    recall_10 = compute_recall_k(top_k_predictions, ground_truth, k=10, n_relevants=n_relevants)
    mr = (recall_1 + recall_5 + recall_10)/3

    print("Recall@1: ", recall_1)
    print("Recall@5: ", recall_5)
    print("Recall@10: ", recall_10)
    print("Mean Recall: ", mr)
