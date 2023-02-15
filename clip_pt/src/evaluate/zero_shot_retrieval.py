import argparse
import json

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
import tqdm
from transformers import CLIPFeatureExtractor, AutoTokenizer

from models.clip_pt_br_wrapper import CLIPPTBRWrapper
from models.mCLIP import mCLIP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", )
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--translation", choices=["marian", "google"], required=False)
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
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=95
        )
    else:
        if args.translation == "marian":
            text_input = text_tokenizer(
                example[1]["captions-pt"][1::2],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=95
            )
        elif args.translation == "google":
            text_input = text_tokenizer(
                example[1]["captions-pt"][0::2],
                return_tensors="pt",
                padding="max_length",
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
        pixel_values.append(img["pixel_values"])
        input_ids.append(txt["input_ids"])
        attention_mask.append(txt["attention_mask"])

    image_input = {"pixel_values": torch.cat(pixel_values, dim=0)}
    text_input = {"input_ids": torch.cat(input_ids, dim=0),
                  "attention_mask": torch.cat(attention_mask, dim=0),}

    return image_input, text_input


def feature_extraction(model, dataloader, device):
    image_features = []
    text_features = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            image_input, text_input = batch
            image_input["pixel_values"] = image_input["pixel_values"].to(device)
            text_input["input_ids"] = text_input["input_ids"].to(device)
            text_input["attention_mask"] = text_input["attention_mask"].to(device)
            #text_input["token_type_ids"] = text_input["token_type_ids"].to(device)

            batch = image_input, text_input

            if isinstance(model, mCLIP):
                img_features, txt_features = model.encode(batch)
            else:
                img_features, txt_features = model.model(batch)

            norm_img_features = img_features / img_features.norm(dim=1, keepdim=True)
            norm_txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)
            image_features.append(norm_img_features)
            text_features.append(norm_txt_features)

    return image_features, text_features


def text_to_image_retrieval(image_features, text_features):
    top_k = 10
    top_k_predictions = []
    for text_feature in tqdm.tqdm(text_features, desc="t2i retrieval"):
        similarities = []
        for image_feature in image_features:
            scores = text_feature @ image_feature.t()  # shape: [batch_size, batch_size]
            similarities.append(scores)

        similarities = torch.cat(similarities, dim=1)  # shape: [batch_size, #images]
        top_k_pred = torch.argsort(similarities, descending=True)[:, : top_k].cpu()  # shape: [batch_size, top_k]
        top_k_predictions.append(top_k_pred)
        # free memory
        del similarities

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
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    dataset = wds.WebDataset(args.dataset_path) \
        .decode("torchrgb") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, args=args)) \
        .batched(args.batch) \
        .map(format_batch)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=10)

    print(">>>>>>> Loading processors")
    vision_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32",
                                                            cache_dir="/hahomes/gabriel.santos/")

    print(">>>>>>> Loading model")
    if args.model_path == "mCLIP":
        text_tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32",
                                                       cache_dir="/hahomes/gabriel.santos/")
        model = mCLIP()
    else:
        text_tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased",
                                                       do_lower_case=False,
                                                       cache_dir="/hahomes/gabriel.santos/")
        model = CLIPPTBRWrapper.load_from_checkpoint(args.model_path)

    print(">>>>>>> Extracting features")
    image_features, text_features = feature_extraction(model, dataloader, device)

    # free memory
    del model

    print(">>>>>>> Text-to-Image retrieval features")
    top_k_predictions = text_to_image_retrieval(image_features, text_features)
    ground_truth, n_relevants = get_ground_truth(args)

    print(">>>>>>> Computing Recall")
    recall_1 = compute_recall_k(top_k_predictions, ground_truth, k=1, n_relevants=n_relevants)
    recall_5 = compute_recall_k(top_k_predictions, ground_truth, k=5, n_relevants=n_relevants)
    recall_10 = compute_recall_k(top_k_predictions, ground_truth, k=10, n_relevants=n_relevants)
    mr = (recall_1 + recall_5 + recall_10)/3

    print("Recall@1: ", recall_1.item())
    print("Recall@5: ", recall_5.item())
    print("Recall@10: ", recall_10.item())
    print("Mean Recall: ", mr.item())

