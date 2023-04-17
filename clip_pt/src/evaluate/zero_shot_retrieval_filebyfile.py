import argparse
import os

import sys
sys.path.append("./")
sys.path.append("../")

import torch
import tqdm
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import CLIPFeatureExtractor, AutoTokenizer

from utils.dataset.mscoco_dataset import MsCocoDataset
from models.clip_pt_br_wrapper import CLIPPTBRWrapper
from models.mCLIP import mCLIP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", )
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--translation-path", type=str, help="Path to dataset metadata")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--gpu", help="GPU", )

    return parser.parse_args()


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
                  "attention_mask": torch.cat(attention_mask, dim=0), }

    return image_input, text_input


def feature_extraction(model, dataloader, device):
    image_features = []
    text_features = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            image_input, text_input = batch[0].to(device), batch[1]

            # Take the first caption
            text_input = text_input[0]

            text_input = text_tokenizer(
                list(text_input),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=95
            ).to(device) # tokenize

            img_features = model.encode_visual(image_input)
            txt_features = model.encode_text(text_input)

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


def get_txt2img_ground_truth(label_size):
    gt = []
    for idx, i in tqdm.tqdm(enumerate(range(label_size)), desc="Computing ground truth"):
        gt.append(idx)

    n_relevants = gt[-1] + 1
    gt = torch.Tensor(gt).reshape(-1, 1)

    return gt, n_relevants


if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print(">>>>>>> Loading processors")
    vision_processor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir="/hahomes/gabriel.santos/"
    )

    print(">>>>>>> Loading model")
    if args.model_path == "mCLIP":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "M-CLIP/XLM-Roberta-Large-Vit-B-32",
            cache_dir="/hahomes/gabriel.santos/"
        )
        model = mCLIP(device=device)
        vision_processor = model.image_preprocessor
    else:
        text_tokenizer = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            do_lower_case=False,
            cache_dir="/hahomes/gabriel.santos/"
        )
        model = CLIPPTBRWrapper.load_from_checkpoint(args.model_path)

    if args.dataset.lower() == 'mscoco':
        dataset = MsCocoDataset(
            root_dir=args.dataset_path,
            translation_path=args.translation_path,
            vision_processor=vision_processor,
            text_tokenizer=text_tokenizer
        )
    elif args.dataset.lower() == 'flicker30k':
        dataset = Flicker30kDataset(
            root_dir=args.dataset_path,
            translation_path=args.translation_path,
            vision_processor=vision_processor,
            text_tokenizer=text_tokenizer
        )
    else:
        raise NotImplementedError(f"{args.dataset} is not a supported dataset.")

    # Load dataset
    dataloader = DataLoader(dataset, batch_size=args.batch, num_workers=10, shuffle=False, drop_last=False)

    print(">>>>>>> Extracting features")
    image_features, text_features = feature_extraction(model, dataloader, device)

    # free memory
    del model

    print(">>>>>>> Text-to-Image retrieval features")
    top_k_predictions = text_to_image_retrieval(image_features, text_features)
    ground_truth, n_relevants = get_txt2img_ground_truth(len(dataset))

    print(">>>>>>> Computing Recall")
    recall_1 = compute_recall_k(top_k_predictions, ground_truth, k=1, n_relevants=n_relevants)
    recall_5 = compute_recall_k(top_k_predictions, ground_truth, k=5, n_relevants=n_relevants)
    recall_10 = compute_recall_k(top_k_predictions, ground_truth, k=10, n_relevants=n_relevants)
    mr = (recall_1 + recall_5 + recall_10) / 3

    print("Recall@1: ", recall_1.item())
    print("Recall@5: ", recall_5.item())
    print("Recall@10: ", recall_10.item())
    print("Mean Recall: ", mr.item())

