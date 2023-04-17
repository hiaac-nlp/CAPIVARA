import os
import sys
sys.path.append("./")
sys.path.append("../")
import json
import tqdm
import torch
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPFeatureExtractor, AutoTokenizer

from models.mCLIP import mCLIP
from models.clip_pt_br_wrapper import CLIPPTBRWrapper
from utils.dataset.mscoco_dataset import MsCocoDataset
from utils.dataset.flicker30k_dataset import Flicker30kDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--dataset-path", type=str, help="Path to validation/test dataset")
    parser.add_argument("--translation-path", type=str, help="Path to dataset metadata")
    parser.add_argument("--single-caption", default=True, action="store_true", help="Use one caption per sample")
    parser.add_argument("--gpu", type=int, help="GPU", )

    return parser.parse_args()


# calculate recall metric from CLIP model embeddings
def calculate_recall(model, dataloader, caption_index, text_tokenizer, device):
    def get_embeddings(model, dataloader, caption_index, text_tokenizer, device):
        all_embeddings = []
        for batch in tqdm.tqdm(dataloader):
            images, texts = batch
            texts = texts[caption_index]

            texts = text_tokenizer(
                list(texts),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=95
            ) # tokenize

            image_features = model.encode_visual(images.to(device))
            text_features = model.encode_text(texts)
            all_embeddings.append(torch.cat([image_features.cpu(), text_features], dim=1))
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings
    # get all embeddings
    def calculate_recall_from_embeddings(all_embeddings, return_ranks=False):
        # calculate similarity scores
        similarity_scores = compute_similarity(all_embeddings, all_embeddings)
        # calculate ranks
        ranks = np.argsort(-similarity_scores.detach().numpy(), axis=1)
        # calculate top1
        top1 = np.sum(ranks[:,0] == np.arange(len(ranks)))
        # calculate recall
        report_dict = {}
        for k in [1, 5, 10]:
            report_dict[f"recall@{k}"] = np.sum(np.any(ranks[:,:k] == np.arange(len(ranks))[:,None], axis=1)) / len(ranks)
        if return_ranks:
            return report_dict, (ranks, top1)
        else:
            return report_dict
    # get all embeddings
    print(">>>>>> Getting embeddings")
    all_embeddings = get_embeddings(model, dataloader, caption_index, text_tokenizer, device)
    # calculate recall
    print(">>>>>> Calculating recall")
    report_dict, (ranks, top1) = calculate_recall_from_embeddings(all_embeddings, return_ranks=True)
    return report_dict, (ranks, top1)

def compute_similarity(a, b):
    return torch.mm(a, b.t()) / (torch.norm(a, dim=1, keepdim=True) * torch.norm(b, dim=1, keepdim=True).t())


def main():
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

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=10)

    if not args.single_caption:
        for cap_idx in range(text_features.shape[1]):
            report_dict, (ranks, top1) = calculate_recall(model, dataloader, cap_idx, text_tokenizer, device)
            print(report_dict, ranks, top1)
    else:
        cap_idx = 0
        report_dict, (ranks, top1) = calculate_recall(model, dataloader, cap_idx, text_tokenizer, device)
        print(report_dict, ranks, top1)

if __name__ == "__main__":
    main()