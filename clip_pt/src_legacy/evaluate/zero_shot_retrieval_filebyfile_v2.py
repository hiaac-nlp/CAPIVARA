# Based on https://github.com/openai/CLIP/issues/115
import argparse
import os

import sys
sys.path.append("./")
sys.path.append("../")

import torch
import tqdm
import numpy as np
import webdataset as wds
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

def compute_similarity(image_features, text_features, bs = 1000):
    # compute similarity
    max_pairs = image_features.shape[0]
    similarity_scores = torch.zeros(max_pairs, max_pairs)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            print('Processing Visual '+str(v)+' Text '+str(t), end='\r')
            batch_visual_emb = image_features[v:v+bs]
            batch_caption_emb = text_features[t:t+bs]

            logits = batch_visual_emb @ batch_caption_emb.t()
            similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done similarity')
    return similarity_scores

def compute_retrieval(a2b_sims, return_ranks=True):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict

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

    dataloader = DataLoader(dataset, batch_size=1, num_workers=10)

    # fwd all samples
    image_features = []
    text_features = []
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        images, texts = batch
        if args.single_caption:
            texts = [texts[0][0]]
        else:
            texts = [txt[0] for txt in texts]

        texts = text_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=95
        ) # tokenize
        text_emb = model.encode_text(texts) # embed with text encoder
        if not args.single_caption:
            text_emb = text_emb.unsqueeze(0)

        image_emb = model.encode_visual(images.to(device)) # embed with image encoder

        text_features.append(text_emb.detach().cpu())
        image_features.append(image_emb.detach().cpu())


    image_features = torch.cat(image_features, 0)
    text_features = torch.cat(text_features, 0)
    print('Done forward')

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    if not args.single_caption:
        for cap_idx in range(text_features.shape[1]):
            similarity_scores = compute_similarity(image_features, text_features[:,cap_idx,:])
            i2t_dict = compute_retrieval(similarity_scores.numpy())
            t2i_dict = compute_retrieval(similarity_scores.t().numpy())
            print(cap_idx, 'i2t', i2t_dict)
            print(cap_idx, 't2i', t2i_dict)
    else:
        similarity_scores = compute_similarity(image_features, text_features)
        i2t_dict = compute_retrieval(similarity_scores.numpy())
        t2i_dict = compute_retrieval(similarity_scores.t().numpy())
        print('i2t', i2t_dict)
        print('t2i', t2i_dict)


if __name__ == "__main__":
    main()