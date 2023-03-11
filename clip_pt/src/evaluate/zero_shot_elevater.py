import os
import sys
import argparse
import logging
# Add previous and current path to search for modules
sys.path.append('./')
sys.path.append('../')

import torch
import torch.nn.functional as F
from elevater.metric import get_metric
from transformers import CLIPFeatureExtractor, AutoTokenizer

from models.mCLIP import mCLIP
from models.model import CLIPTBR
from models.clip_pt_br_wrapper import CLIPPTBRWrapper
from elevater.feature_extractor import extract_feature
from elevater.get_dataloader import construct_dataloader
from elevater.text_feature_extractor import extract_text_features


from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint")
    parser.add_argument("--dataset-name", help="Elevater dataset to be evaluated")
    parser.add_argument("--gpu", help="GPU", )

    return parser.parse_args()


def clip_zeroshot_evaluator(image_features, text_features, image_labels, metric_name, device):
    metric = get_metric(metric_name)

    image_features = torch.from_numpy(image_features).to(device)
    text_features = torch.from_numpy(text_features).to(device)
    image_labels = torch.from_numpy(image_labels).to(device)

    # Normalize image_features
    image_features = F.normalize(image_features)

    # Compute logits
    logits = (100. * image_features @ text_features).softmax(dim=-1)
    result = metric(image_labels.squeeze().cpu().detach().numpy(), logits.cpu().detach().numpy())
    return result, logits, metric.__name__


def main():
    args = parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print(">>>>>>> Loading model")
    if args.model_path == "mCLIP":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "M-CLIP/XLM-Roberta-Large-Vit-B-32",
            cache_dir="/tmp"
        )
        model = mCLIP(device=device)
        vision_processor = model.image_preprocessor
    elif args.model_path == "CLIP-PT":
        vision_processor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="/tmp"
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            do_lower_case=False,
            cache_dir="/tmp"
        )
        # model = CLIPPTBRWrapper.load_from_checkpoint(args.model_path)
        config = OmegaConf.load("../experiment_setup/epoch_finetuning.yaml")
        model = CLIPTBR()
    else:
        raise NotImplementedError(
            f"Model {args.model_path} not implemented"
        )

    test_dataloader = construct_dataloader(
        dataset=args.dataset_name,
        dataset_root="/tmp",
        transform_clip=vision_processor
    )

    image_features, image_labels = extract_feature(
        data_loader=test_dataloader,
        model=model,
        device=device
    )

    text_features = extract_text_features(
        dataset_name=args.dataset_name,
        model=model,
        text_tokenizer=text_tokenizer,
        # device=device
    )

    result, test_predictions, metric = clip_zeroshot_evaluator(image_features, text_features, image_labels, metric)

    msg = f'=> {args.dataset_name} TEST: {metric} {100 * result:.3f}% '
    logging.info(msg)

if __name__ == "__main__":
    main()
