import os
import sys
import argparse
import logging
# Add previous and current path to search for modules
sys.path.append("./")
sys.path.append("../")

import tqdm
import torch
from models.mCLIP import mCLIP
from omegaconf import OmegaConf
import torch.nn.functional as F
from elevater.metric import get_metric
from models.clip_pt_br_wrapper_image_classification import CLIPPTBRWrapperImageClassification
from elevater.feature_extractor import extract_feature
from elevater.get_dataloader import construct_dataloader
from transformers import CLIPFeatureExtractor, AutoTokenizer
from elevater.text_feature_extractor import extract_text_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="mCLIP",
        help="Name of the model to be evaluated"
    )
    parser.add_argument(
        "--model-path",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="GPU",
    )
    parser.add_argument(
        "--runs",
        default="single_run",
        choices=["single_run", "multiple_runs"],
        help="Evaluate one dataset or multiple datasets"
    )
    parser.add_argument(
        "--dataset_name",
        default="cifar-10",
        help="Dataset to be evaluated in single run"
    )

    return parser.parse_args()


def clip_zeroshot_evaluator(image_features, text_features, image_labels, metric_name, device):
    metric = get_metric(metric_name)

    image_features = torch.from_numpy(image_features).to(device)
    text_features = text_features.to(device)
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

    metadata_base_dir = "evaluate/elevater/resources"

    elevater_dataset_metadata_map = {
        "cifar-10": "cifar10",
        "cifar-100": "cifar100",
        "caltech-101": "caltech101",
        "gtsrb": "gtsrb",
        "country211": "country211",
        "fgvc-aircraft-2013b-variants102": "fgvc-aircraft-2013b",
        "oxford-flower-102": "flower102",
        "food-101": "food101",
        "kitti-distance": "kitti-distance",
        "mnist": "mnist",
        "patch-camelyon": "patchcamelyon",
        "resisc45_clip": "resisc45-clip",
        "stanford-cars": "stanfordcar",
        "voc-2007-classification": "voc2007classification",
        "oxford-iiit-pets": "oxford-iiit-pets",
        "eurosat_clip": "eurosat-clip",
        "hateful-memes": "hateful-memes",
        "rendered-sst2": "rendered-sst2",
        "dtd": "dtd",
        "fer-2013": "fer2013",
    }

    if args.runs == "single_run":
        assert args.dataset_name in elevater_dataset_metadata_map.keys(), f"Dataset {args.dataset_name} not available for evaluation"
        dataset_names = [args.dataset_name]

    else:
        dataset_names = elevater_dataset_metadata_map.keys()

    print(">>>>>>> Loading model")
    if args.model_name == "mCLIP":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "M-CLIP/XLM-Roberta-Large-Vit-B-32",
            cache_dir="/tmp"
        )
        model = mCLIP(device=device)
        vision_processor = model.image_preprocessor
    elif args.model_name == "CLIP-PT":
        vision_processor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="/tmp"
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            do_lower_case=False,
            cache_dir="/tmp"
        )
        model = CLIPPTBRWrapperImageClassification.load_from_checkpoint(args.model_path)
        model = model.model
    else:
        raise NotImplementedError(
            f"Model {args.model_name} not implemented"
        )

    for dataset_name in tqdm.tqdm(dataset_names):
        metadata_name = elevater_dataset_metadata_map[dataset_name]
        metadata_path = os.path.join(metadata_base_dir, metadata_name + ".yaml")
        dataset_metadata = OmegaConf.load(metadata_path)
        metric_name = dataset_metadata.TEST.METRIC

        test_dataloader = construct_dataloader(
            dataset=dataset_name,
            dataset_root="/tmp",
            transform_clip=vision_processor
        )

        image_features, image_labels = extract_feature(
            data_loader=test_dataloader,
            model=model,
            device=device
        )

        text_features = extract_text_features(
            dataset_name=dataset_name,
            model=model,
            text_tokenizer=text_tokenizer,
            device=device
        )

        result, test_predictions, metric = clip_zeroshot_evaluator(image_features, text_features, image_labels, metric_name, device=device)

        msg = f"=> {args.model_name} | {dataset_name} | TEST: {metric} {100 * result:.3f}% "
        logging.info(msg)


if __name__ == "__main__":
    main()
