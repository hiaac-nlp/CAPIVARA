import os
import sys
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from functools import partial
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

# Add previous and current path to search for modules
sys.path.append("./")
sys.path.append("../")

import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.open_CLIP import OpenCLIP
from models.open_CLIP_adapter import OpenCLIPAdapter
from models.open_clip_wrapper import OpenCLIPWrapper

if os.environ['LANGUAGE'] == 'pt-BR':
    print("Loading portuguese prompts")
    from utils.resources.translated_prompts import template_map, class_map
elif os.environ['LANGUAGE'] == 'en':
    print("Loading english prompts")
    from utils.resources.english_prompts import template_map, class_map
else:
    raise ValueError("LANGUAGE environment variable must be either 'pt-BR' or 'en'")
from utils.voc2007 import PASCALVoc2007
from utils.metric import get_metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        default="OpenCLIP",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="OpenCLIP",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--model-path",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="GPU",
    )
    parser.add_argument(
        "--open-clip", 
        type=bool, 
        default=False, 
        required=False,
        help="Indicates whether model is fine-tuned (True) or is the original OpenCLIP (False)"
    )
    parser.add_argument(
        "--adapter", 
        type=str,
        default=None, 
        required=False, 
        help="Path to adapter checkpoint"
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to the test set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="Name of the specified dataset.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Specify index path.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Specified dataset.",
    )
    parser.add_argument(
        "--img-batch-size", type=int, default=64, help="Image batch size."
    )    
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for ImageNet dataloader."
    )  
    parser.add_argument(
        "--save-results-json",
        default=False,
        action="store_true",
        help="Save results to json to be submitted"
    )
    args = parser.parse_args()

    return args


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def get_zeroshot_dataset(args, preprocess_fn):
    # Special case in which the dataset is VOC2007 that has multiple labels per image
    if args.dataset == "voc-2007-classification":
        dataset = PASCALVoc2007("/tmp", set="test", transform=preprocess_fn, download=True)
    else:
        dataset = datasets.ImageFolder(args.datapath, transform=preprocess_fn)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.img_batch_size,
        num_workers=args.num_workers,
        sampler=None,
    )

    return DataInfo(dataloader, None)


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    model.to(device)
    model.eval()

    new_classnames = []

    for classname in classnames:
        if type(classname) == list: new_classnames.append(classname[0])

    classnames = new_classnames if len(new_classnames)>0 else classnames

    print(classnames)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if type(templates) == list:
                texts = [template.format(classname) for template in templates]
            else:
                raise ValueError("Template must be a list")
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def run_classification(model, classifier, dataloader, device, index_path = None):
    pred = []
    true = []

    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            image_features = model.encode_visual(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = (100. * image_features @ classifier).softmax(dim=-1)
            
            true.append(target.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)

    if index_path is not None:
        print("Using index to rearrange the logits...")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        pred = pred[index]
        true = true[index]

    return pred, true


def json_prec_dump(data, prec=6):
    return json.dumps(json.loads(json.dumps(data), parse_float=lambda x: round(float(x), prec)))


def main():
    args = parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

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

    if args.open_clip and os.environ['LANGUAGE'] == 'en':
        raise ValueError("A fine-tuned model can be evaluated only in portuguese")

    print(">>>>>>> Loading model")
    if args.open_clip:
        if args.adapter is None:
            model = OpenCLIPWrapper.load_from_checkpoint(args.model_path, strict=False).model
        else:
            model = OpenCLIPAdapter(inference=True, devices=device)
            model.load_adapters(pretrained_adapter=args.model_path)
    else:
        model = OpenCLIP()

    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    # Get eval data.
    print("Preparing zeroshot dataset.")
    data = {}
    data[args.dataset] = get_zeroshot_dataset(
        args, 
        vision_processor
    )

    templates = template_map[args.dataset]
    classnames = class_map[args.dataset]

    metadata_base_dir = "evaluate/utils/resources"
    metadata_name = elevater_dataset_metadata_map[args.dataset]
    metadata_path = os.path.join(metadata_base_dir, metadata_name + ".yaml")
    dataset_metadata = OmegaConf.load(metadata_path)
    metric = get_metric(dataset_metadata.TEST.METRIC)

    # Make inference and evaluation
    classifier = zero_shot_classifier(model, text_tokenizer, classnames, templates, device)
    logits, target = run_classification(model, classifier, data[args.dataset].dataloader, device, args.index if args.index is not None else None)
    
    results = {}
    result = metric(target.cpu().numpy(), logits.cpu().numpy())

    if args.open_clip:
        if args.adapter is None:
            output_path = os.path.join(args.save_dir, args.exp_name+".txt")
            id_name = args.model_path.split("/")[-3]
        else:
            output_path = os.path.join(args.save_dir, "adapter", args.exp_name+".txt")
            id_name = args.model_path
    else:
        output_path = os.path.join(args.save_dir, f"baseline_open_clip-{os.environ['LANGUAGE']}.txt")
        id_name = f"baseline_open_clip_{os.environ['LANGUAGE']}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'a+') as file:
        file.write(
            f"ID: {id_name} | "\
            f"Dataset: {args.dataset} | "\
            f"Metric: {metric.__name__} | "\
            f"Result: {100*result}\n"
        )

    if args.save_results_json:
        results_dict = {
                'model_name': args.model_name,
                'dataset_name': args.dataset,
                'num_trainable_params': 0,
                'num_params': sum(x.numel() for x in model.parameters()),
                'num_visual_params': sum(x.numel() for x in model.model.visual.parameters()),
                'num_backbone_params': sum(x.numel() for x in model.parameters()),
                'n_shot': 0,
                'rnd_seeds': [0],
                'predictions': [logits.cpu().data.numpy().tolist()],
            }
        
        json_string = json_prec_dump(results_dict)

        prediction_folder = os.path.join(os.path.dirname(output_path), args.model_name)
        os.makedirs(prediction_folder, exist_ok=True)
        with open(os.path.join(prediction_folder, f'{args.dataset}.json') , 'w') as outfile:
            outfile.write(json_string)


if __name__ == "__main__":
    main()