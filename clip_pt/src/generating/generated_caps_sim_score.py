import argparse
import os
import sys
from pathlib import Path

import torch
import tqdm
import webdataset as wds

from models.open_CLIP import OpenCLIP
from utils.open_clip_utils import compute_similarity

sys.path.append("./")
sys.path.append("../")


def tokenize(example, text_tokenizer, vision_processor, lang="pt"):
    if len(example[1][f"captions-{lang}"]) > 0:
        captions = list(example[1][f"captions-{lang}"][1])  # Google Translations
    else:
        captions = example[1][f"captions-{lang}"]
    captions += example[1][f"generated-captions-{lang}"]

    text_input = text_tokenizer(captions)
    image_input = vision_processor(example[0])

    return image_input, text_input, example


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", help="Path to dataset")
    parser.add_argument("--gpu", help="GPU", )
    parser.add_argument("--postfix-path", type=str, default="_sim")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--lang", type=str, default="pt")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print(args)

    print(">>>> Loading model")
    model = OpenCLIP()
    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    lang = args.lang.lower()
    print(">>>> Loading dataset")
    dataset = wds.WebDataset(args.dataset_path) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, lang=lang, text_tokenizer=text_tokenizer,
                                vision_processor=vision_processor))

    path = Path(args.dataset_path)
    parent_dir = path.parent
    dir_path = parent_dir.with_name(parent_dir.name + args.postfix_path)
    os.makedirs(dir_path, exist_ok=True)
    dir_path = dir_path / "%05d.tar"

    sink = wds.ShardWriter(str(dir_path), maxcount=10000)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for index, batch in tqdm.tqdm(enumerate(dataset), desc="Computing similarity"):
            similarities = compute_similarity(model, batch, device)
            example = batch[-1]
            example[1][f"similarities-{lang}"] = similarities

            sample = {
                "__key__": "sample%05d" % index,
                "png": example[0],
                "json": example[1],
            }
            sink.write(sample)
