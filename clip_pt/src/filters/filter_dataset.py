import argparse
import os
import sys
from pathlib import Path

import torch
import tqdm
import webdataset as wds
from torch.utils.data import DataLoader

from models.open_CLIP import OpenCLIP
from utils.open_clip_utils import tokenize, format_batch, compute_similarity

sys.path.append("./")
sys.path.append("../")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", help="Path to dataset")
    parser.add_argument("--batch", type=int, help="Batch size", default=1000)
    parser.add_argument("--gpu", help="GPU", )
    parser.add_argument("--postfix-path", type=str, default="_filtered")
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

    print(">>>> Loading dataset")
    dataset = wds.WebDataset(args.dataset_path) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, lang=args.lang, text_tokenizer=text_tokenizer,
                                vision_processor=vision_processor)) \
        .batched(args.batch) \
        .map(lambda x: format_batch(x))

    dataloader = DataLoader(dataset, batch_size=None, num_workers=10)

    path = Path(args.dataset_path)
    parent_dir = path.parent
    dir_path = parent_dir.with_name(parent_dir.name + args.postfix_path)
    os.makedirs(dir_path, exist_ok=True)
    dir_path = dir_path / "%05d.tar"

    sink = wds.ShardWriter(str(dir_path), maxcount=10000)

    model.to(device)
    model.eval()
    index = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Filtering dataset"):
            similarities = compute_similarity(model, batch, device)
            examples = batch[-1]
            for example, sim in zip(examples, similarities):
                if sim >= args.threshold:
                    sample = {
                        "__key__": "sample%05d" % index,
                        "png": example[0],
                        "json": example[1],
                    }
                    sink.write(sample)
                    index += 1
