import argparse
import math
import os
from pathlib import Path

import braceexpand
import tqdm
import webdataset as wds
import pandas as pd
from tqdm.contrib.concurrent import process_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the input dataset")
    parser.add_argument("--translations", help="Path to the translated captions")
    parser.add_argument("--output", help="Path to the output")
    parser.add_argument("--merge", default=False)
    parser.add_argument("--part-size", default=100_000)

    return parser.parse_args()


def split_dataset_to_translation(args):
    print(">>>>> Load dataset")
    dataset = wds.WebDataset(args.dataset) \
        .decode("pil") \
        .to_tuple("jpg;png", "json")

    output_data = {"image": [], "generated-captions": []}
    for example in tqdm.tqdm(dataset, desc="spliting dataset"):
        try:
            image = example[1]["image"]
        except:
            image = example[1]["key"]
        for caption in example[1]["generated-captions-en"]:
            output_data["image"].append(image)
            output_data["generated-captions"].append(caption)

    print(">>>>> Saving splits")
    df = pd.DataFrame(output_data)
    part_size = args.part_size
    n_div = len(df) // part_size
    for i in range(n_div):
        df[i * part_size: (i + 1) * part_size].to_excel(f"{args.output}_{i}.xlsx", index=False)
    if (i + 1) * part_size < len(df):
        df[(i + 1) * part_size:].to_excel(f"{args.output}_{(i + 1)}.xlsx", index=False)


def split_shards(shards_list, size):
    for i in range(0, len(shards_list), size):
        yield shards_list[i:i + size]


def split_dataset(params):
    shards, index, output_path = params

    dataset = wds.WebDataset(shards) \
        .decode("pil") \
        .to_tuple("jpg;png", "json")

    output_data = {"image": [], "generated-captions": []}
    for example in tqdm.tqdm(dataset, desc="spliting dataset"):
        try:
            image = example[1]["image"]
        except:
            image = example[1]["key"]
        for caption in example[1]["generated-captions-en"]:
            output_data["image"].append(image)
            output_data["generated-captions"].append(caption)

    print(">>>>> Saving splits")
    df = pd.DataFrame(output_data)
    df.to_excel(f"{output_path}_{index}.xlsx", index=False)

def split_dataset_to_translation_parallel(args):
    print(">>>>> Load dataset")
    shards = list(braceexpand.braceexpand(args.dataset))
    shards = list(split_shards(shards, size=math.ceil(args.part_size/100_000)))
    print(shards)
    shards = [(shard, index, args.output) for index, shard in enumerate(shards)]
    process_map(split_dataset, shards, max_workers=20)


def get_next_translation_df(translations_path):
    translation_path = next(translations_path)
    df = pd.read_excel(translation_path, names=["image", "translated-caption"], dtype=str)
    return df.applymap(lambda x: x.strip())


def merge_translations(args):
    translations_path = braceexpand.braceexpand(args.translations)

    dataset = wds.WebDataset(args.dataset) \
        .decode("pil") \
        .to_tuple("jpg;png", "json")

    dir_path = Path(args.output)
    os.makedirs(dir_path, exist_ok=True)
    dir_path = dir_path / "%05d.tar"

    sink = wds.ShardWriter(str(dir_path), maxcount=10000)
    df = get_next_translation_df(translations_path)

    for index, example in tqdm.tqdm(enumerate(dataset), desc="merging dataset"):
        try:
            image_name = example[1]["image"]
        except:
            image_name = example[1]["key"]

        translated_captions = list(df["translated-caption"][df["image"] == image_name])
        if len(translated_captions) == 0:
            df = get_next_translation_df(translations_path)
            translated_captions = list(df["translated-caption"][df["image"] == image_name])

        example[1]["generated-captions-pt"] = translated_captions
        sample = {
            "__key__": "sample%05d" % index,
            "png": example[0],
            "json": example[1],
        }

        sink.write(sample)
    sink.close()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.merge:
        merge_translations(args)
    else:
        split_dataset_to_translation_parallel(args)
