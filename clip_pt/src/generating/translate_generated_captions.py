import argparse

import braceexpand
import tqdm
import webdataset as wds
import pandas as pd

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
        image = example[1]["image"]
        for caption in example[1]["generated-captions-en"]:
            output_data["image"].append(image)
            output_data["generated-captions"].append(caption)
    df = pd.DataFrame(output_data)
    part_size = args.part_size
    n_div = len(df) // part_size
    for i in range(n_div):
        df[i * part_size: (i + 1) * part_size].to_excel(f"{args.output}_{i}.xlsx", index=False)
    if (i + 1) * part_size < len(df):
        df[(i + 1) * part_size:].to_excel(f"{args.output}_{(i + 1)}.xlsx", index=False)


def get_next_translation_df(translations_path):
    translation_path = next(translations_path)
    df = pd.read_excel(translation_path, names=["image", "translated-caption"])
    return df.applymap(lambda x: x.strip())


def merge_translations(args):
    translations_path = braceexpand.braceexpand(args.translations)

    dataset = wds.WebDataset(args.dataset) \
        .decode("pil") \
        .to_tuple("jpg;png", "json")

    sink = wds.ShardWriter(str(args.output), maxcount=10000)
    df = get_next_translation_df(translations_path)

    for index, example in tqdm.tqdm(enumerate(dataset), desc="merging dataset"):
        image_name = example[1]["image"]
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


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.merge:
        merge_translations(args)
    else:
        split_dataset_to_translation(args)