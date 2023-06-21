import argparse
import webdataset as wds
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to the input dataset")
    parser.add_argument("--output", help="Path to the output")
    parser.add_argument("--merge", default=False)
    parser.add_argument("--part-size", default=100_000)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    print(">>>>> Load dataset")
    dataset = wds.WebDataset(args.input)\
        .decode("pil") \
        .to_tuple("jpg;png", "json") \


    output_data = {"image": [], "generated-captions":[]}
    for example in dataset:
        image = example[1]["image"]
        for caption in example[1]["generated-captions-en"]:
            output_data["image"].append(image)
            output_data["generated-captions"].append(caption)

    df = pd.DataFrame(output_data)

    part_size = args.part_size
    n_div = len(df) // part_size
    for i in range(n_div):
        df[i * part_size: (i + 1) * part_size].to_csv(f"{args.output}_{i}.csv", index=False)

    if (i + 1) * part_size < len(df):
        df[(i + 1) * part_size:].to_csv(f"{args.output}_{(i + 1)}.csv", index=False)