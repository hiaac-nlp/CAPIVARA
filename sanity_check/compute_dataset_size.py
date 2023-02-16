import argparse

import webdataset as wds
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="dataset path", required=True)
args = parser.parse_args()

dataset = wds.WebDataset(args.dataset)\
            .decode("torchrgb") \
            .to_tuple("jpg;png", "json")

count = 0
for i, example in tqdm.tqdm(enumerate(dataset)):
    count += 1
print(count)