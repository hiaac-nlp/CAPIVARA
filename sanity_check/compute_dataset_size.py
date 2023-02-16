import argparse

import braceexpand
import webdataset as wds
import tqdm
from tqdm.contrib.concurrent import thread_map


def compute_size(dataset_path):
    dataset = wds.WebDataset(dataset_path) \
                .decode("torchrgb") \
                .to_tuple("jpg;png", "json")
    count = 0
    for i, example in tqdm.tqdm(enumerate(dataset)):
        count += 1
    return count

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="dataset path", required=True)
args = parser.parse_args()

dataset_path_list = list(braceexpand.braceexpand(args.dataset))
result = thread_map(compute_size, dataset_path_list, max_workers=20)
print("Total size:", sum(result))