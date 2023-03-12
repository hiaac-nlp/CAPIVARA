import argparse
import tarfile

import braceexpand
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="dataset path", required=True)
args = parser.parse_args()

dataset_path_list = list(braceexpand.braceexpand(args.dataset))

result = 0
for dataset_path in tqdm.tqdm(dataset_path_list):
    tar = tarfile.open(dataset_path)
    result += len(tar.getnames()) // 2  # tar have json and png files for each example

print(">>>>> Total size:", result, " <<<<<")
