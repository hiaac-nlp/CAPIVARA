import csv
import torch
import argparse
import pandas as pd
import webdataset as wds

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--dataset_path",
    type=str,
    help="Path to the directory of the dataset in the format: e.g: http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
)

parser.add_argument(
    "-n",
    "--dataset_name",
    type=str
)

args = parser.parse_args()

dataset = wds.WebDataset(args.dataset_path)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1)

with open(args.dataset_name+'.tsv', 'w',encoding='UTF8') as f:
    print(':)')
    writer = csv.writer(f,delimiter='\t')
    for data in dataloader:
        writer.writerow([data['caption_reference_description.txt'][0].decode("utf-8"),data['image_url.txt'][0].decode("utf-8")])
