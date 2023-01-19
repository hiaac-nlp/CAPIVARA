import webdataset as wds
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--dataset_path",
    type=str,
    help="Path to the directory of the dataset in the format: e.g: http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
)

parser.add_argument(
    "-a1",
    "--argument_1",
    type=str
)

args = parser.parse_args()

dataset = wds.WebDataset(args.dataset_path)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1)

for data in dataloader:
    if args.argument_1:
        print(data[args.argument_1][0].decode("utf-8"))
    else:
        print(data)
