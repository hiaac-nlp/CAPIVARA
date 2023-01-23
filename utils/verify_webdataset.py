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

dataset = wds.WebDataset(args.dataset_path).decode("torchrgb").to_tuple("jpg;png", "json")
#dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1)

for input, output in dataset:
    if args.argument_1:
        try:
            print(output[args.argument_1].decode("utf-8"))
        except:
            print(output[args.argument_1])
    else:
        print(input, output)
