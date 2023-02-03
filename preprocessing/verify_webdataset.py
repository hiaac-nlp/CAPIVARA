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

parser.add_argument(
    "-i",
    "--image",
    default="True",
    type=str
)

parser.add_argument(
    "-p_t",
    "--print_type",
    default="segment",
    type=str
)

parser.add_argument(
    "-a",
    "--amount",
    default=-1,
    type=int
)


args = parser.parse_args()

if args.image == "True":
    dataset = wds.WebDataset(args.dataset_path).decode("torchrgb").to_tuple("jpg;png", "json")
else:
    dataset = wds.WebDataset(args.dataset_path)
#dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1)

count = 0
if args.print_type == "full":
    for data in dataset:
        print(data)
        if args.amount != -1 and count >= args.amount:
            break
        count+=1
else:
    for input, output in dataset:
        if args.argument_1:
            try:
                print(output[args.argument_1].decode("utf-8"))
            except:
                print(output[args.argument_1])
        else:
            print(input, output)
        if args.amount != -1 and count >= args.amount:
            break
        count+=1
print("")
print("Total of " + str(count) + " files")

