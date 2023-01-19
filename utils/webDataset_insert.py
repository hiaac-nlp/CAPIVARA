import os
import sys
import tqdm
import torch
import argparse
import pandas as pd
import webdataset as wds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--dataset_path",
        type=str,
        help="Path to the directory of the dataset in the format: e.g: http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
    )
    
    parser.add_argument(
        "-t",
        "--translate_path",
        type=str,
        help="Path to the csv translation file"
    )

    parser.add_argument(
        "-s",
        "--save_path",
        default="/hadatasets/clip_pt",
        type=str
    )

    parser.add_argument(
        "-d_n",
        "--dataset_name",
        default="wit",
        type=str
    )

    parser.add_argument(
        "-n_w",
        "--num_workers",
        default=8,
        type=int
    )

    parser.add_argument(
        "-s_p",
        "--separator",
        default='',
        type=str,
        help="Separator used to read the csv: default = '' option ='\t'"
    )


    args = parser.parse_args()

    # loads the original data
    dataset = (
        wds.WebDataset(args.dataset_path, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;png", "json")
        .shuffle(False)
    )

    # check and create storage directory
    if not os.path.isdir(args.save_path+'/'+args.dataset_name):
        os.makedirs(args.save_path+'/'+args.dataset_name)

    # loads the translated data
    translation_csv = pd.read_csv(args.translate_path, sep=args.separator, engine='python', names=['caption', 'url'])
    csv_columns = translation_csv.columns
    
    # iterates and saves the data in webdataset
    sink = wds.ShardWriter(args.save_path + '/' + args.dataset_name + '/%05d.tar', maxcount=10000)
    for index, (input, output) in enumerate(dataset):
        try:
            output["caption-pt"] = translation_csv['caption'][int(output['key'][:5])*10000 + int(output['key'][5:])]

            if index%1000==0:
                print(f"{index:5d}", end="\r", flush=True, file=sys.stderr)
            sample = {
                "__key__": "sample%05d" % index,
                "png": input,
                "json": output,
            }
            sink.write(sample)
        except:
            print('Something went wrong in the index' + str(index))
            print('Sample: ' + str(sample) + '  Output["caption-pt"]: ' + str(output["caption-pt"]))
            pass
if __name__ == "__main__":
    main()
