import os
import sys
import time
import tqdm
import torch
import argparse
import pandas as pd
import webdataset as wds
from collections import defaultdict

def change_dict_key_exist(d, old_key, new_key):
    if old_key in d:
        d[new_key] = d.pop(old_key)

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
        "-t_2",
        "--translate_path_2",
        type=str,
        help="Path to the csv translation file"
    )

    parser.add_argument(
        "-s",
        "--save_path",
        default="/hadatasets/clip_pt/final_webdatasets",
        type=str
    )

    parser.add_argument(
        "-d_n",
        "--dataset_name",
        default="wit",
        type=str
    )

    parser.add_argument(
        "-c_d_n",
        "--complementary_dataset_name",
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
        default='\t',
        type=str,
        help="Separator used to read the csv: default = '\t' option =''"
    )

    parser.add_argument(
        "-r",
        "--repetition",
        default='True',
        type=str,
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
    if not os.path.isdir(args.save_path+'/'+args.dataset_name +"_"+ args.complementary_dataset_name):
        os.makedirs(args.save_path+'/'+args.dataset_name +"_"+ args.complementary_dataset_name)

    # loads the translated data
    if args.dataset_name == 'cc12m':
        translation_csv = pd.read_csv(args.translate_path, sep=args.separator, engine='python', names=['url','en-caption', 'caption'])
    elif args.dataset_name == 'wit':
        translation_csv = pd.read_csv(args.translate_path, sep=args.separator, engine='python', names=['en-caption', 'caption','url'])
    else:
        translation_csv = pd.read_csv(args.translate_path, sep=args.separator, engine='python', names=['caption','url'])
    csv_columns = translation_csv.columns
    
    if args.translate_path_2:
        translation_csv_2 = pd.read_csv(args.translate_path_2, sep=args.separator, engine='python', names=['caption'])
        csv_columns_2 = translation_csv_2.columns

    # iterates and saves the data in webdataset
    sink = wds.ShardWriter(args.save_path + '/' + args.dataset_name +"_"+ args.complementary_dataset_name + '/%05d.tar', maxcount=10000)
    if not bool(args.repetition):
        dictionary = defaultdict(lambda: 0)
        print("dictionary created")

    for index, (input, output) in enumerate(dataset):
        if bool(args.repetition) or dictionary[output['caption']] == 0:
            try:
                if ('captions-pt' in output.keys()):
                    output['captions-pt'].append(translation_csv['caption'][int(output['key'][:5])*10000 + int(output['key'][5:])])
                else:
                    output["captions-pt"] = [translation_csv['caption'][int(output['key'][:5])*10000 + int(output['key'][5:])]]
                    if args.translate_path_2:
                        output['captions-pt'].append(translation_csv_2['caption'][int(output['key'][:5])*10000 + int(output['key'][5:])].strip())
                    change_dict_key_exist(output,"caption","captions-en")
                    output["captions-en"] = [output["captions-en"]]
                
                if index%1000==0:
                    print(f"{index:5d}", end="\r", flush=True, file=sys.stderr)
                sample = {
                    "__key__": "sample%05d" % index,
                    "png": input,
                    "json": output,
                }
                sink.write(sample)
                if not bool(args.repetition):
                    dictionary[output['captions-en'][0]] = dictionary[output['captions-en'][0]] + 1
            except:
                print('Something went wrong in the index ' + str(index))
                pass
                
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
