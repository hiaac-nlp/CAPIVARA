import argparse
from tqdm.contrib.concurrent import thread_map
import torch
import webdataset as wds


def is_correct(item):
    img = item[0]
    # check image
    if not isinstance(img, torch.Tensor) or len(img.shape) != 4 or img.shape[1] != 3:
        return False

    # check text
    data = item[1][0]

    return 'captions-pt' in data and \
        isinstance(data['captions-pt'], list) and len(data['captions-pt']) > 0 and \
        'captions-en' in data and \
        isinstance(data['captions-en'], list) and len(data['captions-en']) > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="dataset path", required=True)
    args = parser.parse_args()
    dataset = wds.WebDataset(args.dataset).decode("torchrgb") \
        .to_tuple("jpg;png", "json") \
        .batched(1)

    result = thread_map(is_correct, dataset, max_workers=100)
    count = 0
    for i, correct in enumerate(result):
        if not correct:
            print(i)
            count += 1
    print('Total incorrect:', count)
