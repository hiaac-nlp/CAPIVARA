from collections import defaultdict

import webdataset as wds
import tqdm

if __name__ == '__main__':
    dataset_path_preproc = '/home/gabriel/dataset_clip_pt/cc3m-2translations/{00000..00266}.tar'
    dataset_preproc = wds.WebDataset(dataset_path_preproc).decode("torchrgb") \
                                                        .to_tuple("jpg;png", "json") \
                                                        .batched(1)

    dataset_path_orig = '/home/gabriel/dataset_original/{00000..00331}.tar'
    dataset_orig = wds.WebDataset(dataset_path_orig).decode("torchrgb") \
                                                    .to_tuple("jpg;png", "json") \
                                                    .batched(1)

    captions = defaultdict(lambda: 0)
    for data in tqdm.tqdm(dataset_preproc):
        captions[data[1][0]['captions-en']] += 1

    for data in tqdm.tqdm(dataset_orig):
        captions[data[1][0]['caption']] -= 1

    count = 0
    for k, v in captions.items():
        if v != 0:
            count += 1
            print(k, '-', v)

    print(count, 'different elements')



