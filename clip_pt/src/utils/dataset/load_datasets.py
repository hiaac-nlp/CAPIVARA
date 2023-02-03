from typing import Tuple, Any

import braceexpand
import webdataset as wds
from torch.utils.data import DataLoader

from utils.dataset.data_collator import ImageMultipleTextDataCollator
from utils.dataset.pracegover_data_collator import PraCegoVerImageMultipleTextDataCollator
from utils.dataset.pracegover_dataset import PraCegoVerDataset


def load_datasets(config, vision_processor, text_tokenizer) -> \
        Tuple[DataLoader[Any], DataLoader[Any]]:
    # train = []
    # for dataset in config.datasets.train:
    #     train += list(braceexpand.braceexpand(dataset['path']))
    #
    # val = []
    # for dataset in config.datasets.validation:
    #     val += list(braceexpand.braceexpand(dataset['path']))
    #
    # train_dataset = wds.WebDataset(train, shardshuffle=True).shuffle(10000)\
    #                                                         .decode("torchrgb") \
    #                                                         .to_tuple("jpg;png", "json") \
    #                                                         .batched(config.batch_size)
    # val_dataset = wds.WebDataset(val, shardshuffle=True).shuffle(10000) \
    #                                                     .decode("torchrgb") \
    #                                                     .to_tuple("jpg;png", "json") \
    #                                                     .batched(config.batch_size)

    train_dataset = PraCegoVerDataset(dataset_path='/hadatasets/pracegover/pracegover_400k.json',
                                      image_base_dir='/hadatasets/pracegover/images/',
                                      split='train',
                                      vision_processor=vision_processor,
                                      text_tokenizer=text_tokenizer,
                                      max_length=config.model.text_padding_size)

    val_dataset = PraCegoVerDataset(dataset_path='/hadatasets/pracegover/pracegover_400k.json',
                                    image_base_dir='/hadatasets/pracegover/images/',
                                    split='val',
                                    vision_processor=vision_processor,
                                    text_tokenizer=text_tokenizer,
                                    max_length=config.model.text_padding_size)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=10)

    return train_dataloader, val_dataloader
