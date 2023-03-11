import pathlib
import logging

import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from vision_datasets import DatasetHub
from vision_datasets import ManifestDataset
from vision_datasets import Usages, DatasetTypes
from vision_datasets.pytorch import TorchDataset

VISION_DATASET_STORAGE = 'https://cvinthewildeus.blob.core.windows.net/datasets'

def get_dataset_hub():
    # vision_dataset_json = pathlib.Path("/content/Elevater_Toolkit_IC/vision_benchmark/resources/datasets/vision_datasets.json").read_text()
    vision_dataset_json = (pathlib.Path(__file__).resolve().parents[1] / "elevater" / "resources" / "vision_datasets.json").read_text()
    hub = DatasetHub(vision_dataset_json)

    return hub

def get_dataloader(dataset, batch_size_per_gpu=64, workers=6, pin_memory=True):
    return create_dataloader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=6, pin_memory=True):
    def seed_worker(worker_id):
        worker_seed = worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(0)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=None,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator if shuffle else None,
    )

    return loader


def multilabel_to_vec(indices, n_classes):
    vec = np.zeros(n_classes)
    for x in indices:
        vec[x] = 1
    return vec


def multiclass_to_int(indices):
    return indices[0]


def construct_dataloader(dataset, dataset_root, transform_clip):
    hub = get_dataset_hub()

    vision_dataset_storage = 'https://cvinthewildeus.blob.core.windows.net/datasets'
    local_temp = dataset_root

    results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, dataset, usage=Usages.TEST_PURPOSE)
    if results:
        test_set, test_set_dataset_info, _ = results
    logging.info(f'Test size is {len(test_set.images)}.')

    if test_set_dataset_info.type == DatasetTypes.IC_MULTILABEL:
        previous_transform = transform_clip

        def transform_clip(x, y):
            test_set_ = ManifestDataset(test_set_dataset_info, test_set)
            return (previous_transform(x, return_tensors="pt", padding=True, truncation=True), multilabel_to_vec(y, len(test_set_.labels)))

    elif test_set_dataset_info.type == DatasetTypes.IC_MULTICLASS:
        previous_transform = transform_clip

        def transform_clip(x, y):
            return (previous_transform(x, return_tensors="pt", padding=True, truncation=True), multiclass_to_int(y))

    test_dataloader = get_dataloader(TorchDataset(ManifestDataset(test_set_dataset_info, test_set), transform=transform_clip))

    return test_dataloader