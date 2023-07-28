import json
import os
import random
from typing import Dict

import braceexpand
import numpy as np
import webdataset as wds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset.grocery_store_dataset import GroceryStoreDataset

vectorizer = TfidfVectorizer()


def similarity(captions):
    tfidf_vectors = vectorizer.fit_transform(captions)
    return cosine_similarity(tfidf_vectors)


def remove_similar(captions, k_min=3, thr=0.3):
    """
    Remove similar texts keeping the maximum diversity among them

    :param captions: image captions
    :param k_min: minimum number of texts to keep
    :param thr: maximum similarity between texts allowed
    :return: filtered captions, in which similar text were removed
    """

    if len(captions) < k_min:
        return captions

    sim_matrix = similarity(captions)
    n_nodes = sim_matrix.shape[0]
    sim_matrix = sim_matrix - np.eye(n_nodes)
    while not (sim_matrix <= thr).all() and n_nodes > k_min:
        cost = sim_matrix.sum(axis=0)
        i = np.argmax(cost)
        sim_matrix[i, :] = 0
        sim_matrix[:, i] = 0
        n_nodes -= 1

    cost = sim_matrix.sum(axis=0)
    remove_indices = np.where(cost == 0)[0].tolist()
    return [caption for i, caption in enumerate(captions) if i not in remove_indices]


def image_augmentation(image):
    augmentation = [transforms.Resize(250),
                    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                    transforms.AugMix(severity=2),
                    transforms.ToPILImage()]

    augmentation = transforms.Compose(augmentation)
    return augmentation(image)


def filter_by_ranking_captions(annotations, k=5):
    top_k = np.argsort(annotations["similarities-pt"])[-k:].tolist()
    return [annotations["captions-pt"][i] for i in top_k]


def filter_by_threshold(annotations, thr=0.2):
    return [caption for caption, sim in zip(annotations["captions-pt"], annotations["similarities-pt"]) if sim >= thr]


def tokenize(example, vision_processor, text_tokenizer, config):
    img = example[0]
    if config.get("augment", True):
        img = image_augmentation(img)

    image_input = vision_processor(img)

    if config.get("self_distill", False):
        n_captions = len(example[1]["captions-en"])
        caption_index = random.randint(0, n_captions - 1)

        # Google translation (w/ even indices)
        sample_pt = example[1]["captions-pt"][2 * caption_index + 1]
        sample_en = example[1]["captions-en"][caption_index]

        text_pt_input = text_tokenizer(sample_pt)
        text_en_input = text_tokenizer(sample_en)
        return image_input, text_pt_input, text_en_input
    else:
        lang = config.get("lang", "pt")

        captions = example[1][f"captions-{lang}"]
        generated_captions_strategy = config.get("generated_captions", None)

        if generated_captions_strategy == "all":
            captions += example[1][f"generated-captions-{lang}"]
        elif generated_captions_strategy == "filter-by-ranking":
            captions = filter_by_ranking_captions(example[1], k=config.get("keep_captions", 5))
        elif generated_captions_strategy == "filter-by-threshold":
            captions = filter_by_threshold(example[1], thr=config.get("threshold", 0.2))
        elif generated_captions_strategy == "filter-by-threshold-diversity":
            captions = filter_by_threshold(example[1], thr=config.get("threshold", 0.2))
            captions = remove_similar(captions, k_min=config.get("k_min", 3))
        elif isinstance(generated_captions_strategy, int):
            k = generated_captions_strategy
            captions += random.sample(example[1][f"generated-captions-{lang}"], k=k)
            print(captions)

        if len(captions) == 0:
            return None  # filter example out

        # take a random caption
        text_input = text_tokenizer(random.choice(captions))
        return image_input, text_input


def tokenize_validation(example, vision_processor, text_tokenizer):
    img = example[0]
    image_input = vision_processor(img)

    captions = example[1]["captions-pt"]
    text_input = text_tokenizer(random.choice(captions))
    return image_input, text_input


def format_batch(batch, self_distill=False):
    image_input = batch[0]
    if self_distill:
        text_pt_input = batch[1].reshape((-1, 77))
        text_en_input = batch[2].reshape((-1, 77))

        return image_input, text_pt_input, text_en_input
    else:
        text_input = batch[1].reshape((-1, 77))
        return image_input, text_input


def load_datasets(config, vision_processor, text_tokenizer) -> Dict:
    """
        previously computed dataset sizes. This is necessary because __len__ method in WebDataset
        returns an inaccurate value, so we have to set it manually.
        Reference: https://webdataset.github.io/webdataset/sharding/
    """
    current_path = os.path.dirname(__file__)
    with open(os.path.join(current_path, "datasets_size.json")) as file:
        datasets_sizes = json.load(file)

    print(">>>>> Train datasets:", [dataset['path'] for dataset in config.datasets.train])
    print(">>>>> Validation datasets:", [dataset['path'] for dataset in config.datasets.validation])

    train = []
    train_size = 0
    for dataset in config.datasets.train:
        train_size += datasets_sizes["train"][dataset['name']]
        train += list(braceexpand.braceexpand(dataset['path']))

    val = []
    val_size = 0
    for dataset in config.datasets.validation:
        val_size += datasets_sizes["validation"][dataset['name']]
        val += list(braceexpand.braceexpand(dataset['path']))

    train_dataset = wds.WebDataset(train, shardshuffle=True) \
        .shuffle(10000) \
        .decode("torchrgb8") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, vision_processor, text_tokenizer, config)) \
        .batched(config.batch_size) \
        .map(lambda x: format_batch(x, self_distill=config.get("self_distill", False)))

    val_dataset = wds.WebDataset(val, shardshuffle=True) \
        .shuffle(10000) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize_validation(x, vision_processor, text_tokenizer)) \
        .batched(config.batch_size) \
        .map(lambda x: format_batch(x))

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=10)

    print("train_size:", train_size)

    output = {"train_dataloader": train_dataloader,
              "train_size": train_size,
              "val_dataloader": val_dataloader,
              "val_size": val_size}

    if config.datasets.get("img_classification", False):
        img_classif_dataset = GroceryStoreDataset(
            dataset_path=config.datasets.img_classification.path,
            annotation_path=config.datasets.img_classification.annotation_path,
            vision_processor=vision_processor,
            text_tokenizer=text_tokenizer,
            max_length=77,
            open_clip=True)

        img_classif_dataloader = DataLoader(img_classif_dataset, batch_size=config.batch_size,
                                            num_workers=10)

        output["img_classification"] = img_classif_dataloader
        output["img_classif_labels"] = img_classif_dataset.get_labels()

    return output


