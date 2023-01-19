import os
import sys

# add previous and current path
sys.path.append('./')
sys.path.append('../')

import argparse


import clip
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet

from utils.model import Clip_Multimodal
from utils.clip_multimodal_wrapper import ClipMultimodalWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(output, target, topk=(1,)):
    output = torch.from_numpy(np.asarray(output))
    target = torch.from_numpy(np.asarray(target))
    pred = output.topk(max(topk), dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_dataset_acc(image_dataset, clip_processor, model_clip):

    preds = []
    targets = []

    loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=16,
        num_workers=2
    )

    text_descriptions = [f"This is a photo of a {label}" for label in image_dataset.classes]

    text_data = clip_processor(
        text=text_descriptions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    text_data.to(device)

    with torch.no_grad():
        text_features = model_clip.get_text_features(
            input_ids=text_data['input_ids'],
            attention_mask=text_data['attention_mask']
        )
        text_features /= text_features.norm(dim=-1, keepdim=True)

    top_ns = [1, 5, 10]
    acc_counters = [0. for _ in top_ns]
    n = 0.
    model_clip.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.to(device)
            target = target.numpy()
            # predict
            image_features = model_clip.get_image_features(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features.detach().cpu().numpy()  @ text_features.detach().cpu().numpy().T

            # measure accuracy
            accs = accuracy(logits, target, topk=top_ns)
            for j in range(len(top_ns)):
                acc_counters[j] += accs[j]
            n += images.shape[0]

            output = torch.from_numpy(np.asarray(logits))
            pred = output.topk(max(top_ns), dim=1, largest=True, sorted=True)[1].t().detach().cpu().numpy()[0]

            preds.extend(pred)
            targets.extend(target)

        tops = {f'top{top_ns[i]}': acc_counters[i] / n * 100 for i in range(len(top_ns))}

        print(tops)

        return targets, preds


def save_conf_matrix(targets, preds, classes, output_path):
    cm = confusion_matrix(
        y_true=targets,
        y_pred=preds
    )

    df_cm = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    plt.figure(figsize = (24,12))
    plot = sns.heatmap(df_cm, annot=True,  fmt='g')
    figure1 = plot.get_figure()
    plt.tight_layout()
    figure1.savefig(output_path, format='png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='imagenet',
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default="checkpoints/batch_8_acc_4_last.ckpt",
        help="path of checkpoint pt file, for continue training"
    )
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint_path}")

    pl_model = ClipMultimodalWrapper().load_from_checkpoint(args.checkpoint_path)
    pl_model.to(device)

    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model_clip.to(device)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _, preprocess = clip.load("ViT-B/32")

    if args.dataset_name == "cifar10":
        dataset = CIFAR10(
            root="/tmp",
            transform=preprocess,
            train=False,
            download=True
        )

    if args.dataset_name == "cifar100":
        dataset = CIFAR100(
            root="/tmp",
            transform=preprocess,
            train=False,
            download=True
        )

    if args.dataset_name == "imagenet":
        dataset = ImageNet(
            root="/work/alef.ferreira/clip_multimodal/clip_multimodal_orig/evaluate/data",
            transform=preprocess,
            split='val'
        )

    # Clip Original
    # print("CLIP Original")
    # targets_clip_orig, preds_clip_orig = get_dataset_acc(
    #     image_dataset=dataset,
    #     clip_processor=clip_processor,
    #     model_clip=model_clip,
    # )

    # Clip Original
    print("Trained Multimodal CLIP")
    targets_multi_clip, preds_multi_clip = get_dataset_acc(
        image_dataset=dataset,
        clip_processor=clip_processor,
        model_clip=pl_model.model.image_encoder
    )

    # save_conf_matrix(targets_clip_orig, preds_clip_orig, dataset.classes, "cifar10_clip.png")
    save_conf_matrix(targets_multi_clip, preds_multi_clip, dataset.classes, "cifar10_resnet_batch_8_acc_4_last.png")


if __name__ == "__main__":
    main()