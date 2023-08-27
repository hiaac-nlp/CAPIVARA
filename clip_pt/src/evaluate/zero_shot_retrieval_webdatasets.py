import argparse
import os
import sys

sys.path.append("./")
sys.path.append("../")

import torch
import tqdm
import webdataset as wds
from torch.utils.data import DataLoader

from models.open_CLIP import OpenCLIP
from models.open_CLIP_adapter import OpenCLIPAdapter
from models.open_clip_wrapper import OpenCLIPWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", )
    parser.add_argument("--distill", default=None, type=str, help="From knowledge distillation", )
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--translation", choices=["english", "marian", "google"], required=False)
    parser.add_argument("--batch", type=int, help="Batch size", )
    parser.add_argument("--open-clip", type=bool, default=False, required=False,
                        help="Indicates whether model is fine-tuned (True) or is the original OpenCLIP (False)")
    parser.add_argument("--gpu", help="GPU", )
    parser.add_argument("--adapter", default=None, required=False, help="Load the adapter weights")

    return parser.parse_args()


def tokenize(example, args):
    image_input = vision_processor(example[0])

    captions = None
    if args.translation.lower() == "english":
        captions = example[1]["captions-en"]
    else:
        if len(example[1]["captions-pt"]) == 1:
            captions = example[1]["captions-pt"][0]
        else:
            if args.translation == "marian":
                captions = example[1]["captions-pt"][1::2]
            elif args.translation == "google":
                captions = example[1]["captions-pt"][0::2]

    text_input = text_tokenizer(captions)

    return image_input, text_input


def format_batch(batch):
    image_input = batch[0]
    text_input = batch[1].reshape((-1, 77))
    return image_input, text_input


def feature_extraction(model, dataloader, device):
    image_features = []
    text_features = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            image_input, text_input = batch
            image_input = image_input.to(device)
            text_input = text_input.to(device)
            batch = image_input, text_input

            img_features, txt_features = model(batch)

            norm_img_features = img_features / img_features.norm(dim=1, keepdim=True)
            norm_txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)
            image_features.append(norm_img_features)
            text_features.append(norm_txt_features)

    return image_features, text_features


def text_to_image_retrieval(image_features, text_features):
    top_k = 10
    top_k_predictions = []
    for text_feature in tqdm.tqdm(text_features, desc="t2i retrieval"):
        similarities = []
        for image_feature in image_features:
            scores = text_feature @ image_feature.t()  # shape: [batch_size, batch_size]
            similarities.append(scores)

        similarities = torch.cat(similarities, dim=1)  # shape: [batch_size, #images]
        top_k_pred = torch.argsort(similarities, descending=True)[:,
                     : top_k].cpu()  # shape: [batch_size, top_k]
        top_k_predictions.append(top_k_pred)
        # free memory
        del similarities

    top_k_predictions = torch.cat(top_k_predictions, dim=0)  # shape: [#texts, top_k]
    return top_k_predictions


def image_to_text_retrieval(image_features, text_features):
    top_k = 10
    top_k_predictions = []
    for image_feature in tqdm.tqdm(image_features, desc="i2t retrieval"):
        similarities = []
        for text_feature in text_features:
            scores = image_feature @ text_feature.t()  # shape: [batch_size, batch_size]
            similarities.append(scores)

        similarities = torch.cat(similarities, dim=1)  # shape: [batch_size, #texts]
        top_k_pred = torch.argsort(similarities, descending=True)[:,
                     : top_k].cpu()  # shape: [batch_size, top_k]
        top_k_predictions.append(top_k_pred)
        # free memory
        del similarities

    top_k_predictions = torch.cat(top_k_predictions, dim=0)  # shape: [#texts, top_k]
    return top_k_predictions


def compute_recall_k(predictions, ground_truth, k, n_relevants):
    top_k_preds = predictions[:, :k]
    prev_corrects = None
    for col in range(ground_truth.shape[1]):
        gt = ground_truth[:, col].reshape((-1, 1))
        corrects = (top_k_preds == gt).any(dim=1)
        if prev_corrects is None:
            prev_corrects = corrects
        else:
            new_corrects = torch.logical_or(prev_corrects, corrects)
            prev_corrects = corrects
            corrects = new_corrects

    return sum(corrects) / n_relevants


def get_txt2img_ground_truth(args):
    # Extract the dataset name from the path in 'args' and construct the file path for the ground
    # truth file.
    dataset_name = os.path.basename(os.path.dirname(args.dataset_path))
    filepath = f"evaluate/ground_truth/text-to-image/{dataset_name}_{args.translation}.pt"

    # Check if the ground truth file already exists, if  so, load it from disk
    if os.path.isfile(filepath):
        gt = torch.load(filepath)
        n_relevants = gt.shape[0]

        return gt, n_relevants

    if args.model_path == "mCLIP":
        dataset = wds.WebDataset(args.dataset_path) \
            .decode("pil") \
            .to_tuple("jpg;png", "json")
    else:
        dataset = wds.WebDataset(args.dataset_path) \
            .decode("torchrgb") \
            .to_tuple("jpg;png", "json")
    gt = []
    n_relevants = 0
    for i, example in tqdm.tqdm(enumerate(dataset), desc="Computing ground truth"):
        n_captions = len(example[1]["captions-pt"])
        if n_captions > 1:
            gt += [i] * (n_captions // 2)  # two translations per caption
            n_relevants += (n_captions // 2)
        else:
            gt.append(i)
            n_relevants += 1

    gt = torch.Tensor(gt).reshape(-1, 1)

    os.makedirs("evaluate/ground_truth/text-to-image", exist_ok=True)
    torch.save(gt, filepath)
    return gt, n_relevants


if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    dataset = wds.WebDataset(args.dataset_path) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, args=args)) \
        .batched(args.batch) \
        .map(lambda x: format_batch(x))

    dataloader = DataLoader(dataset, batch_size=None, num_workers=10)

    print(">>>>>>> Loading model")
    if args.open_clip:
        if args.adapter is None:
            model = OpenCLIPWrapper.load_from_checkpoint(args.model_path, strict=False).model
        else:
            model = OpenCLIPAdapter(inference=True, devices=device)
            model.load_adapters(pretrained_adapter=args.adapter)
    else:
        model = OpenCLIP()

    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    print(">>>>>>> Extracting features")
    image_features, text_features = feature_extraction(model, dataloader, device)

    # free memory
    del model

    print(">>>>>>> Text-to-Image retrieval features")
    top_k_predictions = text_to_image_retrieval(image_features, text_features)
    ground_truth, n_relevants = get_txt2img_ground_truth(args)

    print(">>>>>>> Computing Recall")
    recall_1 = compute_recall_k(top_k_predictions, ground_truth, k=1, n_relevants=n_relevants)
    recall_5 = compute_recall_k(top_k_predictions, ground_truth, k=5, n_relevants=n_relevants)
    recall_10 = compute_recall_k(top_k_predictions, ground_truth, k=10, n_relevants=n_relevants)
    mr = (recall_1 + recall_5 + recall_10) / 3

    print("Recall@1: ", recall_1.item())
    print("Recall@5: ", recall_5.item())
    print("Recall@10: ", recall_10.item())
    print("Mean Recall: ", mr.item())
