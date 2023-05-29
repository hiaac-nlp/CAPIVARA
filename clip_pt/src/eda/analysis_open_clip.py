import argparse
import os
import sys

import torch
import tqdm
import webdataset as wds
from torch.utils.data import DataLoader

from models.open_CLIP import OpenCLIP
from models.open_clip_wrapper import OpenCLIPWrapper

sys.path.append("./")
sys.path.append("../")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", )
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--translation", choices=["marian", "google"], required=False)
    parser.add_argument("--batch", type=int, help="Batch size", )
    parser.add_argument("--gpu", help="GPU", )

    return parser.parse_args()


def tokenize(example, lang, args):
    captions = None
    if lang.lower() == 'en':
        captions = example[1]["captions-en"]
    else:
        if len(example[1]["captions-pt"]) == 1:
            captions = example[1]["captions-pt"][0]
        else:
            if args.translation == "marian":
                captions = example[1]["captions-pt"][:5]
            elif args.translation == "google":
                captions = example[1]["captions-pt"][5:]

    text_input = text_tokenizer(captions)
    image_input = vision_processor(example[0])

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


def text_to_image_retrieval(image_features, text_features, top_k=10):
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

    dataset = wds.WebDataset(args.dataset_path) \
        .decode("pil") \
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


def compute_stats(preds_pt, preds_en, ground_truth, percent=True):
    top_1_pred_pt = preds_pt[:, :1]
    top_1_pred_en = preds_en[:, :1]

    total = ground_truth.shape[0]
    count_hits = {"pt-en": 0,  # model predicted correctly labels for PT and EN captions
                  "pt": 0,  # model predicted correctly labels for PT but not for EN captions
                  "en": 0,  # model predicted correctly labels for EN but not for PT captions
                  "none": 0}  # model didn't predict correctly any label

    for pred_en, pred_pt, label in zip(top_1_pred_pt, top_1_pred_en, ground_truth):
        if pred_pt == pred_en == label:
            count_hits["pt-en"] += 1
        elif pred_pt == label:
            count_hits["pt"] += 1
        elif pred_en == label:
            count_hits["en"] += 1
        else:
            count_hits["none"] += 1

    if percent:
        count_hits["pt-en"] /= total
        count_hits["pt"] /= total
        count_hits["en"] /= total
        count_hits["none"] /= total

    return count_hits


def predict(lang):
    dataset = wds.WebDataset(args.dataset_path) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, lang=lang, args=args)) \
        .batched(args.batch) \
        .map(lambda x: format_batch(x))
    dataloader = DataLoader(dataset, batch_size=None, num_workers=10)

    print(">>>>>>> Extracting features")
    image_features, text_features = feature_extraction(model, dataloader, device)

    print(">>>>>>> Text-to-Image retrieval features")
    return text_to_image_retrieval(image_features, text_features)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print(">>>>>>> Loading model")
    if args.open_clip:
        model = OpenCLIPWrapper.load_from_checkpoint(args.model_path, strict=False).model
    else:
        model = OpenCLIP()

    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    preds_pt = predict(lang="pt")
    preds_en = predict(lang="en")
    ground_truth, _ = get_txt2img_ground_truth(args)

    stats = compute_stats(preds_pt, preds_en, ground_truth)
    print(stats)




