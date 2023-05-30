import argparse
import sys
from pathlib import Path

import torch
import webdataset as wds
from torch.utils.data import DataLoader

from models.open_CLIP import OpenCLIP
from models.open_clip_wrapper import OpenCLIPWrapper

sys.path.append("./")
sys.path.append("../")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", default="OpenCLIP")
    parser.add_argument("--dataset-path", help="Path to dataset")
    parser.add_argument("--translation", choices=["marian", "google"], required=False)
    parser.add_argument("--batch", type=int, help="Batch size", )
    parser.add_argument("--gpu", help="GPU", )

    return parser.parse_args()


def tokenize(example, args):
    captions_en = example[1]["captions-en"][:5]
    captions_pt = None
    if len(example[1]["captions-pt"]) == 1:
        captions_pt = example[1]["captions-pt"][0]
    else:
        n = len(example[1]["captions-en"])
        if args.translation == "marian":
            captions_pt = example[1]["captions-pt"][:5]
        elif args.translation == "google":
            captions_pt = example[1]["captions-pt"][n: n + 5]

    text_pt_input = text_tokenizer(captions_pt)
    text_en_input = text_tokenizer(captions_en)
    image_input = vision_processor(example[0])

    return image_input, text_pt_input, text_en_input, example


def format_batch(batch):
    image_input, text_pt_input, text_en_input, example = batch
    text_pt_input = text_pt_input.reshape((-1, 77))
    text_en_input = text_en_input.reshape((-1, 77))
    return image_input, text_pt_input, text_en_input, example


def feature_extraction(batch, device):
    image_input, text_pt_input, text_en_input, example = batch

    image_input = image_input.to(device)
    text_pt_input = text_pt_input.to(device)
    text_en_input = text_en_input.to(device)

    image_features = model.encode_visual(image_input)
    text_pt_features = model.encode_text(text_pt_input)
    text_en_features = model.encode_text(text_en_input)

    norm_img_features = image_features / image_features.norm(dim=1, keepdim=True)
    norm_txt_pt_features = text_pt_features / text_pt_features.norm(dim=1, keepdim=True)
    norm_txt_en_features = text_en_features / text_en_features.norm(dim=1, keepdim=True)

    return norm_img_features, norm_txt_pt_features, norm_txt_en_features, example


if __name__ == "__main__":
    args = parse_args()
    print(args)

    dataset = wds.WebDataset(args.dataset_path) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, args=args)) \
        .batched(args.batch) \
        .map(lambda x: format_batch(x))
    dataloader = DataLoader(dataset, batch_size=None, num_workers=10)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print(">>>>>>> Loading model")
    if args.model_path == "OpenCLIP":
        model = OpenCLIP()
    else:
        model = OpenCLIPWrapper.load_from_checkpoint(args.model_path, strict=False).model

    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    index = 0

    path = Path(args.dataset_path)
    parent_dir = path.parent
    dir_path = parent_dir.with_name(parent_dir.name + "_filtered") / "%05d.tar"
    sink = wds.ShardWriter(str(dir_path), maxcount=10000)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            image_features, text_pt_features, text_en_features, examples = \
                                                                feature_extraction(batch, device)

            scores_pt = text_pt_features @ image_features.t()  # shape: [batch_size, batch_size]
            scores_en = text_en_features @ image_features.t()  # shape: [batch_size, batch_size]
            predictions_pt = torch.argmax(scores_pt, dim=1)    # shape: [batch_size, 1]
            predictions_en = torch.argmax(scores_en, dim=1)    # shape: [batch_size, 1]

            ground_truth = torch.arange(scores_pt.shape[1])
            n = scores_pt.shape[0] // scores_pt.shape[1]  # number of captions per image
            ground_truth = ground_truth.repeat_interleave(n)

            for pred_pt, pred_en, label, example in zip(predictions_pt, predictions_en, ground_truth, examples):
                # select only the examples that the model mis-retrieved from Portuguese caption
                if pred_pt != label:
                    sample = {
                        "__key__": "sample%05d" % index,
                        "png": example[0],
                        "json": example[1],
                    }
                    sink.write(sample)
                    index += 1
