import argparse
import sys
import textwrap

import matplotlib.pyplot as plt
import torch
import tqdm
import webdataset as wds
from torch.utils.data import DataLoader

from models.open_CLIP import OpenCLIP

sys.path.append("./")
sys.path.append("../")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--batch", type=int, help="Batch size", default=1000)
    parser.add_argument("--gpu", help="GPU", )
    parser.add_argument("--plot-figs", help="whether plot similarity distribution",
                        action="store_true")
    parser.add_argument("--lang", type=str, default="pt")
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--threshold", type=float, default=0.2)

    return parser.parse_args()


def tokenize(example, lang="pt"):
    if lang == "pt":
        caption = example[1]["captions-pt"][0]
    else:
        caption = example[1]["captions-en"][0]

    text_input = text_tokenizer(caption)
    image_input = vision_processor(example[0])

    return image_input, text_input, example


def format_batch(batch):
    image_input = batch[0]
    text_input = batch[1].reshape((-1, 77))
    return image_input, text_input, batch[2]


def compute_similarity(model, batch, device, plot_figs, threshold):
    image_input, text_input, examples = batch
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    batch = image_input, text_input

    img_features, txt_features = model(batch)

    norm_img_features = img_features / img_features.norm(dim=1, keepdim=True)
    norm_txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)

    sim = norm_txt_features @ norm_img_features.T
    if plot_figs:
        for s, example in zip(sim.diag(), examples):
            if s < threshold:
                caption_pt = example[1]['captions-pt'][0]
                code = hash(caption_pt)

                fig, ax = plt.subplots()
                ax.imshow(example[0])
                text = f"Portuguese: {textwrap.fill(caption_pt)}\n\n"
                if len(example[1]['captions-en']) > 0:
                    text += f"English: {textwrap.fill(example[1]['captions-en'][0])}\n\n"
                print(code, ":", text)
                print("-"*100)
                plt.text(0,0, text)
                ax.axis("off")
                plt.savefig(f"imgs/{code}_{s:.2f}.png", bbox_inches='tight')

    return sim.diag()  # similarity between corresponding texts and images


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print(args)

    model = OpenCLIP()
    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    dataset = wds.WebDataset(args.dataset_path) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .map(lambda x: tokenize(x, lang=args.lang)) \
        .batched(args.batch) \
        .map(lambda x: format_batch(x))

    dataloader = DataLoader(dataset, batch_size=None, num_workers=10)

    print(">>>>>>> Computing similarities")

    model.to(device)
    model.eval()
    similarities = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            similarities.append(compute_similarity(model, batch, device, args.plot_figs,
                                                   threshold=args.threshold))

    similarities = torch.concat(similarities).cpu().numpy()

    if not args.plot_figs:
        count = sum(similarities < float(args.threshold))
        print("Low sim:", count)

        print(similarities)
        plt.hist(similarities, density=True, bins=50)
        plt.ylabel('Probability')
        plt.xlabel('Similarity')
        plt.xlim(left=-0.1, right=0.5)
        plt.savefig(args.output_path)

