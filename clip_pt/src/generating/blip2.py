import argparse
import os
from pathlib import Path

import torch
import webdataset as wds
from torch.utils.data import DataLoader
import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", help="Path to dataset")
    parser.add_argument("--batch", type=int, help="Batch size", default=1000)
    parser.add_argument("--gpu", help="GPU", )
    parser.add_argument("--postfix-path", help="postfix", default="_blip2_augment")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_name = "Salesforce/blip2-opt-2.7b-coco"

    print(">>>>> Load dataset")
    dataset = wds.WebDataset(args.dataset_path) \
        .decode("pil") \
        .to_tuple("jpg;png", "json") \
        .batched(args.batch)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=10)

    path = Path(args.dataset_path)
    parent_dir = path.parent
    dir_path = parent_dir.with_name(parent_dir.name + args.postfix_path)
    os.makedirs(dir_path, exist_ok=True)
    dir_path = dir_path / "%05d.tar"

    sink = wds.ShardWriter(str(dir_path), maxcount=10000)

    print(">>>>> Load model")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    prompts = ["the foreground features", "a photo of", "a picture of",
               "this is a scene depicting", "an image of", "portrait of a",
               "this image captures a moment of", "a painting of", "an art of",
               "the picture shows"]

    n_prompts = len(prompts)
    index = 0
    for batch in tqdm.tqdm(dataloader, desc="Generating captions"):
        images, captions = batch

        prompts_repeated = prompts * len(images)
        images = [image for image in images for _ in range(n_prompts)]

        inputs = processor(images=images, text=prompts_repeated, return_tensors="pt",
                           padding=True).to(device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        x = []
        for i, (prompt, text) in enumerate(zip(prompts_repeated, generated_texts)):
            if i % n_prompts == 0:
                image = images.pop(0)
                caption = captions.pop(0)
                caption["generated-captions-en"] = []

            caption["generated-captions-en"].append(prompt + " " + text.strip())
            if i % n_prompts == n_prompts - 1:
                sample = {
                    "__key__": "sample%05d" % index,
                    "png": image,
                    "json": caption,
                }
                print(sample)
                sink.write(sample)
                index += 1
