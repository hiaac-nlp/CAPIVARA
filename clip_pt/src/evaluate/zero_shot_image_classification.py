import argparse

import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPFeatureExtractor, AutoTokenizer

from models.mCLIP import mCLIP
from utils.dataset.grocery_store_dataset import GroceryStoreDataset
from utils.dataset.object_net import ObjectNetDataset
from models.clip_pt_br_wrapper import CLIPPTBRWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model checkpoint", )
    parser.add_argument("--dataset", help="Dataset name", )
    parser.add_argument("--dataset-path", help="Path to validation/test dataset")
    parser.add_argument("--translation-path", help="Path to translated labels")
    parser.add_argument("--batch", type=int, help="Batch size", )
    parser.add_argument("--gpu", help="GPU", )

    return parser.parse_args()


def topk_accuracy(logits, targets, topk=(1, 5, 10)):
    predictions = torch.argsort(logits, descending=True)
    results = {}
    for k in topk:
        # Get the index of the top-k predicted probabilities
        predicted_labels = predictions[:, :k]

        # Check if the target label is in the top-k predictions for each sample
        correct_predictions = (predicted_labels == targets.view(-1, 1)).any(dim=1)

        # Compute the top-k accuracy
        accuracy_k = correct_predictions.sum().item() / targets.size(0)
        results[f"acc@{k}"] = accuracy_k

    return results


if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print(">>>>>>> Loading model")
    if args.model_path == "mCLIP":
        text_tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32",
                                                       cache_dir="/hahomes/gabriel.santos/")
        model = mCLIP(device=device)
        vision_processor = model.image_preprocessor
    else:
        vision_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32",
                                                                cache_dir="/hahomes/gabriel.santos/")
        text_tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased",
                                                       do_lower_case=False,
                                                       cache_dir="/hahomes/gabriel.santos/")
        model = CLIPPTBRWrapper.load_from_checkpoint(args.model_path)

    print(">>>>>>> Loading dataset")
    if args.dataset.lower() == 'objectnet':
        dataset = ObjectNetDataset(root_dir=args.dataset_path,
                                   translation_path=args.translation_path,
                                   vision_processor=vision_processor,
                                   text_tokenizer=text_tokenizer)
    elif args.dataset.lower() == 'grocerystore':
        dataset = GroceryStoreDataset(dataset_path=args.dataset_path,
                                      annotation_path=args.translation_path,
                                      vision_processor=vision_processor,
                                      text_tokenizer=text_tokenizer)
    else:
        raise NotImplementedError(f"{args.dataset} is not a supported dataset.")

    dataloader = DataLoader(dataset, batch_size=args.batch, num_workers=10)

    model.to(device)
    model.eval()
    logits = []
    class_idx_list = []
    text_input = None
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            if text_input is None:
                text_input = dataset.get_labels()
                text_input["input_ids"] = text_input["input_ids"].to(device)
                text_input["attention_mask"] = text_input["attention_mask"].to(device)
                if "token_type_ids" in text_input:
                    text_input["token_type_ids"] = text_input["token_type_ids"].to(device)

            image_input, class_idx = batch
            image_input["pixel_values"] = image_input["pixel_values"].squeeze(1).to(device)
            batch = image_input, text_input
            img_features, txt_features = model.model(batch)
            logits_per_image, _ = model.model.compute_logits(img_features,
                                                             txt_features,
                                                             fixed_logit=False)  # shape: [n_imgs, n_classes]
            logits.append(logits_per_image)
            class_idx_list.append(class_idx)

    logits = torch.cat(logits, dim=0).cpu()
    targets = torch.cat(class_idx_list, dim=0).cpu()
    metrics = topk_accuracy(logits, targets)
    print(metrics)
    print("Avg. acc.:", sum(metrics.values())/len(metrics))

