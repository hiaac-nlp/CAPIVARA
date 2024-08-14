import argparse
import sys

import torch
import torch.nn.functional as F
import tqdm
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../")

from models.open_CLIP import OpenCLIP
from models.open_CLIP_adapter import OpenCLIPAdapter
from models.open_clip_wrapper import OpenCLIPWrapper
from utils.capivara_utils import download_pretrained_from_hf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, help="Path to validation/test dataset")
    parser.add_argument("--translation", choices=["english", "marian", "google"], required=False)
    parser.add_argument("--language", default="pt", choices=["pt", "xh", "hi"], required=False)
    parser.add_argument("--batch", type=int, help="Batch size", )
    parser.add_argument(
        "--open-clip",
        type=str,
        default="False",
        required=False,
        help="Indicates whether model is fine-tuned (True) or is the original OpenCLIP (False)"
    )
    parser.add_argument("--gpu", help="GPU")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        required=False,
        help="Load the adapter weights"
    )
    # Optional argument of local-model-path
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        required=False,
        help="Path to the local model"
    )

    return parser.parse_args()


def tokenize(example, args):
    image_input = vision_processor(example[0])

    captions = None
    if args.translation.lower() == "english":
        captions = example[1]["captions-en"]
    else:
        lang = args.language.lower()
        if len(example[1][f"captions-{lang}"]) == 1:
            captions = example[1][f"captions-{lang}"]
        else:
            if lang == 'pt':  # pt has 10 captions, 5 by google and 5 by marian
                if args.translation == "google":
                    captions = example[1][f"captions-{lang}"][1::2]
                elif args.translation == "marian":
                    captions = example[1][f"captions-{lang}"][0::2]
            else:
                captions = example[1][f"captions-{lang}"]

    return image_input, captions

def format_batch(batch):
    image_input = batch[0]
    return image_input, batch[1]


def evaluate(model, dataloader, tokenizer, device, recall_k_list=[1, 5, 10]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------

    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use

    Returns
    -------

    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)

    model.to(device)
    model.eval()

    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)

        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad():
            batch_images_emb = F.normalize(model.encode_visual(batch_images), dim=-1)
            batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (
                    batchify(recall_at_k, scores, positive_pairs, batch_size, device,
                             k=recall_k) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
                    batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device,
                             k=recall_k) > 0).float().mean().item()

    return metrics


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

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
    if args.open_clip == 'True':
        if args.adapter is None:
            if args.local_model_path is not None and os.path.exists(args.local_model_path):
                print("Loading local model... ")
                model = OpenCLIPWrapper.load_from_checkpoint(args.local_model_path, strict=False).model
            else:
                model_path = download_pretrained_from_hf(model_id="hiaac-nlp/CAPIVARA")
                model = OpenCLIPWrapper.load_from_checkpoint(model_path, strict=False).model
        else:
            model = OpenCLIPAdapter(inference=True, devices=device)
            model.load_adapters(pretrained_adapter=True, model_path=args.adapter)
    else:
        print('Using Baseline Model')
        model = OpenCLIP()

    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer
    metrics = evaluate(model=model, dataloader=dataloader, tokenizer=text_tokenizer, device=device)
    print(metrics)
