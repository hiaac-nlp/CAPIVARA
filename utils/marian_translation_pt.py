import argparse


import tqdm
import torch
import pandas as pd
import webdataset as wds
import torchvision.transforms as T

from transformers import MarianMTModel, MarianTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--dataset_path",
        type=str,
        help="Path to the directory of the dataset in the format: e.g: http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=10,
        type=int
    )
    parser.add_argument(
        "-n_w",
        "--num_workers",
        default=8,
        type=int
    )
    parser.add_argument(
        "-d_n",
        "--dataset_name",
        default="wit",
        type=str
    )
    parser.add_argument(
        "-m_t",
        "--max_tokens",
        default="600",
        type=str
    )
    args = parser.parse_args()

    dataset = (
        wds.WebDataset(args.dataset_path, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;png", "json")
        .shuffle(False)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset.batched(args.batch_size),
        num_workers=args.num_workers,
        batch_size=None,
        shuffle=False
    )

    list_of_sentences = []
    keys_list = []
    original_captions_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
    model = model.to(device)
    print(torch.cuda.is_available())
    epochs = 0

    for chunk in tqdm.tqdm(dataloader):
        if epochs > 3:
            break
        _, metadata = chunk
        print(metadata)
        original_captions = [metadata[i]["caption"] for i in range(len(metadata))]
        keys = [metadata[i]["key"] for i in range(len(metadata))]

        inputs = tokenizer(['>>por<<' + sentence for sentence in original_captions], return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,  # disable sampling to test if batching affects output
            max_new_tokens=int(args.max_tokens)
        )

        sentence_decoded = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        keys_list.extend(keys)
        original_captions_list.extend(original_captions)
        list_of_sentences.extend(sentence_decoded)

        epochs+=1
        
        if epochs % 10 == 0:
            d_name = [args.dataset_name for _ in range(len(keys_list))]
            df = pd.DataFrame(list(zip(keys_list, original_captions_list, list_of_sentences, d_name)), columns =['key', 'original_sentence', 'translated_sentence', 'dataset_name'])
            df.to_csv("metadata.csv", index=False)
            keys_list = []
            original_captionkeys_list = []
            list_of_sentences = []

if __name__ == "__main__":
    main()
