import sys
# add previous and current path
sys.path.append('./')
sys.path.append('../')

import os
import json
import argparse

import clip
import torch
import torchaudio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor

from utils.model import Clip_Multimodal
from utils.clip_multimodal_wrapper import ClipMultimodalWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_sampling_rate: int,
        base_wav_path: str,
        target_length: int,
        norm_mean: int,
        norm_std: int,
        melbins: int
    ):
        self.data = df
        self.base_wav_path = base_wav_path
        self.target_sampling_rate = target_sampling_rate
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.melbins = melbins


    def __len__(self):
        return len(self.data)


    def _load_wav(self, filename):
        waveform, source_sr = torchaudio.load(os.path.join(self.base_wav_path, filename))

        # Convert to mono channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Convert source sampling rate to a target sampling rate
        if source_sr != self.target_sampling_rate:
            transform = torchaudio.transforms.Resample(source_sr, self.target_sampling_rate)
            waveform = transform(waveform)

        return waveform, self.target_sampling_rate


    def _wav2fbank(self, filename):
        waveform, sr = self._load_wav(filename=filename)
        waveform = waveform - waveform.mean()

        if self.melbins > 100:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10
            )
        else:
            # Shape: (num_bins, 400), source: https://github.com/lijuncheng16/AudioTaggingDoneRight
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=4096,
                win_length=1024,
                hop_length=400,
                center=True,
                pad_mode="constant",
                power=2,
                norm='slaney',
                onesided=True,
                n_mels=self.melbins,
                mel_scale="slaney"
            )
            spec = melspec(waveform).squeeze()
            a2db = torchaudio.transforms.AmplitudeToDB(spec)
            fbank = a2db(spec)
            fbank = fbank.transpose(0,1)


        n_frames = fbank.shape[0]
        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]

        return fbank


    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data.iloc[index]
        fbank = self._wav2fbank(datum["file_name"])

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]

        return fbank


class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def text_encoder(text, clip_processor, model_clip):
    text_data = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    with torch.no_grad():
        text_features = model_clip.get_text_features(
            input_ids=text_data['input_ids'],
            attention_mask=text_data['attention_mask']
        )
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def precompute_text_features(loader, clip_processor, model_clip):
    text_features_arr = []

    for texts in tqdm(loader):
        text_data = clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_data.to(device)
        with torch.no_grad():
            text_features = model_clip.model_clip.get_text_features(
                input_ids=text_data['input_ids'],
                attention_mask=text_data['attention_mask']
            )
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_features_arr.extend(text_features.cpu().detach().numpy())

    return np.array(text_features_arr)


def precompute_audio_features(loader, model_clip):
    audio_features_arr = []
    for audios in tqdm(loader):
        audios = audios[:,None,:].to(device)
        with torch.no_grad():
            audio_features = model_clip.audio_encoder(audios)
        audio_features /= audio_features.norm(dim=-1, keepdim=True)

        audio_features_arr.extend(audio_features.cpu().detach().numpy())

    return np.array(audio_features_arr)


def get_path_audio(index, df):
    return df.iloc[index]["file_name"]


def compute_mrr(data, dataset, n, audio_features, text_features, df):
    collect_rr = []

    pbar = tqdm(total=len(data), position=0, leave=True)

    found = np.matmul(audio_features, text_features.T)
    for index, distances in enumerate(found):
        # print(distances, index, np.max(distances), np.argmax(distances))
        # exit()
        pbar.update(1)
        audio_path = get_path_audio(index, df)
        collect_rr.append(new_rr(distances, audio_path, dataset, n, df))

    pbar.close()
    return np.average(collect_rr)


def new_rr(distances, target_audio, dataset, n, df):
    audio_paths = []
    idxs = distances.argsort()[-n:][::-1]

    for idx in idxs:
        audio_paths.append(get_path_audio(idx, df))

    if target_audio in audio_paths:
        return 1/(audio_paths.index(target_audio) + 1)
    else:
        return 0


def internal_hits(distances, target_audio, dataset, n, df):
    audio_paths = []
    idxs = distances.argsort()[-n:][::-1]
    for idx in idxs:
        audio_paths.append(get_path_audio(idx, df))

    if target_audio in audio_paths:
        return 1
    else:
        return 0


def compute_hits(data, dataset, n, text_features, audio_features, df):
    collect_rr = []

    pbar = tqdm(total=len(data), position=0, leave=True)

    found = np.matmul(text_features, audio_features.T)
    for index, distances in enumerate(found):
        pbar.update(1)
        audio_path = get_path_audio(index, df)
        collect_rr.append(internal_hits(distances, audio_path, dataset, n, df))

    pbar.close()
    return np.average(collect_rr)


def get_mrr(audio_loader, text_loader, model_clip, clip_processor, df):
    audio_features = precompute_audio_features(
        audio_loader,
        model_clip
    )

    text_features = precompute_text_features(
        loader=text_loader,
        clip_processor=clip_processor,
        model_clip=model_clip
    )

    print('MRR@1:', compute_mrr(df['caption_1'].values.tolist(), df["file_name"].values.tolist(), 1, audio_features, text_features, df))
    print('MRR@5:', compute_mrr(df['caption_1'].values.tolist(), df["file_name"].values.tolist(), 5, audio_features, text_features, df))
    print('MRR@10:', compute_mrr(df['caption_1'].values.tolist(), df["file_name"].values.tolist(), 10, audio_features, text_features, df))

    print(compute_hits(df['caption_1'].values.tolist(), df["file_name"].values.tolist(), 100, text_features, audio_features, df))


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


    df = pd.read_csv("evaluate/data/clotho_captions_evaluation.csv")
    df = df[["file_name", "caption_1"]]

    audio_dataset = AudioDataset(
        df=df,
        target_sampling_rate=44100,
        base_wav_path="evaluate/data/clothov2_evaluation",
        target_length=400,
        norm_mean=-4.2677393,
        norm_std=4.5689974,
        melbins=64
    )

    audio_loader = DataLoader(
        dataset=audio_dataset,
        batch_size=24,
        shuffle=False,
        drop_last=False
    )

    text_dataset = SimpleTextDataset(df['caption_1'].values.tolist())

    text_loader = DataLoader(
        dataset=text_dataset,
        batch_size=24,
        shuffle=False,
        drop_last=False
    )

    get_mrr(
        audio_loader=audio_loader,
        text_loader=text_loader,
        clip_processor=clip_processor,
        model_clip=pl_model.model,
        df=df
    )



if __name__ == "__main__":
    main()