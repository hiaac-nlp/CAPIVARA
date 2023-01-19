import os
import sys

# add previous and current path
sys.path.append('./')
sys.path.append('../')

import argparse


import clip
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet

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
        fbank = self._wav2fbank(datum["filename"])

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]

        return fbank, datum["target"]


def accuracy(output, target, topk=(1,)):
    output = torch.from_numpy(np.asarray(output))
    target = torch.from_numpy(np.asarray(target))
    pred = output.topk(max(topk), dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_dataset_acc(audio_dataset, clip_processor, model, labels):
    preds = []
    targets = []
    loader = torch.utils.data.DataLoader(
        audio_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    text_descriptions = [f"{label}" for label in labels]

    text_data = clip_processor(
        text=text_descriptions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    text_data.to(device)

    with torch.no_grad():
        text_features = model.image_encoder.get_text_features(
            input_ids=text_data['input_ids'],
            attention_mask=text_data['attention_mask']
        )
        text_features /= text_features.norm(dim=-1, keepdim=True)

    top_ns = [1, 5, 10]
    acc_counters = [0. for _ in top_ns]
    n = 0.
    model.eval()
    with torch.no_grad():
        for i, (audios, target) in enumerate(tqdm(loader)):
            # print(audios[:,None,:].shape)
            audios = audios[:,None,:].to(device)
            target = target.numpy()
            # predict
            audio_features = model.audio_encoder(audios)
            audio_features /= audio_features.norm(dim=-1, keepdim=True)
            # logits = audio_features.detach().cpu().numpy()  @ text_features.detach().cpu().numpy().T
            logits = audio_features  @ text_features.T
            preds.extend(torch.argmax(torch.softmax(logits, dim=1), dim=1).detach().cpu().numpy().tolist())
            targets.extend(target.tolist())
            # exit()

            # measure accuracy
        #     accs = accuracy(logits, target, topk=top_ns)
        #     for j in range(len(top_ns)):
        #         acc_counters[j] += accs[j]
        #     n += audios.shape[0]

        # tops = {f'top{top_ns[i]}': acc_counters[i] / n * 100 for i in range(len(top_ns))}

        # print(tops)
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(targets, preds)
    print(acc)

    return acc


def main():
    parser = argparse.ArgumentParser()
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

    folds = [1, 2, 3, 4, 5]
    df = pd.read_csv("evaluate/ESC-50-master/meta/esc50.csv")
    labels = [' '.join(category.split("_")) for category in df.sort_values("target")["category"].unique()]
    folds_acc = []
    for fold in folds:
        df_test = df[df['fold']==fold]

        fold_dataset = AudioDataset(
            df=df_test,
            target_sampling_rate=44100,
            base_wav_path="evaluate/ESC-50-master/audio",
            target_length=400,
            norm_mean=-4.2677393,
            norm_std=4.5689974,
            melbins=64
        )

        fold_acc = get_dataset_acc(
            audio_dataset=fold_dataset,
            clip_processor=clip_processor,
            model=pl_model.model,
            labels=labels
        )

        folds_acc.append(fold_acc)

    print(folds_acc)
    print(f"Avg Acc: {np.mean(folds_acc)}")



if __name__ == "__main__":
    main()