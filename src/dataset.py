import itertools
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from pathlib import Path

from src.constants import CODE2INT
INT2CODE = {v: k for k, v in CODE2INT.items()}


class TorchAudioDataSet(Dataset):
    def __init__(
        self, filenames, durations, labels=None, period=7, num_periods=1, sr=32000, waveform_transforms=None,
    ):
        self.filenames = filenames
        self.durations = durations
        self.labels = labels
        self.period = period
        self.num_periods = num_periods
        self.sr = sr
        self.waveform_transforms = waveform_transforms
        self.effective_length = int(self.sr * self.period * self.num_periods)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        wav_name = self.filenames[idx]
        duration = self.durations[idx]

        # if duration is greater take a random window, else pad it
        if duration > self.effective_length:
            start = random.randint(0, duration - self.effective_length)
            y, _ = sf.read(wav_name, frames=self.effective_length, start=start, always_2d=True, dtype="float32")
            y = np.mean(y, axis=1)
        else:
            y, _ = sf.read(
                wav_name, frames=self.effective_length, start=0, fill_value=0, always_2d=True, dtype="float32"
            )
            y = np.mean(y, axis=1)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        # if len(y) == 0:
        #     y = np.zeros(shape=(self.effective_length,))

        # if len(y) > self.effective_length:
        #     start = np.random.randint(low=0, high=(len(y) - effective_length) + 1)
        #     y = y[start:start+effective_length]

        # elif len(y) < effective_length:
        #     y = np.pad(y, (0, effective_length-len(y)), 'wrap')

        if self.labels is not None:
            labels = np.zeros(len(CODE2INT), dtype="f")

            ebird_codes = self.labels[idx]
            for ecode in ebird_codes:
                if ecode in CODE2INT:
                    labels[CODE2INT[ecode]] = 1
            if sum(labels) == 0:
                labels[CODE2INT["nocall"]] = 1
        else:
            labels = []
        return y.reshape(self.num_periods, -1).astype(np.float32), labels


class TorchAudioValDataSet(Dataset):
    def __init__(
        self, filenames, end_seconds, labels, period=5, sr=32000, waveform_transforms=None, spectrogram_transforms=None,
    ):
        self.filenames = filenames
        self.end_seconds = end_seconds
        self.labels = labels
        self.period = period
        self.sr = sr
        self.effective_length = int(self.period * self.sr)
        self.waveform_transforms = waveform_transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        wav_name = self.filenames[idx]
        start = int((self.end_seconds[idx] - self.period) * self.sr)
        y, _ = sf.read(wav_name, frames=self.effective_length, start=start, fill_value=0, always_2d=True)
        y = np.mean(y, axis=1)

        # if len(y) != effective_length:
        #     y = np.zeros(effective_length, dtype=np.float32)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        if self.labels is not None:
            labels = np.zeros(len(CODE2INT), dtype="f")

            ebird_codes = self.labels[idx]
            for ecode in ebird_codes:
                if ecode in CODE2INT:
                    labels[CODE2INT[ecode]] = 1
            if sum(labels) == 0:
                labels[CODE2INT["nocall"]] = 1

        return y.reshape(1, -1).astype(np.float32), labels


class MelspecDataset(Dataset):
    def __init__(
        self,
        filepaths,
        labels,
        secondary_labels,
        config,
        transforms=None,
        random_power=True,
        add_bad=True,
        add_noise=True,
    ):
        # Initialize the list of melspectrograms | Инициализировать список мелспектрограмм
        self.filepaths = filepaths
        self.labels = labels
        self.secondary_labels = secondary_labels
        self.transforms = transforms
        self.config = config
        self.noise = pd.read_csv("data/nocall.csv")
        self.random_power = random_power
        self.add_bad = add_bad
        self.add_noise = add_noise
        self.mixing_prob = config.mixing_prob  # Probability of stopping mixing | Вероятность прервать смешивание
        self.noise_level = config.noise_level  # level noise | Уровень шума
        self.signal_amp = config.signal_amp  # signal amplification during mixing | Усиления сигнала при смешивании

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        idx2 = random.randint(0, len(self) - 1)  # Second file | Второй файл
        idx3 = random.randint(0, len(self) - 1)  # Third file | Третий файл
        # Streching
        self.width2 = random.randint(self.config.width - 60, self.config.width + 60)

        images = np.zeros((self.config.n_mels, self.width2)).astype(np.float32)
        labels = np.zeros(len(CODE2INT), dtype="f")
        # Loop over files and concatenate labels.
        for i, idy in enumerate([idx, idx2, idx3]):
            # Load numpy melspec
            fp = Path(self.filepaths[idy])
            fp = str(Path("data/melspecs") / fp.parent.stem / f"{fp.stem}.npy")
            mel = np.load(fp, allow_pickle=True)
            # pad select specific period
            # mel = self.pad_select_patch(mel)
            if mel.shape[1] > self.width2:
                start = random.randint(0, mel.shape[1] - self.width2 - 1)
                mel = mel[:, start: start + self.width2]
            else:
                len_zero = random.randint(0, self.width2.shape[1])
                mel = np.concatenate((np.zeros((self.config.n_mels, len_zero)), mel), axis=1)

            mel = np.concatenate((mel, np.zeros((self.config.n_mels, self.width2-mel.shape[1]))), axis=1)
            # Random power
            if self.random_power:
                mel = random_power(mel, power=3, c=0.5)

            # Combine with random amplification
            images += mel * (random.random() * self.signal_amp + 1)
            # Primary labels
            ebird_codes = self.labels[idy]
            for ecode in ebird_codes:
                if ecode in CODE2INT:
                    labels[CODE2INT[ecode]] = 1

            # Secondary labels with lower confidence.
            secondary_codes = self.secondary_labels[idy]
            if len(secondary_codes) > 0:
                for ecode in secondary_codes:
                    if ecode in CODE2INT:
                        labels[CODE2INT[ecode]] = self.config.secondary_confidence

            if random.random() < self.mixing_prob:
                break

        if sum(labels) == 0:
            labels[CODE2INT["nocall"]] = 1

        # Add background noise from BAD dataset
        if self.add_bad:
            idy = random.randint(0, len(self.noise) - 1)
            noise_sample = self.noise.numpyfilepaths[idy]
            noise_mel = np.load(noise_sample, allow_pickle=True)[:, 0:self.width2]  # Generate noise
            noise_mel = random_power(noise_mel)

            images += noise_mel / (noise_mel.max() + 0.0000001) * (random.random() * 1 + 0.5) * images.max()
        # Convert to decibels and normalize
        images = librosa.power_to_db(images.astype(np.float32), ref=np.max)
        images = (images + 80) / 80

        # Add white, pink noise
        if self.add_noise:
            if random.random() < 0.9:
                images = images + (
                    np.random.sample((self.config.n_mels, self.width2)).astype(np.float32) + 9
                ) * images.mean() * self.noise_level * (np.random.sample() + 0.3)

            # Add pink noise
            if random.random() < 0.9:
                r = random.randint(1, self.config.n_mels)
                pink_noise = np.array([np.concatenate((1 - np.arange(r) / r, np.zeros(self.config.n_mels - r)))]).T
                images = images + (
                    np.random.sample((self.config.n_mels, self.width2)).astype(np.float32) + 9
                ) * 2 * images.mean() * self.noise_level * (np.random.sample() + 0.3)

            # Add bandpass noise
            if random.random() < 0.9:
                a = random.randint(0, self.config.n_mels // 2)
                b = random.randint(a + 20, self.config.n_mels)
                images[a:b, :] = images[a:b, :] + (
                    np.random.sample((b - a, self.width2)).astype(np.float32) + 9
                ) * 0.05 * images.mean() * self.noise_level * (np.random.sample() + 0.3)

            # Lower the upper frequencies
            if random.random() < 0.5:
                images = images - images.min()
                r = random.randint(self.config.n_mels // 2, self.config.n_mels)
                x = random.random() / 2
                pink_noise = np.array(
                    [np.concatenate((1 - np.arange(r) * x / r, np.zeros(self.config.n_mels - r) - x + 1))]
                ).T
                images = images * pink_noise
                images = images / images.max()

            # Change the contrast
            images = random_power(images, power=2, c=0.7)

        # Expand to 3 channels
        images = mono_to_color(images, self.config.n_mels, self.config.width)
        # Draw pictures | Рисуем графики
        # if random.random() < 0.0001:
        #     img = images.numpy()
        #     img = img - img.min()
        #     img = img / img.max()
        #     img = np.moveaxis(img, 0, 2)
        #     _ = plt.imshow(img)
        #     codes = np.where(labels == 1)[0]
        #     codestr = [INT2CODE[i] for i in codes]
        #     plt.savefig("log/img/" + ("_".join(codestr)) + "_" + Path(self.filepaths[idx]).stem + ".png")

        return {"x": images, "labels": labels}


def mono_to_color(X, height=64, width=500, mean=0.5, std=0.5, eps=1e-6):
    trans = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([mean, mean, mean], [std, std, std]),
        ]
    )
    X = np.stack([X, X, X], axis=-1)
    V = (255 * X).astype(np.uint8)
    V = (trans(V) + 1) / 2
    return V


def random_power(images, power=1.5, c=0.7):
    images = images - images.min()
    images = images/(images.max()+0.0000001)
    images = images**(random.random()*power + c)
    return images
