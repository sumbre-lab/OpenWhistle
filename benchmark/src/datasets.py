import json
import pandas as pd

from memoization import cached
import numpy as np
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
#from torchvggish import vggish_input

FFT_SIZE_IN_SECS = 0.05
HOP_LENGTH_IN_SECS = 0.01


@cached(thread_safe=False, max_size=100_000)
def _get_spectrogram(filename, max_duration, target_sample_rate, return_mfcc=False):
    try:
        waveform, sample_rate = torchaudio.load(filename)
    except RuntimeError as e:
        import librosa
        waveform, sample_rate = librosa.load(filename, sr=None)
        waveform = torch.tensor(waveform).unsqueeze(0)

    waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    if sample_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = transform(waveform)

    n_fft = int(FFT_SIZE_IN_SECS * target_sample_rate)
    hop_length = int(HOP_LENGTH_IN_SECS * target_sample_rate)

    if waveform.shape[1] < n_fft:
        waveform = F.pad(waveform, (0, n_fft - waveform.shape[1]))

    if return_mfcc:
        transform = torchaudio.transforms.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=20,
            melkwargs={
                'n_mels': 128,
                'n_fft': n_fft,
                'hop_length': hop_length,
                'power': 2.0})
    else:
        transform = torchaudio.transforms.MelSpectrogram(
            n_mels=128,
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0)
    spec = transform(waveform)

    frames_per_sec = int(1 / HOP_LENGTH_IN_SECS)
    max_frames = int(max_duration * frames_per_sec)

    spec = spec[0, :, :max_frames]
    if spec.shape[1] < max_frames:
        spec = F.pad(spec, (0, max_frames - spec.shape[1]))

    return spec

@cached(thread_safe=False, max_size=100_000)
def _get_waveform(filename, max_duration, target_sample_rate):
    try:
        waveform, sample_rate = torchaudio.load(filename)
    except RuntimeError as e:
        import librosa
        waveform, sample_rate = librosa.load(filename, sr=None)
        waveform = torch.tensor(waveform).unsqueeze(0)

    waveform = torch.mean(waveform, dim=0).unsqueeze(0)

    if sample_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = transform(waveform)

    max_samples = int(max_duration * target_sample_rate)
    waveform = waveform[0, :max_samples]
    if waveform.shape[0] < max_samples:
        waveform = F.pad(waveform, (0, max_samples - waveform.shape[0]))

    return waveform

@cached(thread_safe=False, max_size=100_000)
def _get_vggish_spectrogram(filename, max_duration, target_sample_rate=16_000):
    assert target_sample_rate == 16_000

    waveform = _get_waveform(filename, max_duration, target_sample_rate).numpy()
    spec = vggish_input.waveform_to_examples(waveform, target_sample_rate, return_tensor=True)
    return spec


@cached(thread_safe=False, max_size=100_000)
def _get_vggish_spectrogram_with_offset(filename, st, ed, max_duration, target_sample_rate=16_000):
    assert target_sample_rate == 16_000

    waveform = _get_waveform(filename, max_duration, target_sample_rate).numpy()
    waveform = waveform[st:ed]
    spec = vggish_input.waveform_to_examples(waveform, target_sample_rate, return_tensor=True)
    return spec


class ClassificationDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        num_labels,
        labels,
        unknown_label,
        sample_rate,
        max_duration,
        feature_type):

        super().__init__()

        label_to_id = {lbl: i for i, lbl in enumerate(labels)}
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.feature_type = feature_type

        df = pd.read_csv(metadata_path)

        df["path"] = df["path"].str.replace(
            "/lustre/fsn1/projects/rech/vzf/uqe97pu/raw_data/all_categories/",
            # "/Volumes/jz_scratch/raw_data/all_categories/",
            #"/home/rdessi/.cache/huggingface/hub/datasets--dolphinteam--DolphinReef-labeled/snapshots/cba875a4337ad22e834e1d2c54ef8f9cc1ee66b3/",  # gcp
            # "/Users/rdessi/.cache/huggingface/hub/datasets--dolphinteam--DolphinData-Unbalanced/snapshots/38f1d334042ff54b4877a6479b8ab759fa1263b9/",  # local
            "/users/zfne/mustun/.cache/huggingface/hub/datasets--dolphinteam--DolphinReef-labeled/snapshots/cba875a4337ad22e834e1d2c54ef8f9cc1ee66b3/",  # 
            regex=False,
        )

        self.xs = []
        self.ys = []

        for _, row in df.iterrows():
            self.xs.append(row['path'])

            if row['label'] not in label_to_id:
                if unknown_label is not None:
                    label_id = label_to_id[unknown_label]
                else:
                    raise KeyError(f"Unknown label: {row['label']}")
            else:
                label_id = label_to_id[row['label']]

            self.ys.append(label_id)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if self.feature_type == 'waveform':
            x = _get_waveform(
                self.xs[idx],
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate)

        elif self.feature_type == 'vggish':
            x = _get_vggish_spectrogram(
                self.xs[idx],
                max_duration=self.max_duration)

        elif self.feature_type == 'melspectrogram':
            x = _get_spectrogram(
                self.xs[idx],
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate)

        elif self.feature_type == 'mfcc':
            x = _get_spectrogram(
                self.xs[idx],
                max_duration=self.max_duration,
                target_sample_rate=self.sample_rate,
                return_mfcc=True)
        else:
            assert False

        return (x, self.ys[idx])


class CsvAudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        feature_type: str,
        sample_rate: int,
        max_duration: int,
        label_to_id: Optional[Dict[str, int]] = None,
        detection_label_cols: Optional[List[str]] = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.task = task
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.label_to_id = label_to_id
        self.detection_label_cols = detection_label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]

        if self.feature_type == "waveform":
            x = _get_waveform(path, max_duration=self.max_duration, target_sample_rate=self.sample_rate)
        elif self.feature_type == "vggish":
            x = _get_vggish_spectrogram(path, max_duration=self.max_duration)
        elif self.feature_type == "melspectrogram":
            x = _get_spectrogram(path, max_duration=self.max_duration, target_sample_rate=self.sample_rate)
        elif self.feature_type == "mfcc":
            x = _get_spectrogram(
                path, max_duration=self.max_duration, target_sample_rate=self.sample_rate, return_mfcc=True
            )
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        if self.task == "classification":
            if self.label_to_id is None:
                raise ValueError("label_to_id is required for classification task.")
            y = self.label_to_id[str(row["label"])]
            return x, y

        if self.detection_label_cols is None:
            raise ValueError("detection_label_cols is required for detection task.")
        y = torch.tensor(row[self.detection_label_cols].values.astype(np.float32))
        return x, y
