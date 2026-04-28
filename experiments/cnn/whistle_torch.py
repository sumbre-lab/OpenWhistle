import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import soundfile as sf
import torch
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal.windows import blackman
from torch import nn
from torchvision import models


ARCHITECTURE_LEGACY = 'torchvision_vgg16_frozen_features_custom_head'
ARCHITECTURE_TRAINABLE_SMALL_HEAD = 'torchvision_vgg16_trainable_flatten_fc50_fc20_out2'

DEFAULT_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZATION_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SpectrogramConfig:
    image_size: tuple[int, int] = (224, 224)  # (height, width)
    cut_low_frequency: float = 2.0
    cut_high_frequency: float = 22.0
    wlen: int = 1024
    nfft: int = 1024
    sliding_window: float = 0.4
    target_fs: Optional[int] = 96_000

    @property
    def hop(self) -> int:
        return round(0.5 * self.wlen)

    def to_metadata(self) -> dict[str, object]:
        return {
            'image_size': list(self.image_size),
            'cut_low_frequency': float(self.cut_low_frequency),
            'cut_high_frequency': float(self.cut_high_frequency),
            'wlen': int(self.wlen),
            'nfft': int(self.nfft),
            'sliding_window': float(self.sliding_window),
            'target_fs': None if self.target_fs is None else int(self.target_fs),
        }

    @classmethod
    def from_metadata(cls, metadata: Optional[dict[str, object]]) -> 'SpectrogramConfig':
        if not metadata:
            return cls()

        image_size = metadata.get('image_size', (224, 224))
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            parsed_image_size = (int(image_size[0]), int(image_size[1]))
        else:
            parsed_image_size = (224, 224)

        target_fs = metadata.get('target_fs')
        return cls(
            image_size=parsed_image_size,
            cut_low_frequency=float(metadata.get('cut_low_frequency', 2.0)),
            cut_high_frequency=float(metadata.get('cut_high_frequency', 22.0)),
            wlen=int(metadata.get('wlen', 1024)),
            nfft=int(metadata.get('nfft', 1024)),
            sliding_window=float(metadata.get('sliding_window', 0.4)),
            target_fs=None if target_fs is None else int(target_fs),
        )


def _load_torchvision_vgg16(pretrained_backbone: bool) -> models.VGG:
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    return models.vgg16(weights=weights)


class TorchLegacyVGG16WhistleModel(nn.Module):
    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        backbone = _load_torchvision_vgg16(pretrained_backbone)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        if pretrained_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        in_features = backbone.classifier[0].in_features
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 50)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc3(x))
        return self.out(x)


class TorchTrainableSmallHeadVGG16Model(nn.Module):
    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        backbone = _load_torchvision_vgg16(pretrained_backbone)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[0].in_features
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 50)
        self.fc2 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def build_torch_model(
    architecture: str,
    pretrained_backbone: bool = False,
) -> nn.Module:
    if architecture == ARCHITECTURE_LEGACY:
        return TorchLegacyVGG16WhistleModel(pretrained_backbone=pretrained_backbone)
    if architecture == ARCHITECTURE_TRAINABLE_SMALL_HEAD:
        return TorchTrainableSmallHeadVGG16Model(pretrained_backbone=pretrained_backbone)
    raise ValueError(f'Unsupported Torch architecture: {architecture}')


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32)


def decode_audio_payload(audio_payload: object) -> tuple[int, np.ndarray]:
    if isinstance(audio_payload, dict):
        if 'array' in audio_payload and 'sampling_rate' in audio_payload:
            fs = int(audio_payload['sampling_rate'])
            audio = ensure_mono_float32(audio_payload['array'])
            return fs, audio

        audio_bytes = audio_payload.get('bytes')
        audio_path = audio_payload.get('path')

        if audio_bytes is not None:
            audio, fs = sf.read(BytesIO(audio_bytes), always_2d=False)
            return int(fs), ensure_mono_float32(audio)

        if audio_path:
            audio, fs = sf.read(str(audio_path), always_2d=False)
            return int(fs), ensure_mono_float32(audio)

    if isinstance(audio_payload, (str, os.PathLike, Path)):
        audio, fs = sf.read(str(audio_payload), always_2d=False)
        return int(fs), ensure_mono_float32(audio)

    raise TypeError(f'Unsupported audio payload type: {type(audio_payload)!r}')


def resample_audio_if_needed(
    audio: np.ndarray,
    fs: int,
    target_fs: Optional[int],
) -> tuple[int, np.ndarray]:
    audio = ensure_mono_float32(audio)
    if target_fs is None or int(fs) == int(target_fs):
        return int(fs), audio

    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            'librosa is required to resample audio when target_fs differs from the '
            'input sampling rate.'
        ) from exc

    resampled = librosa.resample(audio, orig_sr=int(fs), target_sr=int(target_fs))
    return int(target_fs), np.asarray(resampled, dtype=np.float32)


def make_spectrogram_image(
    audio: np.ndarray,
    fs: int,
    config: SpectrogramConfig,
) -> np.ndarray:
    audio = ensure_mono_float32(audio)
    win = blackman(config.wlen, sym=False)
    f, _, sxx = scipy_spectrogram(
        audio,
        fs,
        nperseg=config.wlen,
        noverlap=config.wlen - config.hop,
        nfft=config.nfft,
        window=win,
        scaling='density',
        mode='psd',
    )

    sxx = 10.0 * np.log10(np.abs(sxx) + 1e-19)
    sxx = (sxx - np.min(sxx)) / (np.max(sxx) - np.min(sxx) + 1e-12) * 255.0

    lo_hz = config.cut_low_frequency * 1000
    hi_hz = config.cut_high_frequency * 1000
    low_idx = int(np.searchsorted(f, lo_hz))
    high_idx = int(np.searchsorted(f, hi_hz))

    sxx_cropped = np.flipud(sxx[low_idx:high_idx, :])
    img_gray = np.clip(sxx_cropped, 0, 255).astype(np.uint8)

    height, width = config.image_size
    resized = cv2.resize(img_gray, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.stack([resized, resized, resized], axis=2)


def audio_payload_to_spectrogram_image(
    audio_payload: object,
    config: SpectrogramConfig,
) -> np.ndarray:
    fs, audio = decode_audio_payload(audio_payload)
    fs, audio = resample_audio_if_needed(audio, fs, config.target_fs)
    return make_spectrogram_image(audio, fs, config)


def make_spectrogram_batch(
    audio: np.ndarray,
    fs: int,
    start_sample: int,
    n_windows: int,
    config: SpectrogramConfig,
) -> tuple[np.ndarray, np.ndarray]:
    audio = ensure_mono_float32(audio)
    samples_per_window = round(config.sliding_window * fs)

    images: list[np.ndarray] = []
    start_sec: list[float] = []
    for index in range(n_windows):
        start = start_sample + index * samples_per_window
        stop = start + samples_per_window
        if stop > len(audio):
            break
        image = make_spectrogram_image(audio[start:stop], fs, config)
        images.append(image)
        start_sec.append(start / fs)

    if not images:
        empty_images = np.empty((0,) + tuple(config.image_size) + (3,), dtype=np.uint8)
        empty_times = np.array([], dtype=np.float64)
        return empty_images, empty_times

    return np.asarray(images, dtype=np.uint8), np.asarray(start_sec, dtype=np.float64)


def normalize_uint8_image_to_tensor(
    image_uint8: np.ndarray,
    mean: tuple[float, float, float] = DEFAULT_NORMALIZATION_MEAN,
    std: tuple[float, float, float] = DEFAULT_NORMALIZATION_STD,
) -> torch.Tensor:
    image = torch.from_numpy(image_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (image - mean_tensor) / std_tensor


def normalize_uint8_batch_to_torch(
    images_uint8: np.ndarray,
    mean: tuple[float, float, float] = DEFAULT_NORMALIZATION_MEAN,
    std: tuple[float, float, float] = DEFAULT_NORMALIZATION_STD,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    images = torch.from_numpy(images_uint8.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    if device is not None:
        images = images.to(device)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor
