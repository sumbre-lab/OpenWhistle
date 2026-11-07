"""Whistle F0 tracks from the bottlenose-dolphin CREPE model of bioacoustic_F0_estimation.

The model and the frequency-compression recipe come from
https://github.com/mim-team/bioacoustic_F0_estimation. ``estimate_f0`` reproduces
``whistle_extraction/whistle_f0_estimation.estimate_f0_crepe`` with the settings that
produced the published HF ``snr_db`` column: stationary noise reduction, 20x frequency
compression, 5 ms step and weighted-argmax decoding. Two deliberate differences:
clips longer than ``chunk_s`` are tracked in independent chunks to bound memory, and
torchcrepe's per-process random pitch dither is drawn from ``dither_seed`` so that
tracks are reproducible.

Import this module before ``datasets``/``pyarrow``: loading torch after them can
crash some environments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch
import torchcrepe
import librosa
import noisereduce
import numpy as np

WEIGHTS_URL = ('https://github.com/mim-team/bioacoustic_F0_estimation/raw/'
               'b57cc78f5b1c42b4eb61bdc4dc0f4430211c62ea/paper_experiments/crepe_weights/'
               'model_only-0_bottlenose_dolphins.pth')
WEIGHTS_SHA256 = '735eebba91c4fdce26c1676af28064bc6a9ac7983a221c519b612837295c04eb'
WEIGHTS_CACHE = Path.home() / '.cache' / 'openwhistle' / 'crepe' / 'model_only-0_bottlenose_dolphins.pth'


@dataclass(frozen=True)
class CrepeConfig:
    step_s: float = 0.005
    compress: float = 20.0
    reduce_noise: bool = True
    decoder: str = 'weighted_argmax'
    chunk_s: float = 120.0
    dither_seed: int = 0

    def to_dict(self):
        return asdict(self)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_weights(path=None):
    """Return a verified weights file: ``path`` if given, else the pinned download (cached)."""
    if path is None:
        path = WEIGHTS_CACHE
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.part')
            torch.hub.download_url_to_file(WEIGHTS_URL, str(temporary), hash_prefix=WEIGHTS_SHA256)
            temporary.replace(path)
    path = Path(path).expanduser()
    if _sha256(path) != WEIGHTS_SHA256:
        raise ValueError(f'{path} is not the pinned bottlenose-dolphin CREPE model (sha256 {WEIGHTS_SHA256}).')
    return path


def load_model(weights=None, device=None):
    """Return (model, device) for the verified bottlenose-dolphin CREPE weights."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchcrepe.Crepe('full').eval().to(device)
    model.load_state_dict(torch.load(resolve_weights(weights), map_location=device, weights_only=True))
    return model, device


def _seed_decoder_dither(seed):
    """Fix torchcrepe's weighted-argmax pitch dither.

    torchcrepe draws the dither of its bin-to-cents weights (up to one 20-cent bin) once per
    process from the unseeded global RNG, so tracks change between runs by up to ~140 Hz at
    12 kHz after compression. This draws the same noise, the same way, from a fixed seed.
    """
    from scipy.stats import triang

    cents = torchcrepe.CENTS_PER_BIN * torch.arange(360) + 1997.3794084376191
    noise = triang.rvs(c=0.5, loc=-torchcrepe.CENTS_PER_BIN, scale=2 * torchcrepe.CENTS_PER_BIN,
                       size=cents.size(), random_state=np.random.RandomState(seed))
    torchcrepe.decode.weighted_argmax.weights = (cents + cents.new_tensor(noise))[None, :, None]


def _track_chunk(model, device, audio, fs, config, batch_size):
    if config.reduce_noise:
        audio = noisereduce.reduce_noise(y=audio, sr=fs, stationary=True)
    # Frequency compression: resample to 20 x 16 kHz, then present it to CREPE as 16 kHz audio.
    compressed_fs = int(torchcrepe.SAMPLE_RATE * config.compress)
    if config.compress != 1.0:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=compressed_fs)
        fs = compressed_fs
    hop = int(config.step_s * config.compress * torchcrepe.SAMPLE_RATE)
    batches = torchcrepe.core.preprocess(torch.tensor(np.asarray(audio, dtype=np.float32)).unsqueeze(0),
                                         torchcrepe.SAMPLE_RATE,
                                         hop_length=hop, batch_size=batch_size, device=device)
    if config.decoder == 'weighted_argmax':
        _seed_decoder_dither(config.dither_seed)
    with torch.inference_mode():
        probabilities = torch.vstack([model(frames).cpu() for frames in batches]).T.unsqueeze(0)
        decoder = getattr(torchcrepe.decode, config.decoder)
        f0 = (torchcrepe.core.postprocess(probabilities, decoder=decoder) * config.compress).squeeze()
    confidence = probabilities.max(axis=1)[0].squeeze()
    time = np.arange(0, len(audio), hop) / fs
    f0, confidence = (np.asarray(x.numpy(), dtype=np.float64).ravel() for x in (f0, confidence))
    n = min(len(time), len(f0), len(confidence))
    return time[:n], f0[:n], confidence[:n]


def estimate_f0(model, device, audio, fs, config=CrepeConfig(), batch_size=64):
    """Return (time_s, f0_hz, confidence) float64 arrays for a mono waveform."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(audio):
        raise ValueError('Cannot track F0 in empty audio.')
    # Clips up to chunk_s are tracked whole; longer ones in equal chunks (no tiny tail chunk).
    n_chunks = -(-len(audio) // max(1, int(round(config.chunk_s * fs))))
    bounds = np.linspace(0, len(audio), n_chunks + 1).astype(int)
    parts = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        time, f0, confidence = _track_chunk(model, device, audio[start:stop], fs, config, batch_size)
        parts.append((time + start / fs, f0, confidence))
    return tuple(np.concatenate(column) for column in zip(*parts))
