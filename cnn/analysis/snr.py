"""Noise-subtracted band SNR around a whistle F0 track: a CREPE track or a spectrogram ridge."""
from dataclasses import asdict, dataclass
from io import BytesIO

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import spectrogram


@dataclass(frozen=True)
class SNRConfig:
    low_hz: float = 3000.0
    high_hz: float = 25000.0
    nfft: int = 2048
    hop: int = 256
    ridge_half_width_hz: float = 300.0
    noise_offset_hz: float = 400.0
    noise_width_hz: float = 2000.0
    f0_confidence_threshold: float = 0.05  # estimate_snr_with_f0 only

    def to_dict(self):
        return asdict(self)


def decode_audio(payload):
    if isinstance(payload, dict) and payload.get('bytes') is not None:
        source = BytesIO(payload['bytes'])
    elif isinstance(payload, dict) and payload.get('path'):
        source = payload['path']
    else:
        raise ValueError('Expected a Hugging Face Audio(decode=False) bytes/path payload.')
    audio, fs = sf.read(source, dtype='float32', always_2d=True)
    return audio.mean(axis=1), int(fs)


def estimate_snr(audio, fs, config=SNRConfig()):
    """Return (dB, status, number of frames); pool energies across the full clip."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError('Audio must be a finite mono waveform.')
    if len(audio) < config.nfft:
        return float('nan'), 'too_short', 0
    frequencies, _, power = spectrogram(
        audio, fs=fs, window='blackman', nperseg=config.nfft,
        noverlap=config.nfft - config.hop, nfft=config.nfft,
        scaling='density', mode='psd',
    )
    selected = (frequencies >= config.low_hz) & (frequencies <= config.high_hz)
    band = power[selected]
    if len(band) < 3:
        return float('nan'), 'frequency_band_unavailable', 0
    enhanced = median_filter(np.maximum(band - np.median(band, axis=0), 0), size=(3, 3))
    indices = np.argmax(enhanced, axis=0)
    peaks = enhanced[indices, np.arange(band.shape[1])]
    valid = (peaks > 0) & (peaks >= np.percentile(peaks, 30))
    if not valid.any():
        return float('nan'), 'no_ridge_detected', 0
    times = np.arange(len(indices))
    indices = median_filter(np.interp(times, times[valid], indices[valid]), size=5)
    indices = np.rint(indices).astype(int)
    df = fs / config.nfft
    width = max(1, int(config.ridge_half_width_hz / df))
    offset = max(width + 1, int(config.noise_offset_hz / df))
    noise_width = max(1, int(config.noise_width_hz / df))
    cumulative = np.vstack([np.zeros((1, band.shape[1])), np.cumsum(band, axis=0, dtype=np.float64)])

    def energy(lo, hi):
        lo, hi = np.clip(lo, 0, len(band)), np.clip(hi, 0, len(band))
        return float(np.sum(cumulative[hi, times] - cumulative[lo, times])), int(np.sum(hi - lo))

    signal, n_signal = energy(indices - width, indices + width + 1)
    lower, n_lower = energy(indices - offset - noise_width, indices - offset)
    upper, n_upper = energy(indices + offset + 1, indices + offset + noise_width + 1)
    if not (n_lower + n_upper):
        return float('nan'), 'no_noise_bins', 0
    noise = (lower + upper) / (n_lower + n_upper) * n_signal
    if noise <= 0:
        return float('nan'), 'no_noise_energy', 0
    # Relative floor preserves amplitude invariance and caps undefined negative signal at -120 dB.
    ratio = max((signal - noise) / noise, 1e-12)
    return float(10 * np.log10(ratio)), 'ok', band.shape[1]


def estimate_snr_with_f0(audio, fs, f0_time, f0_hz, f0_confidence, config=SNRConfig(), block_frames=4096):
    """Return (dB, status, frames used); pool energies only over confident F0-track frames.

    Vectorised port of DolphinDataset ``utils/snr_ridge.estimate_snr_db_with_time_series_f0``,
    which produced the published HF ``snr_db`` column from CREPE tracks. The track is
    interpolated onto a centred Hann STFT (librosa); a frame counts when its confidence is
    at least ``f0_confidence_threshold`` and its frequency lies in the band. Signal is the
    energy within ``ridge_half_width_hz`` of the track bin, noise the mean bin energy of the
    flanking bands ``noise_offset_hz`` away; the noise expected in the signal band is
    subtracted. Undefined values (signal <= noise) use the relative -120 dB floor of
    ``estimate_snr``. The STFT is processed in blocks of frames to bound memory.
    """
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError('Audio must be a finite mono waveform.')
    if not len(audio):
        return float('nan'), 'too_short', 0
    times, f0_hz, f0_confidence = (np.asarray(x, dtype=np.float64).ravel()
                                   for x in (f0_time, f0_hz, f0_confidence))
    if not len(times) == len(f0_hz) == len(f0_confidence) or len(times) < 2:
        return float('nan'), 'f0_track_invalid', 0
    order = np.argsort(times)
    times, f0_hz, f0_confidence = times[order], f0_hz[order], f0_confidence[order]
    f0_hz = np.where(np.isfinite(f0_hz), f0_hz, np.nan)

    frequencies = librosa.fft_frequencies(sr=fs, n_fft=config.nfft)
    selected = np.flatnonzero((frequencies >= config.low_hz) & (frequencies <= config.high_hz))
    if len(selected) < 2:
        return float('nan'), 'frequency_band_unavailable', 0
    band_frequencies = frequencies[selected]
    n_bins = len(selected)
    n_frames = 1 + len(audio) // config.hop  # centred STFT
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=fs, hop_length=config.hop)
    f0 = np.interp(frame_times, times, f0_hz, left=np.nan, right=np.nan)
    confidence = np.interp(frame_times, times, f0_confidence, left=0.0, right=0.0)
    with np.errstate(invalid='ignore'):
        valid = ((confidence >= config.f0_confidence_threshold) & np.isfinite(f0)
                 & (f0 >= config.low_hz) & (f0 <= config.high_hz))
    used = np.flatnonzero(valid)
    if not len(used):
        return float('nan'), 'no_confident_f0', 0

    df = band_frequencies[1] - band_frequencies[0]
    width = max(1, int(config.ridge_half_width_hz / df))
    guard = max(1, int(config.noise_offset_hz / df))  # equals estimate_snr's offset below ~200 kHz
    noise_width = max(1, int(config.noise_width_hz / df))
    # Nearest band bin to the track (np.argmin semantics: lower bin on a tie) on the uniform grid.
    target = f0[used]
    lower = np.clip(np.floor((target - band_frequencies[0]) / df).astype(int), 0, n_bins - 1)
    upper = np.minimum(lower + 1, n_bins - 1)
    track_bins = np.where(np.abs(band_frequencies[upper] - target) < np.abs(band_frequencies[lower] - target),
                          upper, lower)

    padded = np.pad(audio, config.nfft // 2)  # librosa's centred, zero-padded framing
    signal = noise = 0.0
    n_signal = n_noise = 0
    for start in range(0, n_frames, block_frames):
        stop = min(start + block_frames, n_frames)
        in_block = (used >= start) & (used < stop)
        if not in_block.any():
            continue
        spectrum = librosa.stft(padded[start * config.hop:(stop - 1) * config.hop + config.nfft],
                                n_fft=config.nfft, hop_length=config.hop, window='hann', center=False)
        band = np.abs(spectrum[selected][:, used[in_block] - start]) ** 2
        cumulative = np.vstack([np.zeros((1, band.shape[1])), np.cumsum(band, axis=0, dtype=np.float64)])
        columns = np.arange(band.shape[1])
        bins = track_bins[in_block]

        def energy(lo, hi):
            lo, hi = np.clip(lo, 0, n_bins), np.clip(hi, 0, n_bins)
            return float(np.sum(cumulative[hi, columns] - cumulative[lo, columns])), int(np.sum(hi - lo))

        s, n_s = energy(bins - width, bins + width + 1)
        lo, n_lo = energy(bins - guard - noise_width, bins - guard)
        up, n_up = energy(bins + guard + 1, bins + guard + noise_width + 1)
        signal, n_signal = signal + s, n_signal + n_s
        noise, n_noise = noise + lo + up, n_noise + n_lo + n_up
    if not (n_noise and n_signal):
        return float('nan'), 'no_noise_bins', 0
    noise_in_signal = noise / n_noise * n_signal
    if noise_in_signal <= 0:
        return float('nan'), 'no_noise_energy', 0
    ratio = max((signal - noise_in_signal) / noise_in_signal, 1e-12)
    return float(10 * np.log10(ratio)), 'ok', int(len(used))
