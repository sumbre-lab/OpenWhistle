"""Characterize CNN false negatives (missed whistles) on the test split.

Runs the published VGG16 whistle detector over the held-out `test` split of
dolphinteam/OpenWhistle-CNN, isolates false negatives (true label
whistle, predicted noise), and computes an SNR estimate plus other acoustic
measures for every whistle-labeled clip (missed and detected) so the two
groups can be compared.
"""

import argparse
import sys
import csv
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasets import Audio as HFAudio, load_dataset
from scipy.ndimage import median_filter
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm

# Allow direct execution from any working directory.
CNN_DIR = Path(__file__).resolve().parents[1]
if str(CNN_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_DIR))

from inference import InferenceConfig, load_inference_model, predict_batch
from utils.model import (
    SpectrogramConfig,
    build_spectrogram_plan,
    decode_audio_payload,
    make_spectrogram_image,
    resample_audio_if_needed,
)

DATASET_REPO = 'dolphinteam/OpenWhistle-CNN'
TEST_SPLIT = 'test'

COLOR_DETECTED = '#2a78d6'
COLOR_MISSED = '#eb6834'
COLOR_GRID = '#e1e0d9'
COLOR_TEXT = '#52514e'
COLOR_INK = '#0b0b0b'

CSV_FIELDS = [
    'recording', 'file_name', 'whistle_type', 'onset', 'offset', 'label',
    'prediction', 'score', 'snr_db', 'n_stft_frames_used', 'f0_source',
    'f0_min_hz', 'f0_max_hz', 'f0_mean_hz', 'f0_median_hz', 'status',
    'spectral_centroid_hz', 'spectral_bandwidth_hz',
]

NUMERIC_MEASURES = ['snr_db', 'score', 'f0_mean_hz', 'spectral_centroid_hz', 'spectral_bandwidth_hz']

# Ridge-tracking / SNR-pooling tuning, ported from DolphinDataset's
# utils/snr_ridge.py (estimate_f0_hz_spectrogram_ridge + snr_db_from_band_and_f0),
# which generated the existing snr_classification.csv/snr_detection_windows.csv
# sidecar figures. Frequency band and STFT parameters intentionally stay the
# CNN's own (SpectrogramConfig) rather than snr_ridge.py's independent
# 3-25kHz/2048-pt/256-hop defaults, so the SNR reflects what the CNN sees;
# only the ridge-tracking/pooling method itself is carried over.
RIDGE_HALF_WIDTH_HZ = 300.0
GUARD_HZ = 400.0
NOISE_BAND_HZ = 2000.0
RIDGE_PEAK_PERCENTILE = 30.0
RIDGE_SMOOTH_FRAMES = 5

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Characterize CNN false negatives (missed whistles) on the test split.'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=Path,
        default=None,
        help='Local checkpoint override. Defaults to the published Hugging Face model.',
    )
    parser.add_argument('--model-repo', default='dolphinteam/OpenWhistle-CNN-VGG16', help=argparse.SUPPRESS)
    parser.add_argument('--model-filename', default='model_vgg_final_best.pt', help=argparse.SUPPRESS)
    parser.add_argument('--output-dir', type=Path, default=CNN_DIR / 'runs/analysis')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cpu-only', action='store_true', default=False)
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Optional cap on the number of test rows processed (0 = no cap). Useful for smoke testing.',
    )
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser()
    if args.checkpoint_path is not None:
        args.checkpoint_path = args.checkpoint_path.expanduser()
    return args

def parse_whistle_type(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    if not stem.startswith('manual_'):
        return 'unknown'
    parts = stem[len('manual_'):].split('_')
    if len(parts) >= 2 and parts[0] in ('SW', 'NSW'):
        return f'{parts[0]}_{parts[1]}'
    return parts[0] if parts else 'unknown'

def load_test_rows(limit: int = 0):
    dataset = load_dataset(
        DATASET_REPO,
        split=TEST_SPLIT,
        columns=['audio', 'label', 'recording', 'file_name', 'onset', 'offset'],
    )
    dataset = dataset.cast_column('audio', HFAudio(decode=False))
    if limit > 0:
        dataset = dataset.select(range(min(limit, dataset.num_rows)))
    return dataset

def _empty_measures() -> dict[str, object]:
    return {
        'snr_db': float('nan'),
        'n_stft_frames_used': 0,
        'f0_source': 'spectrogram_ridge',
        'f0_min_hz': float('nan'),
        'f0_max_hz': float('nan'),
        'f0_mean_hz': float('nan'),
        'f0_median_hz': float('nan'),
        'status': 'no_ridge_detected',
        'spectral_centroid_hz': float('nan'),
        'spectral_bandwidth_hz': float('nan'),
    }

def _track_ridge_f0(band_power: np.ndarray, band_freqs: np.ndarray) -> np.ndarray | None:
    """Spectrogram-ridge per-frame frequency track (DolphinDataset utils/snr_ridge.py method).

    Subtracts each frame's own median power (across the band) as a
    per-frame background estimate, denoises with a 3x3 median filter, takes
    the per-frame argmax, discards low-confidence frames (peak below the
    30th percentile of all peak values), interpolates over the gaps, and
    median-smooths the resulting track.
    """
    if band_power.shape[0] == 0 or band_power.shape[1] == 0:
        return None

    background = np.median(band_power, axis=0, keepdims=True)
    enhanced = band_power - background
    enhanced[enhanced < 0] = 0
    enhanced = median_filter(enhanced, size=(3, 3))

    peak_idx = np.argmax(enhanced, axis=0)
    f0_hz = band_freqs[peak_idx].astype(np.float64)

    peak_vals = enhanced[peak_idx, np.arange(enhanced.shape[1])]
    threshold = np.percentile(peak_vals, RIDGE_PEAK_PERCENTILE)
    valid = peak_vals > threshold
    if not np.any(valid):
        return None
    f0_hz[~valid] = np.nan

    x = np.arange(len(f0_hz))
    nan_mask = np.isnan(f0_hz)
    f0_hz[nan_mask] = np.interp(x[nan_mask], x[valid], f0_hz[valid])
    return median_filter(f0_hz, size=RIDGE_SMOOTH_FRAMES).astype(np.float64)

def compute_clip_measures(audio: np.ndarray, fs: int, config: SpectrogramConfig) -> dict[str, object]:
    """Ridge-tracked, noise-subtracted band SNR, plus frequency and spectral descriptors.

    Frequency band and STFT parameters (window, wlen, hop, nfft) are the
    CNN's own SpectrogramConfig, so the SNR reflects exactly what the CNN
    sees. The ridge-tracking and SNR-pooling method itself is ported from
    DolphinDataset's utils/snr_ridge.py, which generated the existing
    snr_classification.csv/snr_detection_windows.csv sidecar figures: signal
    energy is pooled in a narrow band around the tracked ridge, noise energy
    from two bands flanking it (offset by a guard band), and the noise's
    estimated contribution to the signal band is subtracted before taking
    the ratio.

    An earlier version of this function used a plain per-frame STFT argmax
    (no background subtraction, smoothing, or noise-subtracted ratio) —
    before that, librosa.pyin (a speech-tuned pitch tracker), which found no
    usable pitch on ~83% of confirmed whistle clips. The ridge method
    recovers a track on nearly all of them while pooling SNR the same way
    the existing SNR sidecar figures do.
    """
    plan = build_spectrogram_plan(fs, config)
    freqs, _, sxx = scipy_spectrogram(
        audio,
        fs,
        nperseg=config.wlen,
        noverlap=config.wlen - config.hop,
        nfft=config.nfft,
        window=plan.window,
        scaling='density',
        mode='psd',
    )
    band_freqs = freqs[plan.low_idx:plan.high_idx]
    band_power = sxx[plan.low_idx:plan.high_idx, :]

    f0_track = _track_ridge_f0(band_power, band_freqs)
    if f0_track is None:
        return _empty_measures()

    fmin_hz = config.cut_low_frequency * 1000.0
    fmax_hz = min(config.cut_high_frequency * 1000.0, fs / 2.0 - 1.0)
    valid = np.isfinite(f0_track) & (f0_track >= fmin_hz) & (f0_track <= fmax_hz)
    if not np.any(valid):
        return _empty_measures()

    df = float(band_freqs[1] - band_freqs[0]) if len(band_freqs) > 1 else 1.0
    ridge_half_width_bins = max(1, int(RIDGE_HALF_WIDTH_HZ / df))
    guard_bins = max(1, int(GUARD_HZ / df))
    noise_band_bins = max(1, int(NOISE_BAND_HZ / df))

    signal_energy = 0.0
    noise_sum = 0.0
    noise_count = 0
    signal_bin_count = 0
    n_used = 0
    for t in range(band_power.shape[1]):
        if not valid[t]:
            continue
        n_used += 1
        f_idx = int(np.argmin(np.abs(band_freqs - f0_track[t])))

        s_lo = max(0, f_idx - ridge_half_width_bins)
        s_hi = min(len(band_freqs), f_idx + ridge_half_width_bins + 1)
        signal_slice = band_power[s_lo:s_hi, t]
        signal_energy += float(np.sum(signal_slice))
        signal_bin_count += signal_slice.size

        n1_lo = max(0, f_idx - guard_bins - noise_band_bins)
        n1_hi = max(0, f_idx - guard_bins)
        n2_lo = min(len(band_freqs), f_idx + guard_bins + 1)
        n2_hi = min(len(band_freqs), f_idx + guard_bins + noise_band_bins + 1)
        if n1_hi > n1_lo:
            noise_slice = band_power[n1_lo:n1_hi, t]
            noise_sum += float(np.sum(noise_slice))
            noise_count += noise_slice.size
        if n2_hi > n2_lo:
            noise_slice = band_power[n2_lo:n2_hi, t]
            noise_sum += float(np.sum(noise_slice))
            noise_count += noise_slice.size

    if n_used == 0 or noise_count == 0 or signal_bin_count == 0:
        return _empty_measures()

    noise_mean = noise_sum / noise_count
    noise_energy_in_signal = noise_mean * signal_bin_count
    signal_energy_above_noise = max(signal_energy - noise_energy_in_signal, 1e-12)
    snr_db = 10.0 * float(np.log10(signal_energy_above_noise / noise_energy_in_signal))

    voiced_f0 = f0_track[valid]

    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=fs, n_fft=config.nfft, hop_length=config.hop, win_length=config.wlen,
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=fs, n_fft=config.nfft, hop_length=config.hop, win_length=config.wlen,
    )

    return {
        'snr_db': snr_db,
        'n_stft_frames_used': int(n_used),
        'f0_source': 'spectrogram_ridge',
        'f0_min_hz': float(np.min(voiced_f0)),
        'f0_max_hz': float(np.max(voiced_f0)),
        'f0_mean_hz': float(np.mean(voiced_f0)),
        'f0_median_hz': float(np.median(voiced_f0)),
        'status': 'ok',
        'spectral_centroid_hz': float(np.mean(centroid)),
        'spectral_bandwidth_hz': float(np.mean(bandwidth)),
    }

def write_measures_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

def _describe(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'n': 0, 'mean': None, 'median': None, 'q1': None, 'q3': None}
    return {
        'n': int(arr.size),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'q1': float(np.percentile(arr, 25)),
        'q3': float(np.percentile(arr, 75)),
    }

def _miss_rate_table(records: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        key = str(record.get(group_key) or 'unknown')
        bucket = counts.setdefault(key, {'total': 0, 'missed': 0})
        bucket['total'] += 1
        if record['prediction'] == 0:
            bucket['missed'] += 1
    rows = [
        {
            'key': key,
            'total': bucket['total'],
            'missed': bucket['missed'],
            'miss_rate': bucket['missed'] / bucket['total'],
        }
        for key, bucket in counts.items()
    ]
    rows.sort(key=lambda item: item['missed'], reverse=True)
    return rows

def build_summary(records: list[dict[str, object]], confusion: dict[str, int]) -> dict[str, object]:
    missed = [r for r in records if r['prediction'] == 0]
    detected = [r for r in records if r['prediction'] == 1]

    measures: dict[str, object] = {}
    for measure in NUMERIC_MEASURES:
        missed_values = [r[measure] for r in missed]
        detected_values = [r[measure] for r in detected]
        entry: dict[str, object] = {
            'missed': _describe(missed_values),
            'detected': _describe(detected_values),
        }
        missed_arr = np.asarray(missed_values, dtype=float)
        detected_arr = np.asarray(detected_values, dtype=float)
        missed_arr = missed_arr[np.isfinite(missed_arr)]
        detected_arr = detected_arr[np.isfinite(detected_arr)]
        if missed_arr.size >= 2 and detected_arr.size >= 2:
            statistic, p_value = mannwhitneyu(missed_arr, detected_arr, alternative='two-sided')
            entry['mannwhitney_u'] = float(statistic)
            entry['p_value'] = float(p_value)
        measures[measure] = entry

    return {
        'counts': confusion,
        'n_missed': len(missed),
        'n_detected': len(detected),
        'measures': measures,
        'miss_rate_by_whistle_type': _miss_rate_table(records, 'whistle_type'),
        'miss_rate_by_recording': _miss_rate_table(records, 'recording'),
    }

def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

def _style_ax(ax) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT)
    ax.set_axisbelow(True)

def _finite(records: list[dict[str, object]], key: str) -> np.ndarray:
    arr = np.asarray([r[key] for r in records], dtype=float)
    return arr[np.isfinite(arr)]

def _violin(ax, detected_values: np.ndarray, missed_values: np.ndarray, ylabel: str, title: str) -> None:
    datasets_to_plot = [values for values in (detected_values, missed_values) if values.size > 0]
    if len(datasets_to_plot) < 2:
        ax.set_title(f'{title} (insufficient data)', color=COLOR_INK)
        _style_ax(ax)
        return
    parts = ax.violinplot([detected_values, missed_values], showmedians=True)
    for body, color in zip(parts['bodies'], (COLOR_DETECTED, COLOR_MISSED)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[key].set_color(COLOR_TEXT)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Detected (TP)', 'Missed (FN)'])
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=COLOR_INK)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    _style_ax(ax)

def _miss_rate_bar(
    records: list[dict[str, object]],
    group_key: str,
    out_path: Path,
    title: str,
    min_count: int = 10,
    top_n: int = 20,
) -> None:
    rows = [row for row in _miss_rate_table(records, group_key) if row['total'] >= min_count]
    if not rows:
        return
    rows.sort(key=lambda item: item['miss_rate'], reverse=True)
    rows = rows[:top_n]
    labels = [f"{row['key']} (n={row['total']})" for row in rows]
    values = [row['miss_rate'] * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.35 * len(rows) + 1)))
    y_pos = np.arange(len(rows))
    ax.barh(y_pos, values, color=COLOR_MISSED, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Miss rate (%)')
    ax.set_title(title, color=COLOR_INK)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor='white')
    plt.close(fig)

def make_plots(records: list[dict[str, object]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    detected = [r for r in records if r['prediction'] == 1]
    missed = [r for r in records if r['prediction'] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _violin(axes[0], _finite(detected, 'snr_db'), _finite(missed, 'snr_db'), 'SNR (dB)', 'Whistle SNR: detected vs. missed')
    _violin(axes[1], _finite(detected, 'score'), _finite(missed, 'score'), 'Model score P(whistle)', 'Model confidence: detected vs. missed')
    fig.tight_layout()
    fig.savefig(figures_dir / 'snr_and_score_violin.png', dpi=200, facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _violin(axes[0], _finite(detected, 'f0_mean_hz'), _finite(missed, 'f0_mean_hz'), 'Mean f0 (Hz)', 'Dominant frequency: detected vs. missed')
    _violin(axes[1], _finite(detected, 'spectral_bandwidth_hz'), _finite(missed, 'spectral_bandwidth_hz'), 'Spectral bandwidth (Hz)', 'Spectral bandwidth: detected vs. missed')
    fig.tight_layout()
    fig.savefig(figures_dir / 'frequency_descriptors_violin.png', dpi=200, facecolor='white')
    plt.close(fig)

    _miss_rate_bar(
        records, 'whistle_type', figures_dir / 'miss_rate_by_whistle_type.png',
        'Miss rate by signature-whistle type (min n=10)',
    )
    _miss_rate_bar(
        records, 'recording', figures_dir / 'miss_rate_by_recording.png',
        'Miss rate by recording (min n=10)',
    )

def run_analysis(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    figures_dir = output_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_config = InferenceConfig(
        checkpoint_path=args.checkpoint_path,
        model_repo=args.model_repo,
        model_filename=args.model_filename,
    )
    checkpoint_path = infer_config.resolved_checkpoint_path()
    print(f'Using checkpoint: {checkpoint_path}')
    model = load_inference_model(checkpoint_path, args.cpu_only)
    print(f'Using device: {model.device}')

    dataset = load_test_rows(args.limit)
    print(f'Loaded {dataset.num_rows} test rows from {DATASET_REPO}[{TEST_SPLIT}]')

    plan_cache: dict[int, object] = {}

    def get_plan(fs: int):
        if fs not in plan_cache:
            plan_cache[fs] = build_spectrogram_plan(fs, model.spectrogram_config)
        return plan_cache[fs]

    records: list[dict[str, object]] = []
    confusion = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}

    batch_images: list[np.ndarray] = []
    batch_meta: list[tuple] = []

    def flush_batch() -> None:
        if not batch_images:
            return
        images = np.stack(batch_images, axis=0)
        scores = predict_batch(model, images)
        for meta, score in zip(batch_meta, scores):
            label, recording, file_name, onset, offset, audio, fs = meta
            prediction = int(float(score) >= args.threshold)
            if label == 1 and prediction == 1:
                confusion['tp'] += 1
            elif label == 1 and prediction == 0:
                confusion['fn'] += 1
            elif label == 0 and prediction == 1:
                confusion['fp'] += 1
            else:
                confusion['tn'] += 1
            if label == 1:
                measures = compute_clip_measures(audio, fs, model.spectrogram_config)
                records.append({
                    'recording': recording,
                    'file_name': file_name,
                    'whistle_type': parse_whistle_type(file_name),
                    'onset': float(onset),
                    'offset': float(offset),
                    'label': int(label),
                    'prediction': prediction,
                    'score': float(score),
                    **measures,
                })
        batch_images.clear()
        batch_meta.clear()

    for row in tqdm(dataset, total=dataset.num_rows, desc='test rows'):
        fs, audio = decode_audio_payload(row['audio'])
        fs, audio = resample_audio_if_needed(audio, fs, model.spectrogram_config.target_fs)
        plan = get_plan(fs)
        image = make_spectrogram_image(audio, fs, model.spectrogram_config, plan)
        batch_images.append(image)
        batch_meta.append((row['label'], row['recording'], row['file_name'], row['onset'], row['offset'], audio, fs))
        if len(batch_images) >= args.batch_size:
            flush_batch()
    flush_batch()

    print('\nConfusion counts (recomputed locally):')
    print(f"  TP={confusion['tp']}  FN={confusion['fn']}  FP={confusion['fp']}  TN={confusion['tn']}")

    measures_path = output_dir / 'test_whistle_measures.csv'
    write_measures_csv(measures_path, records)

    summary = build_summary(records, confusion)
    summary_path = output_dir / 'missed_whistles_summary.json'
    write_summary_json(summary_path, summary)

    make_plots(records, figures_dir)

    print(f'\nWrote {len(records)} whistle-clip measures to {measures_path}')
    print(f'Wrote summary to {summary_path}')
    print(f'Wrote figures to {figures_dir}')



def main(argv: list[str] | None = None) -> None:
    run_analysis(parse_args(argv))

if __name__ == '__main__':
    main()
