"""Compute audio SNR from OpenWhistle's HF datasets and render manuscript panel F."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from snr import SNRConfig, decode_audio, estimate_snr

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTION = 'https://huggingface.co/collections/dolphinteam/neurips26-openwhistle'
SOURCES = {
    'pretraining': ('dolphinteam/OpenWhistle-Pretraining', 'default', ('train', 'validation')),
    'classification_all': ('dolphinteam/OpenWhistle-Classification-Finetuning', 'all', ('train', 'validation', 'test')),
}
FIELDS = ('split', 'index', 'sampling_rate', 'duration_s', 'snr_db', 'status', 'n_stft_frames_used')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'cnn/runs/snr')
    parser.add_argument('--limit', type=int, default=0, help='First N audio rows per dataset for a diagnostic (0 = all).')
    parser.add_argument('--pretraining-revision', default='main')
    parser.add_argument('--classification-revision', default='main')
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1),
                        help='Concurrent CPU SNR calculations (default: up to 16).')
    parser.add_argument('--plot-only', action='store_true', help='Reuse this script’s computed CSVs and provenance; no HF access.')
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit must be nonnegative.')
    if args.workers < 1:
        parser.error('--workers must be positive.')
    return args


def compute_row(task):
    split, index, payload, config = task
    audio, fs = decode_audio(payload)
    snr, status, frames = estimate_snr(audio, fs, config)
    return dict(split=split, index=index, sampling_rate=fs,
                duration_s=len(audio) / fs, snr_db=snr,
                status=status, n_stft_frames_used=frames)


def bounded_results(pool, tasks, max_pending):
    """Overlap reading and computation without queuing the whole audio corpus."""
    pending = deque()
    try:
        for task in tasks:
            pending.append(pool.submit(compute_row, task))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()


def compute_dataset(repo, config, splits, revision, limit, output, snr_config, workers=1):
    from datasets import Audio, load_dataset
    from pyarrow.dataset import ParquetFragmentScanOptions
    values, counts, processed = [], Counter(), 0
    temporary = output.with_suffix('.csv.tmp')
    with ThreadPoolExecutor(max_workers=workers) as pool, temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for split in splits:
            dataset = load_dataset(repo, name=config, split=split, revision=revision,
                                   streaming=True, columns=['audio'], batch_size=workers,
                                   fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False))
            dataset = dataset.cast_column('audio', Audio(decode=False))
            if limit:
                dataset = dataset.take(limit - processed)
            split_info = getattr(dataset.info, 'splits', None) if hasattr(dataset, 'info') else None
            total = split_info[split].num_examples if split_info and split in split_info else None
            if limit and total is not None:
                total = min(total, limit - processed)
            tasks = ((split, index, row['audio'], snr_config) for index, row in enumerate(dataset))
            records = bounded_results(pool, tasks, max_pending=workers * 2)
            for record in tqdm(records, total=total, desc=f'{repo}/{split}', unit='clip'):
                writer.writerow(record)
                status, snr = record['status'], record['snr_db']
                counts[status] += 1
                processed += 1
                if status == 'ok' and np.isfinite(snr):
                    values.append(snr)
            if limit and processed >= limit:
                break
    temporary.replace(output)
    return np.asarray(values), {'processed': processed, 'status_counts': dict(counts)}


def plot_dataset_snr(arrays, output_dir, diagnostic=False):
    import seaborn as sns
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from datasets_figures.scripts.classification_overview import draw_snr_detection_vs_gold_on_ax
    for values in arrays:
        if len(values) < 2 or np.ptp(values) == 0:
            raise ValueError('At least two distinct valid SNR values per dataset are required.')
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    draw_snr_detection_vs_gold_on_ax(ax, arrays[0], arrays[1], sns, slim_violin=True,
                                   group_labels=('Pretraining', 'Classification\n(all)'))
    ax.set_ylabel('Estimated SNR (dB)')
    if diagnostic:
        ax.set_title('SNR (diagnostic subset)', fontweight='bold')
    fig.subplots_adjust(left=0.18, right=0.77, bottom=0.16, top=0.9)
    for extension in ('png', 'pdf'):
        path = output_dir / f'snr_pretraining_vs_classification_all.{extension}'
        fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        print(f'Wrote {path}')
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / 'snr_protocol.json'
    arrays = []
    if args.plot_only:
        protocol = json.loads(protocol_path.read_text())
        if protocol['estimator'] != SNRConfig().to_dict():
            raise ValueError('Cached SNR protocol differs from the current estimator.')
        for group in SOURCES:
            with (args.output_dir / f'snr_{group}.csv').open() as handle:
                values = [float(row['snr_db']) for row in csv.DictReader(handle) if row['status'] == 'ok']
            arrays.append(np.asarray([v for v in values if np.isfinite(v)]))
    else:
        from huggingface_hub import HfApi
        config = SNRConfig()
        protocol = {'collection': COLLECTION, 'estimator': config.to_dict(),
                    'method': 'spectrogram-ridge, noise-subtracted flanking-band SNR; one value per full audio row',
                    'limit_per_dataset': args.limit, 'workers': args.workers, 'datasets': {}}
        # A partial failed recomputation must not be presented as an earlier completed run.
        if protocol_path.exists():
            protocol_path.unlink()
        for group, (repo, subset, splits) in SOURCES.items():
            requested = args.pretraining_revision if group == 'pretraining' else args.classification_revision
            revision = HfApi().dataset_info(repo, revision=requested).sha
            print(f'Computing {group}: {repo}, config={subset}, revision={revision}, workers={args.workers}', flush=True)
            values, summary = compute_dataset(repo, subset, splits, revision, args.limit,
                                              args.output_dir / f'snr_{group}.csv', config, workers=args.workers)
            arrays.append(values)
            protocol['datasets'][group] = dict(repo=repo, config=subset, splits=splits,
                                               revision=revision, **summary)
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    plot_dataset_snr(arrays, args.output_dir, diagnostic=bool(protocol['limit_per_dataset']))
    print(f'Valid SNR values: pretraining={len(arrays[0])}, classification all={len(arrays[1])}')


if __name__ == '__main__':
    main()
