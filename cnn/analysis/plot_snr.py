"""Compute audio SNR from OpenWhistle's HF datasets and render manuscript panel F.

By default (``--f0 crepe``) SNR is measured around the CREPE F0 tracks written beforehand
by ``compute_f0_crepe.py``; ``--f0 ridge`` uses the spectrogram-ridge estimator instead
and needs no tracks.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from snr import SNRConfig, decode_audio, estimate_snr, estimate_snr_with_f0

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTION = 'https://huggingface.co/collections/dolphinteam/neurips26-openwhistle'
SOURCES = {
    'pretraining': ('dolphinteam/OpenWhistle-Pretraining', 'default', ('train', 'validation')),
    'classification_all': ('dolphinteam/OpenWhistle-Classification-Finetuning', 'unbalanced', ('train', 'validation', 'test')),
}
FIELDS = ('split', 'index', 'sampling_rate', 'duration_s', 'snr_db', 'status', 'n_stft_frames_used')
METHODS = {
    'crepe': ('CREPE F0 track (bottlenose-dolphin model), noise-subtracted flanking-band SNR pooled '
              'over confident frames; one value per full audio row'),
    'ridge': 'spectrogram-ridge, noise-subtracted flanking-band SNR; one value per full audio row',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'cnn/runs/snr')
    parser.add_argument('--f0', choices=tuple(METHODS), default='crepe',
                        help='F0 track around which SNR is measured (default: crepe).')
    parser.add_argument('--f0-dir', type=Path, default=REPO_ROOT / 'cnn/runs/f0_crepe',
                        help='Completed compute_f0_crepe.py outputs, used with --f0 crepe.')
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
    args.output_dir = args.output_dir.expanduser()
    args.f0_dir = args.f0_dir.expanduser()
    return args


def portable_path(path):
    """Path for provenance records: repository-relative, or only the name outside the checkout."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def stream_split(repo, config, split, revision, limit, batch_size):
    """Stream one split's undecoded audio in published row order; return (rows, count or None)."""
    from dataclasses import fields
    from datasets import Audio, load_dataset
    from datasets.packaged_modules.parquet.parquet import ParquetConfig
    from pyarrow.dataset import ParquetFragmentScanOptions
    options = dict(streaming=True, columns=['audio'], batch_size=batch_size)
    if 'fragment_scan_options' in {field.name for field in fields(ParquetConfig)}:  # datasets >= 4
        options['fragment_scan_options'] = ParquetFragmentScanOptions(pre_buffer=False)
    dataset = load_dataset(repo, name=config, split=split, revision=revision, **options)
    dataset = dataset.cast_column('audio', Audio(decode=False))
    if limit:
        dataset = dataset.take(limit)
    split_info = getattr(dataset.info, 'splits', None) if hasattr(dataset, 'info') else None
    total = split_info[split].num_examples if split_info and split in split_info else None
    if limit and total is not None:
        total = min(total, limit)
    return dataset, total


def compute_row(task):
    split, index, payload, config, track = task
    audio, fs = decode_audio(payload)
    if track is None:
        snr, status, frames = estimate_snr(audio, fs, config)
    else:
        if track['sampling_rate'] != fs:
            raise ValueError(f'F0 track of {split}/{index} is at {track["sampling_rate"]} Hz, its audio at {fs} Hz.')
        snr, status, frames = estimate_snr_with_f0(audio, fs, track['f0_time'], track['f0_hz'],
                                                   track['f0_conf'], config)
    return dict(split=split, index=index, sampling_rate=fs,
                duration_s=len(audio) / fs, snr_db=snr,
                status=status, n_stft_frames_used=frames)


def bounded_results(pool, function, tasks, max_pending):
    """Overlap reading and computation without queuing the whole audio corpus."""
    pending = deque()
    try:
        for task in tasks:
            pending.append(pool.submit(function, task))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()


def iter_tracks(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        yield from batch.to_pylist()


def next_track(tracks, split, index):
    """The F0 track of this audio row; tracks are stored in the same streamed order."""
    if tracks is None:
        return None
    track = next(tracks, None)
    if track is None or (track['split'], track['index']) != (split, index):
        raise ValueError(f'F0 tracks are out of step with the audio at {split}/{index}; recompute them.')
    return track


def load_f0_protocol(args):
    """Read and verify completed compute_f0_crepe.py outputs for an SNR run."""
    path = args.f0_dir / 'f0_protocol.json'
    if not path.is_file():
        raise ValueError(f'No CREPE tracks in {args.f0_dir}: run python cnn/analysis/compute_f0_crepe.py '
                         'first, or use --f0 ridge.')
    protocol = json.loads(path.read_text())
    tracked = protocol['limit_per_dataset']
    if tracked and not 0 < args.limit <= tracked:
        raise ValueError(f'The CREPE tracks cover only the first {tracked} rows per dataset; '
                         f'pass --limit {tracked} or less, or recompute them without --limit.')
    for group in SOURCES:
        if sha256_file(args.f0_dir / f'f0_{group}.parquet') != protocol['datasets'][group]['sha256']:
            raise ValueError(f'f0_{group}.parquet does not match {path}.')
    return protocol


def compute_dataset(repo, config, splits, revision, limit, output, snr_config, workers=1, tracks=None):
    values, counts, processed = [], Counter(), 0
    temporary = output.with_suffix('.csv.tmp')
    with ThreadPoolExecutor(max_workers=workers) as pool, temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for split in splits:
            dataset, total = stream_split(repo, config, split, revision, limit - processed if limit else 0, workers)
            tasks = ((split, index, row['audio'], snr_config, next_track(tracks, split, index))
                     for index, row in enumerate(dataset))
            records = bounded_results(pool, compute_row, tasks, max_pending=workers * 2)
            for record in tqdm(records, total=total, desc=f'{repo}/{split}', unit='clip'):
                writer.writerow(record)
                status, snr = record['status'], record['snr_db']
                counts[status] += 1
                processed += 1
                if status == 'ok' and np.isfinite(snr):
                    values.append(snr)
            if limit and processed >= limit:
                break
    if tracks is not None and not limit and next(tracks, None) is not None:
        raise ValueError(f'The F0 tracks of {repo} have more rows than its audio; recompute them.')
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
                                   group_labels=('Pretraining', 'Expert-\nannotated set'))
    ax.set_ylabel('SNR (dB)')
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
        if protocol['estimator'] != SNRConfig().to_dict() or protocol.get('f0', 'ridge') != args.f0:
            raise ValueError('Cached SNR protocol differs from the current estimator or --f0 choice.')
        for group in SOURCES:
            with (args.output_dir / f'snr_{group}.csv').open() as handle:
                values = [float(row['snr_db']) for row in csv.DictReader(handle) if row['status'] == 'ok']
            arrays.append(np.asarray([v for v in values if np.isfinite(v)]))
    else:
        from huggingface_hub import HfApi
        config = SNRConfig()
        f0_protocol = load_f0_protocol(args) if args.f0 == 'crepe' else None
        protocol = {'collection': COLLECTION, 'f0': args.f0, 'estimator': config.to_dict(),
                    'method': METHODS[args.f0],
                    'limit_per_dataset': args.limit, 'workers': args.workers, 'datasets': {}}
        if f0_protocol:
            protocol['f0_tracks'] = {'dir': portable_path(args.f0_dir), 'protocol': f0_protocol}
        # A partial failed recomputation must not be presented as an earlier completed run.
        if protocol_path.exists():
            protocol_path.unlink()
        for group, (repo, subset, splits) in SOURCES.items():
            requested = args.pretraining_revision if group == 'pretraining' else args.classification_revision
            revision = HfApi().dataset_info(repo, revision=requested).sha
            tracks = None
            if f0_protocol:
                source = f0_protocol['datasets'][group]
                if (source['repo'], source['config'], list(source['splits']), source['revision']) != \
                        (repo, subset, list(splits), revision):
                    raise ValueError(f'CREPE tracks for {group} come from {source["repo"]}@{source["revision"]} '
                                     f'({source["config"]}), but this run reads {repo}@{revision} ({subset}). '
                                     'Recompute them with compute_f0_crepe.py.')
                tracks = iter_tracks(args.f0_dir / f'f0_{group}.parquet')
            print(f'Computing {group}: {repo}, config={subset}, revision={revision}, f0={args.f0}, '
                  f'workers={args.workers}', flush=True)
            values, summary = compute_dataset(repo, subset, splits, revision, args.limit,
                                              args.output_dir / f'snr_{group}.csv', config,
                                              workers=args.workers, tracks=tracks)
            arrays.append(values)
            protocol['datasets'][group] = dict(repo=repo, config=subset, splits=splits,
                                               revision=revision, **summary)
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    plot_dataset_snr(arrays, args.output_dir, diagnostic=bool(protocol['limit_per_dataset']))
    print(f'Valid SNR values: pretraining={len(arrays[0])}, classification all={len(arrays[1])}')


if __name__ == '__main__':
    main()
