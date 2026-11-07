"""Track whistle F0 with CREPE on OpenWhistle's HF audio: step 1 of the SNR figure.

Writes, under cnn/runs/f0_crepe/, one Parquet file per dataset group (split, index,
sampling rate, duration, status and the float32 CREPE track f0_time / f0_hz / f0_conf,
in published row order) and f0_protocol.json. ``plot_snr.py`` then measures SNR around
these tracks on the CPU. A GPU is strongly recommended.
"""
from __future__ import annotations

import crepe_f0  # noqa: I001  (first: torch must load before datasets/pyarrow)

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from plot_snr import COLLECTION, REPO_ROOT, SOURCES, bounded_results, sha256_file, stream_split
from snr import decode_audio

COLUMNS = ('split', 'index', 'sampling_rate', 'duration_s', 'status', 'f0_time', 'f0_hz', 'f0_conf')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'cnn/runs/f0_crepe')
    parser.add_argument('--limit', type=int, default=0, help='First N audio rows per dataset for a diagnostic (0 = all).')
    parser.add_argument('--pretraining-revision', default='main')
    parser.add_argument('--classification-revision', default='main')
    parser.add_argument('--weights', type=Path, default=None,
                        help='Local copy of the CREPE weights (sha256-checked). Default: download the pinned '
                             f'file once to {crepe_f0.WEIGHTS_CACHE}.')
    parser.add_argument('--device', default=None, help="Torch device (default: 'cuda' if available, else 'cpu').")
    parser.add_argument('--batch-size', type=int, default=64, help='CREPE frames per network batch (default: 64).')
    parser.add_argument('--workers', type=int, default=4,
                        help='Clips tracked concurrently; noise reduction and resampling run on the CPU (default: 4).')
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit must be nonnegative.')
    if args.workers < 1 or args.batch_size < 1:
        parser.error('--workers and --batch-size must be positive.')
    args.output_dir = args.output_dir.expanduser()
    return args


def track_row(task):
    split, index, payload, model, device, config, batch_size = task
    audio, fs = decode_audio(payload)
    record = dict(split=split, index=index, sampling_rate=fs, duration_s=len(audio) / fs)
    if not len(audio):
        empty = np.zeros(0, dtype=np.float32)
        return dict(record, status='empty_audio', f0_time=empty, f0_hz=empty, f0_conf=empty)
    track = crepe_f0.estimate_f0(model, device, audio, fs, config, batch_size)
    return dict(record, status='ok', **{name: values.astype(np.float32)
                                        for name, values in zip(('f0_time', 'f0_hz', 'f0_conf'), track)})


def compute_group(repo, subset, splits, revision, limit, output, model, device, config, batch_size, workers):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([('split', pa.string()), ('index', pa.int64()), ('sampling_rate', pa.int32()),
                        ('duration_s', pa.float64()), ('status', pa.string()),
                        ('f0_time', pa.list_(pa.float32())), ('f0_hz', pa.list_(pa.float32())),
                        ('f0_conf', pa.list_(pa.float32()))])
    counts, processed, buffer = Counter(), 0, []
    temporary = output.with_suffix('.parquet.tmp')

    def flush(writer):
        writer.write_table(pa.Table.from_pydict({name: [row[name] for row in buffer] for name in COLUMNS},
                                                schema=schema))
        buffer.clear()

    with ThreadPoolExecutor(max_workers=workers) as pool, \
            pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
        for split in splits:
            dataset, total = stream_split(repo, subset, split, revision, limit - processed if limit else 0, workers)
            tasks = ((split, index, row['audio'], model, device, config, batch_size)
                     for index, row in enumerate(dataset))
            for record in tqdm(bounded_results(pool, track_row, tasks, max_pending=workers * 2),
                               total=total, desc=f'CREPE {repo}/{split}', unit='clip'):
                buffer.append(record)
                counts[record['status']] += 1
                processed += 1
                if len(buffer) >= 256:
                    flush(writer)
            if limit and processed >= limit:
                break
        if buffer:
            flush(writer)
    temporary.replace(output)
    return {'processed': processed, 'status_counts': dict(counts), 'sha256': sha256_file(output)}


def main(argv=None):
    args = parse_args(argv)
    from huggingface_hub import HfApi
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / 'f0_protocol.json'
    # A partial failed recomputation must not be presented as an earlier completed run.
    if protocol_path.exists():
        protocol_path.unlink()
    model, device = crepe_f0.load_model(args.weights, args.device)
    config = crepe_f0.CrepeConfig()
    protocol = {
        'collection': COLLECTION,
        'method': 'CREPE F0 tracks, bottlenose-dolphin model of mim-team/bioacoustic_F0_estimation',
        'crepe': config.to_dict(),
        'weights': {'url': crepe_f0.WEIGHTS_URL, 'sha256': crepe_f0.WEIGHTS_SHA256},
        'device': crepe_f0.torch.cuda.get_device_name(device) if str(device).startswith('cuda') else str(device),
        'batch_size': args.batch_size,
        'versions': {name: version(name) for name in ('torch', 'torchcrepe', 'noisereduce', 'librosa', 'numpy')},
        'limit_per_dataset': args.limit,
        'datasets': {},
    }
    for group, (repo, subset, splits) in SOURCES.items():
        requested = args.pretraining_revision if group == 'pretraining' else args.classification_revision
        revision = HfApi().dataset_info(repo, revision=requested).sha
        print(f'Tracking {group}: {repo}, config={subset}, revision={revision}, device={device}', flush=True)
        summary = compute_group(repo, subset, splits, revision, args.limit, args.output_dir / f'f0_{group}.parquet',
                                model, device, config, args.batch_size, args.workers)
        protocol['datasets'][group] = dict(repo=repo, config=subset, splits=list(splits), revision=revision, **summary)
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    print(f'Wrote {protocol_path}')


if __name__ == '__main__':
    main()
