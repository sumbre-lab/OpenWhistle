"""Fixed-recipe CNN learning curve for the reviewer rebuttal.

This file is intentionally self-contained inside ``cnn/learning_curve``. It reuses
the published CNN trainer without changing the main training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CNN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CNN_DIR.parent
if str(CNN_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_DIR))

from train import TorchTrainingRun  # noqa: E402
from utils.config import TrainConfig  # noqa: E402
from utils.data import (  # noqa: E402
    build_split_summary_payload,
    make_data_loader,
    make_training_dataset,
)
from utils.runtime_utils import extract_session_id  # noqa: E402

DEFAULT_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
DEFAULT_SEEDS = (7, 17, 27)
METRICS = ('accuracy', 'f1', 'precision', 'recall')


def fraction_tag(fraction: float) -> str:
    return f'{fraction:.4f}'.rstrip('0').rstrip('.').replace('.', 'p')


def run_dir(output_root: Path, fraction: float, seed: int) -> Path:
    return output_root / f'fraction_{fraction_tag(fraction)}' / f'seed_{seed}'


def select_nested_session_subset(
    train_dataset,
    fraction: float,
    seed: int,
):
    """Select a nested fraction of independent sessions for one seed."""
    if 'recording' not in train_dataset.column_names:
        raise ValueError('The training split must contain a `recording` column.')

    row_sessions = [
        extract_session_id(recording)
        for recording in train_dataset['recording']
    ]
    all_sessions = sorted(set(row_sessions))
    if not all_sessions:
        raise ValueError('The training split contains no sessions.')

    rng = np.random.default_rng(seed)
    ordered_sessions = list(rng.permutation(all_sessions))
    retained_count = max(1, math.ceil(fraction * len(all_sessions)))
    retained_sessions = set(ordered_sessions[:retained_count])
    retained_indices = [
        index
        for index, session_id in enumerate(row_sessions)
        if session_id in retained_sessions
    ]
    subset = train_dataset.select(retained_indices)
    if set(int(label) for label in subset['label']) != {0, 1}:
        raise ValueError(
            'The selected sessions do not contain both classes. '
            'Increase the fraction or use another seed.'
        )
    return subset, retained_count, len(all_sessions)


class RebuttalTrainingRun(TorchTrainingRun):
    """Published trainer with training-session subsampling added after setup."""

    def __init__(self, config: TrainConfig, session_fraction: float) -> None:
        self.session_fraction = session_fraction
        super().__init__(config)

    def setup(self) -> None:
        super().setup()
        if self.prepared_data is None:
            raise RuntimeError('CNN data preparation failed.')

        full_train = self.prepared_data.split_datasets[self.config.train_split]
        subset, retained_count, total_count = select_nested_session_subset(
            full_train,
            self.session_fraction,
            self.config.random_state,
        )
        self.prepared_data.split_datasets[self.config.train_split] = subset
        self.prepared_data.split_summary = build_split_summary_payload(
            self.prepared_data.split_datasets,
            self.config,
        )
        self.prepared_data.train_dataset = make_training_dataset(
            self.prepared_data.split_datasets,
            self.config.train_split,
            self.config,
            self.prepared_data.spectrogram_cache,
        )
        self.prepared_data.train_loader = make_data_loader(
            self.prepared_data.train_dataset,
            shuffle=True,
            config=self.config,
            pin_memory=self.pin_memory,
        )
        print(
            'Rebuttal training subset: '
            f'fraction={self.session_fraction:g}  '
            f'seed={self.config.random_state}  '
            f'sessions={retained_count}/{total_count}  '
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run the fixed-recipe CNN learning curve on nested fractions of '
            'training sessions.'
        )
    )
    parser.add_argument('--fractions', type=float, nargs='+', default=DEFAULT_FRACTIONS)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('cnn/learning_curve/results'),
    )
    parser.add_argument(
        '--dataset-source',
        default='dolphinteam/OpenWhistle-CNN',
    )
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--window-seconds', type=float, default=0.4)
    parser.add_argument(
        '--cpu-only',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--use-amp',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List planned runs without training or writing results.',
    )
    args = parser.parse_args(argv)
    args.fractions = sorted(set(args.fractions))
    args.seeds = list(dict.fromkeys(args.seeds))
    if not args.fractions or any(not 0 < value <= 1 for value in args.fractions):
        parser.error('--fractions values must all be in (0, 1].')
    if not args.seeds:
        parser.error('--seeds requires at least one value.')
    if args.window_seconds <= 0:
        parser.error('--window-seconds must be positive.')
    if not args.output_root.is_absolute():
        args.output_root = REPO_ROOT / args.output_root
    return args


def experiment_settings(
    args: argparse.Namespace,
    fraction: float,
    seed: int,
) -> dict[str, object]:
    return {
        'session_fraction': fraction,
        'seed': seed,
        'dataset_source': args.dataset_source,
        'train_input_source': 'spectrogram',
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'learning_rate': args.learning_rate,
        'num_workers': args.num_workers,
        'use_amp': args.use_amp,
        'cpu_only': args.cpu_only,
    }


def make_train_config(
    args: argparse.Namespace,
    fraction: float,
    seed: int,
) -> TrainConfig:
    destination = run_dir(args.output_root, fraction, seed)
    return replace(
        TrainConfig.defaults_from_env(),
        dataset_source=args.dataset_source,
        train_input_source='spectrogram',
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        random_state=seed,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        cpu_only=args.cpu_only,
        wandb_enabled=False,
        models_dir=str(destination / 'models'),
        figs_dir=str(destination / 'figures'),
        reports_dir=str(destination / 'reports'),
    )


def run_experiments(args: argparse.Namespace) -> None:
    for fraction in args.fractions:
        for seed in args.seeds:
            destination = run_dir(args.output_root, fraction, seed)
            summary_path = destination / 'reports' / 'run_summary.json'
            metadata_path = destination / 'run_metadata.json'
            expected = experiment_settings(args, fraction, seed)

            if summary_path.exists() and not args.overwrite:
                if not metadata_path.exists():
                    raise ValueError(f'Missing run metadata: {metadata_path}')
                with metadata_path.open(encoding='utf-8') as handle:
                    observed = json.load(handle)
                if observed != expected:
                    raise ValueError(
                        f'Existing run settings differ in {metadata_path}. '
                        'Use --overwrite to regenerate it.'
                    )
                print(f'Skipping completed run: fraction={fraction:g}, seed={seed}')
                continue

            print(f'Planned run: fraction={fraction:g}, seed={seed}')
            if args.dry_run:
                continue
            destination.mkdir(parents=True, exist_ok=True)
            with metadata_path.open('w', encoding='utf-8') as handle:
                json.dump(expected, handle, indent=2)
            config = make_train_config(args, fraction, seed)
            RebuttalTrainingRun(config, fraction).run()


def load_rows(args: argparse.Namespace) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for fraction in args.fractions:
        for seed in args.seeds:
            destination = run_dir(args.output_root, fraction, seed)
            summary_path = destination / 'reports' / 'run_summary.json'
            metadata_path = destination / 'run_metadata.json'
            if not summary_path.exists():
                print(f'Incomplete run omitted: fraction={fraction:g}, seed={seed}')
                continue
            with summary_path.open(encoding='utf-8') as handle:
                summary = json.load(handle)
            with metadata_path.open(encoding='utf-8') as handle:
                metadata = json.load(handle)
            expected = experiment_settings(args, fraction, seed)
            if metadata != expected:
                raise ValueError(
                    f'Run settings differ in {metadata_path}; use matching CLI '
                    'arguments or regenerate with --overwrite.'
                )

            train_summary = summary['split_summary']['train']
            test_metrics = summary['metrics'].get('test')
            if test_metrics is None:
                raise ValueError(f'Test metrics missing from {summary_path}')
            train_rows = int(train_summary['rows'])
            row: dict[str, int | float] = {
                'session_fraction': fraction,
                'seed': seed,
                'train_sessions': int(train_summary['sessions']),
                'train_rows': train_rows,
                'train_window_hours': train_rows * args.window_seconds / 3600,
                'best_epoch': int(summary['best_epoch']),
            }
            for metric in METRICS:
                row[f'test_{metric}'] = float(test_metrics[metric])
            rows.append(row)
    return rows


def aggregate_rows(
    rows: list[dict[str, int | float]],
) -> list[dict[str, int | float]]:
    grouped: dict[float, list[dict[str, int | float]]] = defaultdict(list)
    for row in rows:
        grouped[float(row['session_fraction'])].append(row)

    aggregated: list[dict[str, int | float]] = []
    for fraction, group in sorted(grouped.items()):
        result: dict[str, int | float] = {
            'session_fraction': fraction,
            'repeats': len(group),
        }
        fields = (
            'train_sessions',
            'train_rows',
            'train_window_hours',
            *(f'test_{metric}' for metric in METRICS),
        )
        for field in fields:
            values = np.asarray([float(row[field]) for row in group])
            result[f'{field}_mean'] = float(values.mean())
            result[f'{field}_std'] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        aggregated.append(result)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curve(
    output_root: Path,
    rows: list[dict[str, int | float]],
    aggregated: list[dict[str, int | float]],
) -> None:
    plt.rcParams.update(
        {
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    x = np.arange(len(aggregated))
    y = np.asarray([100 * float(row['test_f1_mean']) for row in aggregated])
    yerr = np.asarray([100 * float(row['test_f1_std']) for row in aggregated])
    fractions = [100 * float(row['session_fraction']) for row in aggregated]
    sessions = [round(float(row['train_sessions_mean'])) for row in aggregated]
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker='o',
        color='#075985',
        markerfacecolor='#075985',
        markeredgecolor='white',
        markeredgewidth=0.8,
        markersize=7,
        linewidth=2.2,
        capsize=3.5,
        zorder=3,
    )
    ax.annotate(
        f'{y[0]:.2f}% F1\n−{y[-1] - y[0]:.2f} pp vs. full data',
        xy=(x[0], y[0]),
        xytext=(12, -39),
        textcoords='offset points',
        fontsize=9,
        color='#075985',
    )
    ax.annotate(
        f'{y[-1]:.2f}% F1',
        xy=(x[-1], y[-1]),
        xytext=(-8, 14),
        textcoords='offset points',
        ha='right',
        fontsize=9,
        color='#075985',
    )

    tick_labels = [
        f'{fraction:g}%\n{session} sessions'
        for fraction, session in zip(fractions, sessions)
    ]
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel('Training-set size')
    ax.set_ylabel('Test F1 (%)')
    fig.suptitle(
        'Near-full CNN performance with 5% of training sessions',
        x=0.12,
        y=0.96,
        ha='left',
        fontsize=12,
    )
    fig.text(
        0.125,
        0.89,
        'Mean test F1 ± SD across three seeds; fixed training recipe',
        fontsize=9,
        color='#4b5563',
    )
    lower = float((y - yerr).min()) - 0.15
    upper = float((y + yerr).max()) + 0.25
    ax.set_ylim(lower, upper)
    ax.grid(axis='y', color='#d1d5db', linewidth=0.7, alpha=0.65)
    ax.spines[['top', 'right']].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.80)
    for suffix in ('png', 'pdf'):
        fig.savefig(output_root / f'learning_curve_test_f1.{suffix}', dpi=300)
    plt.close(fig)


def write_protocol(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        'fractions': args.fractions,
        'seeds': args.seeds,
        'dataset_source': args.dataset_source,
        'subsampling_unit': 'session',
        'subsets_nested_within_seed': True,
        'validation_and_test_fixed': True,
        'primary_metric': 'test_f1',
        'hyperparameter_policy': 'fixed published training recipe for all sizes',
        'window_seconds': args.window_seconds,
    }
    with (args.output_root / 'protocol.json').open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def aggregate_and_plot(args: argparse.Namespace) -> None:
    rows = load_rows(args)
    if not rows:
        raise RuntimeError('No completed rebuttal runs were found.')
    aggregated = aggregate_rows(rows)
    write_csv(args.output_root / 'learning_curve_runs.csv', rows)
    write_csv(args.output_root / 'learning_curve_summary.csv', aggregated)
    plot_learning_curve(args.output_root, rows, aggregated)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.plot_only:
        if not args.dry_run:
            write_protocol(args)
        run_experiments(args)
    if not args.dry_run:
        aggregate_and_plot(args)


if __name__ == '__main__':
    main()
