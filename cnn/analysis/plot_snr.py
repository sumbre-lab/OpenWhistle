"""Plot manuscript panel F: pretraining vs classification-all SNR from local CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir', type=Path,
        default=REPO_ROOT / 'cnn/runs/snr',
        help='Output directory for PNG and PDF.',
    )
    parser.add_argument(
        '--pretraining-snr-csv', type=Path,
        default=REPO_ROOT / 'datasets_figures/data/snr_detection_windows.csv',
    )
    parser.add_argument(
        '--classification-snr-csv', type=Path,
        default=REPO_ROOT / 'datasets_figures/data/snr_classification.csv',
    )
    return parser.parse_args(argv)


def plot_dataset_snr(args: argparse.Namespace) -> None:
    """Reuse panel F's data filtering and visual style with explicit dataset labels."""
    import seaborn as sns

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from datasets_figures.scripts.classification_overview import (
        draw_snr_detection_vs_gold_on_ax,
        load_snr_db_ok_only,
    )

    arrays = []
    for path in (args.pretraining_snr_csv, args.classification_snr_csv):
        if not path.is_file():
            raise FileNotFoundError(f'SNR CSV not found: {path}')
        values = load_snr_db_ok_only(path)
        if values is None:
            raise ValueError(f'No valid SNR values with status=ok in {path}')
        values = values[np.isfinite(values)]
        if len(values) < 2 or np.ptp(values) == 0:
            raise ValueError(f'Not enough SNR variation to plot a violin: {path}')
        arrays.append(values)

    figures_dir = args.output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    draw_snr_detection_vs_gold_on_ax(
        ax, arrays[0], arrays[1], sns,
        slim_violin=True,
        group_labels=('Pretraining', 'Classification\n(all)'),
    )
    fig.subplots_adjust(left=0.18, right=0.77, bottom=0.16, top=0.9)
    for extension in ('png', 'pdf'):
        path = figures_dir / f'snr_pretraining_vs_classification_all.{extension}'
        fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        print(f'Wrote {path}')
    plt.close(fig)
    print(f'Valid SNR values: pretraining={len(arrays[0])}, classification all={len(arrays[1])}')


def main(argv: list[str] | None = None) -> None:
    plot_dataset_snr(parse_args(argv))


if __name__ == '__main__':
    main()
