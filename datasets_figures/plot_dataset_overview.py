"""Render ``fig_dataset_overview`` (2×2 pretraining overview).

Recording-hour statistics are taken from ``data/pretraining_recording_hours.csv`` (not fully
reconstructible from the public segment HF set). CNN evaluation provides panel D's confusion matrix.

With ``--refresh-data``, optional Hub downloads run when the environment variables documented in
``datasets_figures.scripts.sidecars`` are set.

Examples::

    python datasets_figures/plot_dataset_overview.py
    cd datasets_figures && python plot_dataset_overview.py --refresh-data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets_figures.scripts.paths import DATA_DIR
from datasets_figures.scripts.pretraining_dataset_overview import load_and_clean, plot_combined
from datasets_figures.scripts.sidecars import refresh_dataset_overview_sidecars


def main() -> None:
    ap = argparse.ArgumentParser(description="Render fig_dataset_overview (2×2 pretraining overview).")
    ap.add_argument(
        "--recording-csv",
        type=Path,
        default=DATA_DIR / "pretraining_recording_hours.csv",
        help="Path to pretraining_recording_hours.csv.",
    )
    ap.add_argument(
        "--confusion-csv",
        type=Path,
        default=_REPO / "cnn/runs/reports/test_confusion_matrix.csv",
        help="CNN evaluation matrix CSV (defaults to cnn/runs/reports/test_confusion_matrix.csv).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for fig_dataset_overview.{png,pdf}.",
    )
    ap.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh optional Hub CSV sidecars (see scripts/sidecars.py docstring).",
    )
    args = ap.parse_args()
    args.recording_csv = args.recording_csv.expanduser()
    args.confusion_csv = args.confusion_csv.expanduser()
    args.output_dir = args.output_dir.expanduser()

    if args.refresh_data:
        refresh_dataset_overview_sidecars(
            recording_csv=args.recording_csv,
            confusion_csv=args.confusion_csv,
        )

    if not args.recording_csv.is_file():
        raise SystemExit(f"Recording-hours CSV not found: {args.recording_csv}")

    print(f"[dataset_overview] Loading {args.recording_csv}")
    df = load_and_clean(args.recording_csv)
    print(f"[dataset_overview] {len(df)} unique recordings after load_and_clean.")
    total_voc_h = float(df["detection_duration"].sum()) / 3600.0
    print(f"[dataset_overview] Total vocalisation duration: {total_voc_h:.2f} h")

    if not args.confusion_csv.is_file():
        ap.error(f"Confusion matrix not found: {args.confusion_csv}. "
                 "Run python cnn/train.py --test-only --no-wandb-enabled first, "
                 "or provide --confusion-csv.")

    plot_combined(df, args.output_dir, confusion_matrix_csv=args.confusion_csv)


if __name__ == "__main__":
    main()
