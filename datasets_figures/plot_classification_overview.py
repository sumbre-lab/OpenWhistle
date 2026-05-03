"""Render ``fig_classification_overview`` from the HF classification dataset plus CSV sidecars.

Loads ``dolphinteam/OpenWhistle-1.0-Classification-Finetuning`` (config ``all``) by default.
Whistle-sequence durations for panel B are read from the HF pretraining segments dataset when
possible; otherwise from ``data/audio_segment_durations.csv``.

SNR and inter-whistle panels use CSVs under ``data/`` (refresh with ``--refresh-data`` and the
env vars in ``datasets_figures.scripts.sidecars`` where applicable).

Examples::

    python datasets_figures/plot_classification_overview.py
    cd datasets_figures && python plot_classification_overview.py --refresh-data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset

from datasets_figures.scripts.classification_overview import plot_classification_overview_figure
from datasets_figures.scripts.paths import CLASSIFICATION_HF_ID, DATA_DIR, PRETRAINING_SEGMENTS_HF_ID
from datasets_figures.scripts.sidecars import refresh_classification_sidecars, segment_duration_seconds_from_hf


def _sequence_durations_s(*, data_dir: Path) -> np.ndarray | None:
    try:
        arr = segment_duration_seconds_from_hf(hf_id=PRETRAINING_SEGMENTS_HF_ID)
    except Exception as exc:
        print(f"[classification] HF pretraining segment durations failed ({exc}); trying CSV.")
        arr = np.array([], dtype=float)
    if len(arr):
        print(f"[classification] Sequence durations: {len(arr)} values from HF ({PRETRAINING_SEGMENTS_HF_ID}).")
        return arr
    seq_path = data_dir / "audio_segment_durations.csv"
    if not seq_path.is_file():
        return None
    sdf = pd.read_csv(seq_path)
    if "duration_s" not in sdf.columns:
        return None
    sdf["duration_s"] = pd.to_numeric(sdf["duration_s"], errors="coerce")
    sdf = sdf.dropna(subset=["duration_s"])
    if not len(sdf):
        return None
    print(f"[classification] Sequence durations: {len(sdf)} values from {seq_path.name}.")
    return sdf["duration_s"].to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render fig_classification_overview.")
    ap.add_argument("--hf-id", default=CLASSIFICATION_HF_ID, help="Hugging Face classification dataset id.")
    ap.add_argument("--config", default="all", help="Dataset config name (e.g. all).")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory with auxiliary CSVs (SNR, IWI, optional segment durations).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for fig_classification_overview.{png,pdf}.",
    )
    ap.add_argument(
        "--refresh-data",
        action="store_true",
        help="Regenerate HF-backed CSV sidecars (see scripts/sidecars.py docstring).",
    )
    args = ap.parse_args()

    if args.refresh_data:
        refresh_classification_sidecars(args.data_dir)

    print(f"[classification] Loading {args.hf_id!r} (config={args.config!r})")
    ds = load_dataset(args.hf_id, args.config, trust_remote_code=True)
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    splits = [s for s in ("train", "test") if s in ds]
    if not splits:
        raise SystemExit(f"No train/test splits in dataset: {list(ds.keys())}")

    first = ds[splits[0]]
    if "label" not in first.features:
        raise SystemExit("Dataset has no 'label' column.")
    label_names = list(first.features["label"].names)
    n_classes = len(label_names)

    parts = [ds[s].to_pandas() for s in splits]
    df_all = pd.concat(parts, ignore_index=True)

    n_train = len(ds["train"]) if "train" in ds else 0
    n_test = len(ds["test"]) if "test" in ds else 0
    print(f"[classification] rows: {', '.join(f'{s}={len(ds[s])}' for s in splits)}  classes={n_classes}")

    all_labels = df_all[["label"]]
    total_counts = all_labels["label"].value_counts().reindex(range(n_classes), fill_value=0)
    order = total_counts.sort_values(ascending=False).index
    total_sorted = total_counts.iloc[order].values
    names_sorted = [label_names[i] for i in order]

    durations = pd.to_numeric(df_all["duration"], errors="coerce").dropna().to_numpy()
    sequence_durations = _sequence_durations_s(data_dir=args.data_dir)

    snr_csv = args.data_dir / "snr_classification.csv"
    if not snr_csv.is_file():
        print(f"[classification] Warning: no {snr_csv.name}; panel F gold violin may be empty.")

    iwi = args.data_dir / "inter_detected_whistle_intervals.csv"
    if not iwi.is_file():
        print(
            f"[classification] Note: no {iwi.name}. Panel A placeholder — use "
            "`--refresh-data` with OPENWHISTLE_IWI_HF_IDS or copy predictions outputs."
        )

    plot_classification_overview_figure(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_classes=n_classes,
        label_names=label_names,
        total_sorted=total_sorted,
        names_sorted=names_sorted,
        n_train=n_train,
        n_test=n_test,
        df_all=df_all,
        durations=durations if len(durations) else np.array([]),
        sequence_durations=sequence_durations,
        snr_csv=snr_csv,
    )
    print("[classification] Done.")


if __name__ == "__main__":
    main()
