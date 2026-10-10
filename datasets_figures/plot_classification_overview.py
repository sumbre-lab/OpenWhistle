"""Render ``fig_classification_overview`` from the HF classification dataset plus CSV sidecars.

Loads ``dolphinteam/OpenWhistle-Classification-Finetuning`` (config ``all``) by default.
Whistle-sequence durations for panel B use ``data/audio_segment_durations.csv``
when available; otherwise they are read from the HF pretraining dataset.

Panel F reads the completed audio-derived SNR run under ``cnn/runs/snr/``.
Other auxiliary panels use CSVs under ``data/``.

Examples::

    python datasets_figures/plot_classification_overview.py
    cd datasets_figures && python plot_classification_overview.py --refresh-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from datasets import DatasetDict, IterableDatasetDict, load_dataset

from datasets_figures.scripts.classification_overview import load_snr_comparison, plot_classification_overview_figure
from datasets_figures.scripts.paths import CLASSIFICATION_HF_ID, DATA_DIR, PRETRAINING_SEGMENTS_HF_ID
from datasets_figures.scripts.sidecars import refresh_classification_sidecars, segment_duration_seconds_from_hf


def _sequence_durations_s(*, data_dir: Path) -> np.ndarray | None:
    seq_path = data_dir / "audio_segment_durations.csv"
    if seq_path.is_file():
        sdf = pd.read_csv(seq_path)
        if "duration_s" in sdf.columns:
            values = pd.to_numeric(sdf["duration_s"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values):
                print(f"[classification] Sequence durations: {len(values)} values from {seq_path.name}.")
                return values
    try:
        arr = segment_duration_seconds_from_hf(hf_id=PRETRAINING_SEGMENTS_HF_ID)
    except Exception as exc:
        print(f"[classification] HF pretraining segment durations failed ({exc}); panel B will be empty.")
        arr = np.array([], dtype=float)
    if len(arr):
        print(f"[classification] Sequence durations: {len(arr)} values from HF ({PRETRAINING_SEGMENTS_HF_ID}).")
        return arr
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Render fig_classification_overview.")
    ap.add_argument("--hf-id", default=CLASSIFICATION_HF_ID, help="Hugging Face classification dataset id.")
    ap.add_argument("--config", default="all", help="Dataset config name (e.g. all).")
    ap.add_argument(
        "--snr-dir", type=Path, default=_REPO / "cnn/runs/snr",
        help="Completed full-corpus plot_snr.py outputs, including snr_protocol.json.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory with auxiliary IWI and optional sequence-duration CSVs.",
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
    args.snr_dir = args.snr_dir.expanduser()
    args.data_dir = args.data_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    try:
        snr_pretraining, snr_classification, snr_protocol = load_snr_comparison(args.snr_dir)
    except (OSError, ValueError, KeyError) as exc:
        ap.error(f"Cannot use panel F inputs: {exc}. Run python cnn/analysis/plot_snr.py first.")
    source = snr_protocol["datasets"]["classification_all"]
    if (args.hf_id, args.config) != (source["repo"], source["config"]):
        ap.error("Classification dataset/config must match the SNR protocol for panel F.")

    if args.refresh_data:
        refresh_classification_sidecars(args.data_dir)

    iwi = args.data_dir / "inter_detected_whistle_intervals.csv"
    if not iwi.is_file():
        ap.error(f"Inter-whistle intervals CSV not found: {iwi}. "
                 "Use the supplied data/ directory or regenerate it with scripts/sidecars.py iwi.")

    print(f"[classification] Loading {args.hf_id!r} (config={args.config!r})")
    ds = load_dataset(args.hf_id, args.config, revision=source["revision"], streaming=True,
                      columns=["label", "name", "duration"])
    if not isinstance(ds, (DatasetDict, IterableDatasetDict)):
        ds = {"train": ds}

    # Concatenate every split (train / validation / test / …), not only train+test.
    _order = ("train", "validation", "test", "dev")
    splits = [s for s in _order if s in ds]
    splits.extend(sorted(s for s in ds.keys() if s not in splits))
    if not splits:
        raise SystemExit(f"No splits in dataset: {list(ds.keys())}")

    first = ds[splits[0]]
    if "label" not in first.features:
        raise SystemExit("Dataset has no 'label' column.")
    label_names = list(first.features["label"].names)
    n_classes = len(label_names)

    frames = {s: pd.DataFrame(ds[s]) for s in splits}
    df_all = pd.concat(list(frames.values()), ignore_index=True)

    n_train = len(frames["train"]) if "train" in frames else 0
    n_test = len(frames["test"]) if "test" in frames else 0
    print(f"[classification] rows: {', '.join(f'{s}={len(frames[s])}' for s in splits)}  classes={n_classes}")

    all_labels = df_all[["label"]]
    total_counts = all_labels["label"].value_counts().reindex(range(n_classes), fill_value=0)
    order = total_counts.sort_values(ascending=False).index
    total_sorted = total_counts.iloc[order].values
    names_sorted = [label_names[i] for i in order]

    durations = pd.to_numeric(df_all["duration"], errors="coerce").dropna().to_numpy()
    sequence_durations = _sequence_durations_s(data_dir=args.data_dir)

    if len(df_all) != source["processed"]:
        raise ValueError("Classification metadata and SNR run have different row counts.")
    print(f"[classification] Panel F: {len(snr_pretraining)} pretraining and "
          f"{len(snr_classification)} classification (all) valid SNR values.")

    if sequence_durations is None or len(sequence_durations) == 0:
        ap.error("No sequence durations available for panel B. "
                 "Provide audio_segment_durations.csv in --data-dir.")

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
        snr_pretraining=snr_pretraining,
        snr_classification=snr_classification,
    )
    inputs = [args.snr_dir / "snr_pretraining.csv",
              args.snr_dir / "snr_classification_all.csv",
              args.snr_dir / "snr_protocol.json"]
    sidecars = [args.data_dir / "inter_detected_whistle_intervals.csv",
                args.data_dir / "audio_segment_durations.csv"]
    provenance = {
        "classification": source,
        "snr_protocol": snr_protocol,
        "snr_inputs_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "auxiliary_csv_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sidecars if p.is_file()},
        "valid_snr_counts": {"pretraining": len(snr_pretraining), "classification_all": len(snr_classification)},
        "snr_display_range_db": [-10, 36],
    }
    (args.output_dir / "fig_classification_overview_sources.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print("[classification] Done.")


if __name__ == "__main__":
    main()
