"""HF helpers, CSV sidecar refresh, and inter-whistle export for the two overview plots.

Used by ``plot_* --refresh-data`` and optional CLI::

    python datasets_figures/scripts/sidecars.py audio-durations
    python datasets_figures/scripts/sidecars.py iwi --hf-id org/name
    python datasets_figures/scripts/sidecars.py classification-data --iwi-hf-id org/preds
    python datasets_figures/scripts/sidecars.py download --repo-id org/ds --filename f.csv --dest out.csv

Environment variables for ``refresh_classification_sidecars`` / ``refresh_dataset_overview_sidecars``:
``OPENWHISTLE_IWI_HF_IDS``, ``OPENWHISTLE_IWI_HF_SUBSET``, ``OPENWHISTLE_SNR_CLASSIFICATION_*``,
``OPENWHISTLE_SNR_DETECTION_*``, ``OPENWHISTLE_RECORDING_HOURS_*``, ``OPENWHISTLE_CONFUSION_MATRIX_*``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import Audio, DatasetDict, DownloadConfig, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from datasets_figures.scripts.paths import DATA_DIR, PRETRAINING_SEGMENTS_HF_ID


# ── Hub download + refresh (used by plot --refresh-data) ─────────────────────


def _hub_download_to(
    repo_id: str,
    filename: str,
    dest: Path,
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> None:
    print(f"[refresh] hf_hub_download {repo_id!r} / {filename!r} → {dest}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def refresh_audio_segment_durations(*, out: Path) -> None:
    write_audio_segment_durations_csv(out)


def refresh_inter_whistle_intervals(*, hf_ids: list[str], out: Path, subset: str | None = None) -> None:
    df = build_iwi_rows_from_hf_ids(hf_ids, split="train", subset=subset)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[refresh] Wrote {out} ({len(df)} rows)")


def refresh_classification_sidecars(data_dir: Path, *, include_audio: bool = True) -> None:
    if include_audio:
        refresh_audio_segment_durations(out=data_dir / "audio_segment_durations.csv")

    raw = (os.environ.get("OPENWHISTLE_IWI_HF_IDS") or "").strip()
    if raw:
        ids = [x.strip() for x in raw.split(",") if x.strip()]
        subset = (os.environ.get("OPENWHISTLE_IWI_HF_SUBSET") or "").strip() or None
        refresh_inter_whistle_intervals(
            hf_ids=ids,
            out=data_dir / "inter_detected_whistle_intervals.csv",
            subset=subset,
        )
    else:
        print("[refresh] OPENWHISTLE_IWI_HF_IDS unset; skipping inter-whistle CSV generation.")

    repo = (os.environ.get("OPENWHISTLE_SNR_CLASSIFICATION_REPO_ID") or "").strip()
    fn = (os.environ.get("OPENWHISTLE_SNR_CLASSIFICATION_FILENAME") or "").strip()
    if repo and fn:
        _hub_download_to(repo, fn, data_dir / "snr_classification.csv")
    else:
        print("[refresh] SNR classification hub env unset; keeping existing snr_classification.csv if any.")

    repo2 = (os.environ.get("OPENWHISTLE_SNR_DETECTION_REPO_ID") or "").strip()
    fn2 = (os.environ.get("OPENWHISTLE_SNR_DETECTION_FILENAME") or "").strip()
    if repo2 and fn2:
        _hub_download_to(repo2, fn2, data_dir / "snr_detection_windows.csv")
    else:
        print("[refresh] SNR detection hub env unset; keeping existing snr_detection_windows.csv if any.")


def refresh_dataset_overview_sidecars(*, recording_csv: Path, confusion_csv: Path) -> None:
    repo = (os.environ.get("OPENWHISTLE_RECORDING_HOURS_REPO_ID") or "").strip()
    fn = (os.environ.get("OPENWHISTLE_RECORDING_HOURS_FILENAME") or "").strip()
    if repo and fn:
        rtype = (os.environ.get("OPENWHISTLE_RECORDING_HOURS_REPO_TYPE") or "dataset").strip()
        rev = (os.environ.get("OPENWHISTLE_RECORDING_HOURS_REVISION") or "").strip() or None
        _hub_download_to(repo, fn, recording_csv, repo_type=rtype, revision=rev)
    else:
        print(
            "[refresh] OPENWHISTLE_RECORDING_HOURS_REPO_ID / OPENWHISTLE_RECORDING_HOURS_FILENAME unset; "
            "not downloading pretraining_recording_hours.csv."
        )

    repo_cm = (os.environ.get("OPENWHISTLE_CONFUSION_MATRIX_REPO_ID") or "").strip()
    fn_cm = (os.environ.get("OPENWHISTLE_CONFUSION_MATRIX_FILENAME") or "").strip()
    if repo_cm and fn_cm:
        rtype = (os.environ.get("OPENWHISTLE_CONFUSION_MATRIX_REPO_TYPE") or "dataset").strip()
        rev = (os.environ.get("OPENWHISTLE_CONFUSION_MATRIX_REVISION") or "").strip() or None
        _hub_download_to(repo_cm, fn_cm, confusion_csv, repo_type=rtype, revision=rev)
    else:
        print("[refresh] Confusion-matrix hub env unset; keeping existing CNN_test_confusion_matrix.csv if any.")


def download_hub_csv(
    *,
    repo_id: str,
    filename: str,
    dest: Path,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> None:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    print(f"[download] Copied hub file → {dest}")


# ── Pretraining segment durations (HF) ───────────────────────────────────────


def _duration_from_audio_field(audio_obj: dict) -> float | None:
    import soundfile as sf

    if not isinstance(audio_obj, dict):
        return None
    path = audio_obj.get("path")
    if not path:
        return None
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def compute_split_durations(ds, split: str) -> pd.DataFrame:
    n = len(ds)
    if "duration" in ds.column_names:
        durs = ds["duration"]
        return pd.DataFrame({"split": split, "duration_s": pd.to_numeric(durs, errors="coerce")})
    rows: list[dict] = []
    for ex in tqdm(ds, desc=f"duration/{split}", unit="seg", total=n):
        dur = _duration_from_audio_field(ex.get("audio", {}))
        rows.append({"split": split, "duration_s": dur})
    return pd.DataFrame(rows)


def _load_pretraining_segments(hf_id: str, subset: str | None):
    names: list[str | None] = []
    if subset:
        names.append(subset)
    if "default" not in (subset or ""):
        names.append("default")
    names.append(None)
    dedup: list[str | None] = []
    for n in names:
        if n not in dedup:
            dedup.append(n)

    last_exc: Exception | None = None
    for name in dedup:
        for local_only in (True, False):
            try:
                kw: dict = {"trust_remote_code": True}
                if local_only:
                    kw["download_config"] = DownloadConfig(local_files_only=True)
                if name is not None:
                    ds = load_dataset(hf_id, name, **kw)
                else:
                    ds = load_dataset(hf_id, **kw)
                if local_only:
                    print("[audio_segment_durations] Loaded from local HF cache.")
                return ds
            except Exception as exc:
                last_exc = exc
                continue
    raise RuntimeError(f"Could not load dataset {hf_id!r}: {last_exc}") from last_exc


def _segment_duration_frames(hf_id: str, subset: str | None) -> pd.DataFrame:
    print(f"[audio_segment_durations] HF dataset: {hf_id!r} subset={subset!r}")
    ds_root = _load_pretraining_segments(hf_id, subset)

    if isinstance(ds_root, DatasetDict):
        splits = list(ds_root.keys())
        ds_dict = ds_root
    else:
        splits = ["train"]
        ds_dict = {"train": ds_root}

    for s in splits:
        if "audio" in ds_dict[s].column_names:
            try:
                ds_dict[s] = ds_dict[s].cast_column("audio", Audio(decode=False))
            except Exception:
                pass

    parts: list[pd.DataFrame] = []
    for split in splits:
        part = compute_split_durations(ds_dict[split], split)
        part = part.dropna(subset=["duration_s"])
        parts.append(part)
        print(f"[audio_segment_durations] {split}: {len(part)} rows")

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["split", "duration_s"])


def segment_duration_seconds_from_hf(
    hf_id: str | None = None,
    subset: str | None = None,
) -> np.ndarray:
    hid = hf_id or PRETRAINING_SEGMENTS_HF_ID
    out_df = _segment_duration_frames(hid, subset)
    if out_df.empty:
        return np.array([], dtype=float)
    return out_df["duration_s"].to_numpy(dtype=float)


def write_audio_segment_durations_csv(
    out: Path,
    *,
    hf_id: str | None = None,
    subset: str | None = None,
) -> pd.DataFrame:
    hid = hf_id or PRETRAINING_SEGMENTS_HF_ID
    out_df = _segment_duration_frames(hid, subset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"[audio_segment_durations] Wrote {out} ({len(out_df)} rows)")
    return out_df


# ── Inter-whistle intervals from flat HF prediction tables ───────────────────

INTER_SEQUENCE_MERGE_MAX_GAP_S = 0.05
INTER_SEGMENT_EXPORT_MAX_GAP_S = 20.0


def _normalize_stem(stem: str) -> str:
    stem = re.sub(r"^\d{4}(?=Exp)", "", stem)
    stem = re.sub(r"(?i)(\d{4})(am|pm)", r"\1", stem)
    return stem


def _prediction_ip_fp_columns(pred_df: pd.DataFrame) -> tuple[str | None, str | None]:
    colmap = {str(c).strip().lower(): c for c in pred_df.columns}
    return colmap.get("initial_point"), colmap.get("finish_point")


def _intervals_from_predictions_df(pred_df: pd.DataFrame) -> list[tuple[float, float]]:
    if pred_df is None or pred_df.empty:
        return []
    ip_col, fp_col = _prediction_ip_fp_columns(pred_df)
    if ip_col is None or fp_col is None:
        return []
    ip = pd.to_numeric(pred_df[ip_col], errors="coerce")
    fp = pd.to_numeric(pred_df[fp_col], errors="coerce")
    ok = ip.notna() & fp.notna()
    if not ok.any():
        return []
    lo = np.minimum(ip[ok], fp[ok]).to_numpy(dtype=float)
    hi = np.maximum(ip[ok], fp[ok]).to_numpy(dtype=float)
    return list(zip(lo.tolist(), hi.tolist()))


def _merge_following_detection_intervals(
    intervals: list[tuple[float, float]],
    *,
    max_gap_s: float = INTER_SEQUENCE_MERGE_MAX_GAP_S,
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    cur_lo, cur_hi = intervals[0]
    merged: list[tuple[float, float]] = []
    for lo, hi in intervals[1:]:
        if lo <= cur_hi + max_gap_s:
            cur_hi = max(cur_hi, hi)
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    return merged


def _inter_merged_sequence_gaps_seconds(merged: list[tuple[float, float]]) -> list[float]:
    if len(merged) < 2:
        return []
    out: list[float] = []
    for i in range(len(merged) - 1):
        gap = float(merged[i + 1][0] - merged[i][1])
        if gap > 0:
            out.append(gap)
    return out


def _iwi_dict_rows_for_intervals(
    gaps: list[float],
    *,
    subfolder: str,
    folder_name: str,
) -> list[dict]:
    return [
        {
            "subfolder": subfolder,
            "folder_name": folder_name,
            "inter_sequence_interval_s": g,
        }
        for g in gaps
        if 0 < g < INTER_SEGMENT_EXPORT_MAX_GAP_S
    ]


def _collect_iwi_from_hf_predictions_dataframe(
    df_hf: pd.DataFrame,
    hf_id: str,
    iwi_seen_keys: set[str],
) -> list[dict]:
    if df_hf.empty or "file_name" not in df_hf.columns:
        return []
    ip_col, fp_col = _prediction_ip_fp_columns(df_hf)
    if ip_col is None or fp_col is None:
        return []

    stems = df_hf["file_name"].astype(str).str.replace(r"\.wav$", "", regex=True, case=False)
    df_work = df_hf.copy()
    df_work["_stem"] = stems

    out: list[dict] = []
    for _stem, grp in df_work.groupby("_stem", sort=False):
        folder_name = str(_stem)
        folder_key = _normalize_stem(folder_name)
        if folder_key in iwi_seen_keys:
            continue
        iwi_seen_keys.add(folder_key)
        ivals = _intervals_from_predictions_df(grp)
        merged = _merge_following_detection_intervals(ivals)
        gaps = _inter_merged_sequence_gaps_seconds(merged)
        out.extend(_iwi_dict_rows_for_intervals(gaps, subfolder=hf_id, folder_name=folder_name))
    return out


def _load_predictions_split(hf_id: str, *, split: str, subset: str | None):
    names: list[str | None] = []
    if subset:
        names.append(subset)
    if (subset or "") != "default":
        names.append("default")
    names.append(None)
    dedup: list[str | None] = []
    for n in names:
        if n not in dedup:
            dedup.append(n)

    last_exc: Exception | None = None
    for name in dedup:
        try:
            if name is not None:
                return load_dataset(hf_id, name, split=split, trust_remote_code=True)
            return load_dataset(hf_id, split=split, trust_remote_code=True)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Could not load {hf_id!r} split={split!r}: {last_exc}") from last_exc


def build_iwi_rows_from_hf_ids(
    hf_ids: Iterable[str],
    *,
    split: str = "train",
    subset: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    seen: set[str] = set()
    for hf_id in hf_ids:
        hid = hf_id.strip()
        if not hid:
            continue
        print(f"[iwi] Loading {hid} split={split!r} subset={subset!r}")
        ds = _load_predictions_split(hid, split=split, subset=subset)
        df_hf = ds.to_pandas()
        rows.extend(_collect_iwi_from_hf_predictions_dataframe(df_hf, hid, seen))
    return pd.DataFrame(
        rows,
        columns=["subfolder", "folder_name", "inter_sequence_interval_s"],
    )


# ── Optional CLI (replaces separate scripts under ``scripts/``) ──────────────


def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _cli_main() -> None:
    _ensure_repo_on_path()
    ap = argparse.ArgumentParser(description="Sidecar CSV helpers for datasets_figures plots.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_audio = sub.add_parser("audio-durations", help="Write audio_segment_durations.csv from HF pretraining.")
    p_audio.add_argument("--hf-id", default=PRETRAINING_SEGMENTS_HF_ID)
    p_audio.add_argument("--subset", default=None)
    p_audio.add_argument("--out", type=Path, default=DATA_DIR / "audio_segment_durations.csv")

    p_iwi = sub.add_parser("iwi", help="Write inter_detected_whistle_intervals.csv from HF prediction dataset(s).")
    p_iwi.add_argument("--hf-id", action="append", dest="hf_ids", metavar="ID", required=True)
    p_iwi.add_argument("--split", default="train")
    p_iwi.add_argument("--subset", default=None)
    p_iwi.add_argument("--out", type=Path, default=DATA_DIR / "inter_detected_whistle_intervals.csv")

    p_cls = sub.add_parser(
        "classification-data",
        help="Refresh classification plot CSV sidecars (same as plot_classification_overview --refresh-data).",
    )
    p_cls.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p_cls.add_argument("--iwi-hf-id", action="append", dest="iwi_hf_ids", metavar="ID")
    p_cls.add_argument("--skip-audio-durations", action="store_true")

    p_dl = sub.add_parser("download", help="Download one file from a Hub repo.")
    p_dl.add_argument("--repo-id", required=True)
    p_dl.add_argument("--filename", required=True)
    p_dl.add_argument("--dest", type=Path, required=True)
    p_dl.add_argument("--repo-type", choices=("dataset", "model", "space"), default="dataset")
    p_dl.add_argument("--revision", default=None)

    args = ap.parse_args()

    if args.cmd == "audio-durations":
        write_audio_segment_durations_csv(args.out, hf_id=args.hf_id, subset=args.subset)
    elif args.cmd == "iwi":
        df = build_iwi_rows_from_hf_ids(args.hf_ids, split=args.split, subset=args.subset)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[iwi] Wrote {args.out} ({len(df)} rows)")
    elif args.cmd == "classification-data":
        saved = os.environ.get("OPENWHISTLE_IWI_HF_IDS")
        try:
            if args.iwi_hf_ids:
                os.environ["OPENWHISTLE_IWI_HF_IDS"] = ",".join(args.iwi_hf_ids)
            refresh_classification_sidecars(
                args.data_dir,
                include_audio=not args.skip_audio_durations,
            )
        finally:
            if args.iwi_hf_ids:
                if saved is None:
                    os.environ.pop("OPENWHISTLE_IWI_HF_IDS", None)
                else:
                    os.environ["OPENWHISTLE_IWI_HF_IDS"] = saved
    elif args.cmd == "download":
        download_hub_csv(
            repo_id=args.repo_id,
            filename=args.filename,
            dest=args.dest,
            repo_type=args.repo_type,
            revision=args.revision,
        )


if __name__ == "__main__":
    _cli_main()
