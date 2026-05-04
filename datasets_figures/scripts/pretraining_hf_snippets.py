"""Small excerpts from DolphinDataset ``scripts/pretraining_hf/figures.py`` for classification overview."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_snr_ok_detection_subframe(data_dir: Path) -> pd.DataFrame | None:
    """Rows with ``status == ok`` and numeric ``snr_db`` from ``snr_detection_windows.csv`` in *data_dir*."""
    csv_path = data_dir / "snr_detection_windows.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    ok = df["status"].astype(str) == "ok"
    sub = df.loc[ok, ["snr_db"]].copy()
    sub["snr_db"] = pd.to_numeric(sub["snr_db"], errors="coerce")
    sub = sub.dropna(subset=["snr_db"])
    if len(sub) == 0:
        return None
    return sub


def draw_segment_duration_hist_ax(
    ax: plt.Axes,
    durations: np.ndarray,
    *,
    compact: bool = False,
    font_extra: int = 0,
) -> None:
    fe = max(0, int(font_extra))
    median = float(np.median(durations))
    mean = float(np.mean(durations))
    x_min = 4.0
    x_max = 50.0
    bin_width = 1.0
    bin_edges = np.arange(x_min, x_max + bin_width, bin_width)
    durations_clip = durations[(durations >= x_min) & (durations <= x_max)]
    sns.histplot(
        durations_clip,
        bins=bin_edges,
        color="#4E79A7",
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    ax.axvline(median, color="#222222", linewidth=1.3, linestyle="-", label=f"Median: {median:.2f}s")
    ax.axvline(mean, color="#222222", linewidth=1.3, linestyle="--", label=f"Mean: {mean:.2f}s")
    title_fs = (11 if compact else 13) + fe
    label_fs = (9 if compact else 11) + fe
    ax.set_title(
        "Audio segment duration histogram",
        fontsize=title_fs,
        fontweight="bold",
        pad=8 if compact else 10,
    )
    ax.set_xlabel("Duration (seconds)", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    ax.set_xlim(x_min, x_max)
    ax.legend(framealpha=0.85, fontsize=(9 if compact else 10) + fe, loc="upper right")
    if fe:
        ax.tick_params(axis="both", labelsize=label_fs)
