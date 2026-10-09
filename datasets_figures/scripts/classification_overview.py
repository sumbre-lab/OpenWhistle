"""Drawing helpers for ``fig_classification_overview`` (from DolphinDataset ``scripts/classification/figures.py``)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Same cap as ``INTER_SEGMENT_EXPORT_MAX_GAP_S`` in DolphinDataset ``scripts/pretraining/stats.py``.
INTER_SEGMENT_HISTOGRAM_X_MAX_S = 20.0

# ── Palette ───────────────────────────────────────────────────────────────────
# One colour per individual; train = solid, test = lighter tint
CLASS_COLORS = [
    "#4E79A7",   # NSW (first class, e.g. NSW_1)
    "#F28E2B",   # SW_Luna
    "#59A14F",   # SW_Nana
    "#E15759",   # SW_Neo
    "#B07AA1",   # SW_Nikita
    "#76B7B2",   # SW_Yosefa
]

# Extra points added to panel typography in ``fig_classification_overview`` only.
CLASSIFICATION_OVERVIEW_FONT_EXTRA = 2


def _panel_label(ax, letter: str, *, fontsize: float = 20) -> None:
    """Bold panel letter at top-left, slightly outside the axes."""
    ax.text(
        -0.04,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
    )


# SNR quality bands (dB) for classification_old stats CSV
SNR_THRESHOLDS = (3.0, 6.0, 10.0)
SNR_BAND_LABELS = (
    ("poor", "< 3 dB", "#f5cac9"),
    ("fair", "3–6 dB", "#fde9ce"),
    ("good", "6–10 dB", "#d5e8d4"),
    ("very good", "≥ 10 dB", "#c5d9e8"),
)

# Duration histogram: fixed display range (counts beyond xmax omitted from bars)
DURATION_HIST_XMAX_SEC = 2.0
DURATION_BIN_WIDTH_SEC = 0.05

# Expedition naming:
#   - Exp_{day}_{Mon}_{year}_{time} (e.g. Exp_11_Dec_2019_1145am)
#   - DD-MM-YY-HHMM (e.g. 13-11-19-1221 → 13 Nov 2019 12:21)


def parse_expedition_date(name: object) -> datetime | None:
    """Parse recording date/time from dataset ``name``, or None if unknown."""
    s = str(name).strip()

    parts = s.split("_")
    if len(parts) >= 5 and parts[0] == "Exp":
        try:
            return datetime.strptime(f"{int(parts[1])} {parts[2]} {int(parts[3])}", "%d %b %Y")
        except ValueError:
            pass

    # Rare alternate: day-month-2digit_year-time as HHMM (24h)
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2})-(\d{4})", s)
    if not m:
        return None
    try:
        day, month, yy, hhmm = (int(m.group(i)) for i in range(1, 5))
        year = 2000 + yy if yy < 100 else yy
        hour, minute = hhmm // 100, hhmm % 100
        if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def _coast_from_label_name(name: object) -> str | None:
    """Return 'NSW' or 'SW' from whistle label_name, else None."""
    s = str(name or "").strip()
    if s.startswith("NSW_"):
        return "NSW"
    if s.startswith("SW_"):
        return "SW"
    return None


def _whistle_type_display_name(name: object) -> str:
    """Strip ``SW_`` only for axis labels; keep ``NSW_…`` (e.g. ``SW_Luna`` → ``Luna``)."""
    s = str(name or "").strip()
    if s.startswith("SW_"):
        return s[3:]
    return s

def load_snr_db_ok_only(snr_csv: Path):
    """All ``status == ok`` rows with valid ``snr_db`` (no coast filter)."""
    import pandas as pd

    if not snr_csv.is_file():
        return None
    df = pd.read_csv(snr_csv)
    ok = df["status"].astype(str) == "ok"
    s = pd.to_numeric(df.loc[ok, "snr_db"], errors="coerce").dropna()
    if len(s) == 0:
        return None
    return s.to_numpy(dtype=float)


def load_inter_detected_whistle_intervals_csv(csv_path: Path) -> np.ndarray:
    """Load ``inter_sequence_interval_s`` from pretraining stats CSV (< 20 s rows only)."""
    import pandas as pd

    if not csv_path.is_file():
        return np.array([], dtype=float)
    df = pd.read_csv(csv_path)
    col = "inter_sequence_interval_s"
    if col not in df.columns:
        return np.array([], dtype=float)
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return np.array([], dtype=float)
    return s.to_numpy(dtype=float)


def draw_inter_whistle_interval_hist_on_ax(
    ax,
    iwis: np.ndarray,
    sns,
    *,
    compact: bool = False,
    bin_width_s: float = 0.5,
    font_extra: int = 0,
) -> None:
    """Histogram of inter-whistles gaps (pretraining ``inter_detected_whistle_intervals.csv``)."""
    fe = max(0, int(font_extra))
    fs_t = (11 if compact else 13) + fe
    fs_l = (8 if compact else 11) + fe
    if len(iwis) == 0:
        ax.set_title("Inter-whistles interval (no data)", fontsize=fs_t, fontweight="bold")
        ax.text(
            0.5,
            0.5,
            "Missing or empty\ninter_detected_whistle_intervals.csv\n"
            "(run scripts/pretraining/stats.py)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9 + fe,
            color="#666666",
        )
        return

    iwis = np.maximum(iwis, 0.0)
    x_min = 0.0
    x_max = INTER_SEGMENT_HISTOGRAM_X_MAX_S
    bw = float(bin_width_s) if bin_width_s > 0 else 0.5
    n_bins = max(1, int(round((x_max - x_min) / bw)))
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    clip = iwis[(iwis >= x_min) & (iwis < x_max)]
    median = float(np.median(iwis))
    mean = float(np.mean(iwis))

    sns.histplot(
        clip,
        bins=bin_edges,
        color="#4E79A7",
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    ax.axvline(median, color="#222222", linewidth=1.1, linestyle="-", label=f"Median: {median:.2f}s")
    ax.axvline(mean, color="#222222", linewidth=1.1, linestyle="--", label=f"Mean: {mean:.2f}s")
    ax.set_title(
        "Inter-whistle segments interval",
        fontsize=fs_t,
        fontweight="bold",
        pad=6 if compact else 10,
    )
    ax.set_xlabel("Interval (seconds)", fontsize=fs_l)
    ax.set_ylabel("Count", fontsize=fs_l)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    ax.legend(
        framealpha=0.85,
        fontsize=(7 if compact else 9) + fe,
        loc="upper right",
    )
    if fe:
        ax.tick_params(axis="both", labelsize=fs_l)
    # (intentionally no n= annotation for this panel)


def draw_class_distribution_on_ax(
    ax,
    n_classes: int,
    label_names: list,
    total_sorted: np.ndarray,
    names_sorted: list,
    n_train: int,
    n_test: int,
    *,
    compact: bool = False,
    ylabel: str = "Count",
) -> None:
    import seaborn as sns

    fs_t = 11 if compact else 13
    fs_l = 8 if compact else 11
    x = np.arange(n_classes)
    width = 0.55
    ymax = float(total_sorted.max()) if len(total_sorted) else 1.0
    total_bar = float(np.sum(total_sorted)) if len(total_sorted) else 1.0

    for i, (count, name) in enumerate(zip(total_sorted, names_sorted)):
        orig_idx = label_names.index(name)
        col = CLASS_COLORS[orig_idx % len(CLASS_COLORS)]
        ax.bar(x[i], count, width, color=col, edgecolor="white", linewidth=0.5)
        pct = 100.0 * float(count) / total_bar if total_bar > 0 else 0.0
        fs_n = 8 if compact else 10
        ax.text(
            x[i],
            count + ymax * 0.01,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=fs_n,
            fontweight="normal",
            color="#111111",
        )
        ax.text(
            x[i],
            count + ymax * 0.078,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=fs_n,
            fontweight="bold",
            color="#111111",
        )

    display_names = [_whistle_type_display_name(n) for n in names_sorted]
    ax.set_xticks(x)
    _rot = 35 if compact else 15
    ax.set_xticklabels(display_names, fontsize=fs_l, rotation=_rot, ha="right")
    ax.set_ylabel(ylabel, fontsize=fs_l)
    ax.set_xlabel("Whistle type", fontsize=fs_l, labelpad=4 if compact else 8)
    ax.set_title("Class distribution", fontsize=fs_t, fontweight="bold", pad=6 if compact else 12)
    ax.set_xlim(-0.5, n_classes - 0.5)
    ax.set_ylim(0, ymax * (1.32 if compact else 1.30) if ymax > 0 else 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    total = int(total_sorted.sum())
    ax.text(
        0.02,
        0.97,
        f"n={total}",
        transform=ax.transAxes,
        fontsize=7 if compact else 9,
        va="top",
        color="#555555",
    )


def draw_class_distribution_coast_on_ax(
    ax,
    n_classes: int,
    label_names: list,
    total_sorted: np.ndarray,
    names_sorted: list,
    n_train: int,
    n_test: int,
    *,
    compact: bool = False,
    ylabel: str = "Count",
    font_extra: int = 0,
) -> None:
    """Class distribution bars: NSW vs SW use the same two colours as the coast pie chart."""
    import matplotlib.patches as mpatches
    import seaborn as sns

    fe = max(0, int(font_extra))
    fs_t = (11 if compact else 13) + fe
    fs_l = (8 if compact else 11) + fe
    x = np.arange(n_classes)
    width = 0.55
    ymax = float(total_sorted.max()) if len(total_sorted) else 1.0
    total_bar = float(np.sum(total_sorted)) if len(total_sorted) else 1.0
    col_nsw = CLASS_COLORS[0]
    col_sw = CLASS_COLORS[2]
    sw_sum = sum(
        float(c) for c, n in zip(total_sorted, names_sorted) if _coast_from_label_name(n) == "SW"
    )
    nsw_sum = sum(
        float(c) for c, n in zip(total_sorted, names_sorted) if _coast_from_label_name(n) == "NSW"
    )
    sw_pct = 100.0 * sw_sum / total_bar if total_bar > 0 else 0.0
    nsw_pct = 100.0 * nsw_sum / total_bar if total_bar > 0 else 0.0

    for i, (count, name) in enumerate(zip(total_sorted, names_sorted)):
        coast = _coast_from_label_name(name)
        col = col_nsw if coast == "NSW" else col_sw
        ax.bar(x[i], count, width, color=col, edgecolor="white", linewidth=0.5)
        pct = 100.0 * float(count) / total_bar if total_bar > 0 else 0.0
        fs_n = (8 if compact else 10) + fe
        ax.text(
            x[i],
            count + ymax * 0.01,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=fs_n,
            fontweight="normal",
            color="#111111",
        )
        ax.text(
            x[i],
            count + ymax * 0.078,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=fs_n,
            fontweight="bold",
            color="#111111",
        )

    display_names = [_whistle_type_display_name(n) for n in names_sorted]
    ax.set_xticks(x)
    _rot = 35 if compact else 15
    ax.set_xticklabels(display_names, fontsize=fs_l, rotation=_rot, ha="right")
    ax.set_ylabel(ylabel, fontsize=fs_l)
    ax.set_xlabel("Whistle type", fontsize=fs_l, labelpad=4 if compact else 8)
    ax.set_title("Class distribution", fontsize=fs_t, fontweight="bold", pad=6 if compact else 12)
    ax.set_xlim(-0.5, n_classes - 0.5)
    ax.set_ylim(0, ymax * (1.32 if compact else 1.30) if ymax > 0 else 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    total = int(total_sorted.sum())
    ax.text(
        0.02,
        0.97,
        f"n={total}",
        transform=ax.transAxes,
        fontsize=(7 if compact else 9) + fe,
        va="top",
        color="#555555",
    )
    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor=col_sw,
                edgecolor="white",
                label=f"Signature Whistle (SW) " + rf"$\mathbf{{{sw_pct:.1f}\%}}$",
            ),
            mpatches.Patch(
                facecolor=col_nsw,
                edgecolor="white",
                label=f"Non-Signature Whistle (NSW) " + rf"$\mathbf{{{nsw_pct:.1f}\%}}$",
            ),
        ],
        loc="upper right",
        fontsize=(7 if compact else 9) + fe,
        framealpha=0.92,
    )
    if fe:
        ax.tick_params(axis="y", labelsize=fs_l)


def draw_duration_histogram_on_ax(
    ax,
    durations: np.ndarray,
    sns,
    *,
    compact: bool = False,
    small_panel: bool = False,
    title: str | None = None,
    xlim_max: float = DURATION_HIST_XMAX_SEC,
    bin_width: float = DURATION_BIN_WIDTH_SEC,
    font_extra: int = 0,
) -> None:
    fe = max(0, int(font_extra))
    if len(durations) == 0:
        ax.set_title("Duration (no data)", fontsize=11 + fe, fontweight="bold")
        return

    fs_t = (11 if compact else 13) + fe
    fs_l = (8 if compact else 11) + fe
    if small_panel:
        fs_t = max(9, fs_t - 2)
        fs_l = max(7, fs_l - 1)
    median = float(np.median(durations))
    mean = float(np.mean(durations))
    xmax = float(xlim_max)
    n_bins = max(1, int(round(xmax / bin_width)))
    bin_edges = np.linspace(0.0, xmax, n_bins + 1)

    sns.histplot(
        durations,
        bins=bin_edges,
        color="#4E79A7",
        edgecolor="white",
        linewidth=0.5 if compact else 0.6,
        ax=ax,
    )
    ax.axvline(median, color="#222222", linewidth=1.1, linestyle="-", label=f"Med {median:.2f}s")
    ax.axvline(mean, color="#222222", linewidth=1.1, linestyle="--", label=f"Mean {mean:.2f}s")
    ax.set_title(
        title if title is not None else "Whistle duration",
        fontsize=fs_t,
        fontweight="bold",
        pad=4 if small_panel else (6 if compact else 10),
    )
    ax.set_xlabel("Duration (s)", fontsize=fs_l)
    ax.set_ylabel("Count", fontsize=fs_l)
    ax.set_xlim(0.0, xmax)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    if small_panel:
        ax.tick_params(axis="both", labelsize=fs_l)
    elif fe:
        ax.tick_params(axis="both", labelsize=fs_l)
    leg_fs = (6 if small_panel else (7 if compact else 10)) + fe
    ax.legend(framealpha=0.85, fontsize=leg_fs, loc="upper right")
    # (intentionally no n= annotation for this panel)


def draw_nsw_sw_pie_on_ax(ax, df_all, label_names: list, *, compact: bool = False) -> None:
    """Pie chart of NSW vs SW counts from integer ``label`` and ``label_names``."""
    fs_t = 11 if compact else 13
    nsw = sw = 0
    for lab in df_all["label"].dropna():
        name = label_names[int(lab)]
        c = _coast_from_label_name(name)
        if c == "NSW":
            nsw += 1
        elif c == "SW":
            sw += 1

    sizes = [nsw, sw]
    if sum(sizes) == 0:
        ax.set_title("NSW / SW (no data)", fontsize=fs_t, fontweight="bold")
        return

    colors = [CLASS_COLORS[0], CLASS_COLORS[2]]
    explode = (0.02, 0.02)
    ax.pie(
        sizes,
        explode=explode,
        labels=["NSW", "SW"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8 if compact else 10},
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.set_title("NSW vs SW", fontsize=fs_t, fontweight="bold", pad=6 if compact else 10)
    ax.annotate(
        f"n = {sum(sizes)}",
        xy=(0.5, -0.06),
        xycoords="axes fraction",
        ha="center",
        fontsize=7 if compact else 9,
        color="#555555",
    )


# Time range for expedition-date plots (inclusive day bounds)
WHISTLE_TIME_START = "2019-11-01"
WHISTLE_TIME_END = "2020-03-31"
# Finer than monthly: ISO weeks starting Monday
WHISTLE_TIME_BIN_FREQ = "W-MON"


def draw_monthly_whistles_on_ax(
    ax,
    df_all,
    *,
    compact: bool = False,
    title: str | None = None,
    xtick_every: int = 1,
    font_extra: int = 0,
) -> None:
    """Bar counts per week (``W-MON``) from Nov 2019 through Mar 2020, from ``name``.

    ``xtick_every``: show an x tick/label every Nth week (1 = every week).
    """
    import pandas as pd
    import seaborn as sns

    fe = max(0, int(font_extra))
    # Slightly smaller than other non-compact panels to keep dense x-ticks readable
    fs_t = (11 if compact else 12) + fe
    fs_l = (7 if compact else 9) + fe
    t0 = pd.Timestamp(WHISTLE_TIME_START)
    t1 = pd.Timestamp(WHISTLE_TIME_END) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    dts: list[pd.Timestamp] = []
    for name in df_all["name"]:
        raw = parse_expedition_date(name)
        if raw is None:
            continue
        d = pd.Timestamp(raw)
        if t0 <= d <= t1:
            dts.append(d)

    default_title = "Whistles by week (Nov 2019 – Mar 2020)"
    if not dts:
        ax.set_title(
            f"{title} (no data)" if title else "Whistles over time (no dates in window)",
            fontsize=fs_t,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.5,
            "No whistles in Nov 2019 – Mar 2020",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11 + fe,
        )
        return

    ser = (
        pd.DataFrame({"t": dts})
        .set_index("t")
        .sort_index()
        .resample(WHISTLE_TIME_BIN_FREQ, label="left", closed="left")
        .size()
    )
    ys = ser.to_numpy(dtype=float)
    idx = ser.index
    x = np.arange(len(ys))
    w = 0.82 if len(ys) > 18 else 0.88
    ax.bar(x, ys, width=w, color="#4E79A7", edgecolor="white", linewidth=0.4)
    step = max(1, int(xtick_every))
    tick_x = x[::step]
    if compact:
        tick_lbl = [d.strftime("%d/%m/%y") for d in idx]
        rot, ha = 72, "right"
    else:
        tick_lbl = [d.strftime("%d/%m/%y") for d in idx]
        rot, ha = 45, "right"
    tick_lbl_stepped = tick_lbl[::step]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_lbl_stepped, fontsize=fs_l, rotation=rot, ha=ha)
    ax.set_ylabel("Whistle count", fontsize=fs_l + 1)
    ax.set_xlabel("Week starting (Mon)", fontsize=fs_l + 1)
    ax.set_title(
        title if title is not None else default_title,
        fontsize=fs_t,
        fontweight="bold",
        pad=6 if compact else 10,
    )
    ymax = float(np.max(ys)) if len(ys) else 1.0
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
    ax.set_xlim(-0.5, len(ys) - 0.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    if fe:
        ax.tick_params(axis="y", labelsize=fs_l + 1)
    # (intentionally no n= annotation for this panel)


def draw_snr_violin_on_ax(ax, sub, sns, mpatches, *, compact: bool = False) -> bool:
    """
    Vertical SNR violins by coast on ``ax``. Returns False if no data (axes cleared).
    """
    fs_t = 11 if compact else 13
    fs_l = 8 if compact else 11
    snr_title_fs = max(9, fs_t - 2)
    if sub is None or len(sub) == 0:
        ax.set_title("SNR distribution", fontsize=snr_title_fs, fontweight="bold")
        ax.text(
            0.5,
            0.5,
            "SNR sidecar data unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="#666666",
        )
        ax.set_axis_off()
        return False

    snr_all = sub["snr_db"].to_numpy()
    y_lo, y_hi = -10.0, 36.0
    t1, t2, t3 = SNR_THRESHOLDS

    spans = [
        (y_lo, t1, SNR_BAND_LABELS[0][2]),
        (t1, t2, SNR_BAND_LABELS[1][2]),
        (t2, t3, SNR_BAND_LABELS[2][2]),
        (t3, y_hi, SNR_BAND_LABELS[3][2]),
    ]
    for lo, hi, color in spans:
        ax.axhspan(lo, hi, facecolor=color, edgecolor="none", alpha=0.92, zorder=0)

    coast_order = ["NSW", "SW"]
    palette = {"NSW": CLASS_COLORS[0], "SW": CLASS_COLORS[2]}
    sns.violinplot(
        data=sub,
        x="coast",
        y="snr_db",
        order=coast_order,
        hue="coast",
        hue_order=coast_order,
        palette=palette,
        dodge=False,
        ax=ax,
        inner="quart",
        cut=0,
        linewidth=0.8 if compact else 0.9,
        saturation=0.82,
        width=0.68 if compact else 0.72,
        legend=False,
        zorder=2,
    )

    thr_labels = {t1: "fair", t2: "good", t3: "very good"}
    for tv in (t1, t2, t3):
        ax.axhline(tv, color="#333333", linewidth=1.0, linestyle="--", zorder=3)
        ax.text(
            1.03,
            tv,
            thr_labels.get(tv, ""),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7 if compact else 9,
            fontweight="bold",
            color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
            zorder=4,
            clip_on=False,
        )

    ax.set_ylim(y_lo, y_hi)
    ax.set_title("SNR distribution", fontsize=snr_title_fs, fontweight="bold", pad=6 if compact else 10)
    ax.set_xlabel("Population", fontsize=fs_l)
    ax.set_ylabel("SNR (dB)", fontsize=fs_l)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    ax.text(
        0.02,
        0.98,
        f"n={len(snr_all)}",
        transform=ax.transAxes,
        va="top",
        fontsize=7 if compact else 9,
        color="#555555",
        zorder=4,
    )
    return True


def draw_snr_detection_vs_gold_on_ax(
    ax,
    snr_detection: np.ndarray | None,
    snr_gold: np.ndarray | None,
    sns,
    *,
    compact: bool = False,
    small_panel: bool = False,
    slim_violin: bool = False,
    font_extra: int = 0,
    group_labels: tuple[str, str] = ("All", "Gold set"),
) -> bool:
    """SNR violins: all detection windows (``All``) vs gold classification set."""
    fe = max(0, int(font_extra))
    fs_t = (11 if compact else 13) + fe
    fs_l = (8 if compact else 11) + fe
    if small_panel:
        fs_t = max(10, fs_t - 1)
        fs_l = max(9, fs_l - 1)
    snr_title_fs = max(9, fs_t - 2)
    # slim_violin: narrower violins only; do not shrink text
    det = np.asarray(snr_detection, dtype=float) if snr_detection is not None and len(snr_detection) else np.array([])
    gold = np.asarray(snr_gold, dtype=float) if snr_gold is not None and len(snr_gold) else np.array([])

    if len(det) == 0 and len(gold) == 0:
        ax.set_title("SNR", fontsize=snr_title_fs, fontweight="bold")
        ax.text(
            0.5,
            0.5,
            "Need snr_detection_windows.csv\nand/or snr_classification.csv",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10 + fe,
            color="#666666",
        )
        ax.set_axis_off()
        return False

    y_lo, y_hi = -10.0, 36.0
    t1, t2, t3 = SNR_THRESHOLDS

    spans = [
        (y_lo, t1, SNR_BAND_LABELS[0][2]),
        (t1, t2, SNR_BAND_LABELS[1][2]),
        (t2, t3, SNR_BAND_LABELS[2][2]),
        (t3, y_hi, SNR_BAND_LABELS[3][2]),
    ]
    for lo, hi, color in spans:
        ax.axhspan(lo, hi, facecolor=color, edgecolor="none", alpha=0.92, zorder=0)

    detection_label, gold_label = group_labels
    rows: list[dict] = []
    if len(det):
        rows.extend({"snr_db": float(v), "group": detection_label} for v in det)
    if len(gold):
        rows.extend({"snr_db": float(v), "group": gold_label} for v in gold)
    import pandas as pd

    plot_df = pd.DataFrame(rows)
    order = [detection_label, gold_label]
    order = [g for g in order if g in plot_df["group"].unique()]
    # Same NSW / SW base hues as the coast pie (CLASS_COLORS[0] vs [2])
    palette = {gold_label: CLASS_COLORS[0], detection_label: CLASS_COLORS[2]}

    if slim_violin:
        v_width = 0.34 if small_panel else (0.48 if compact else 0.56)
    elif small_panel:
        v_width = 0.52
    else:
        v_width = 0.72 if compact else 0.75
    sns.violinplot(
        data=plot_df,
        x="group",
        y="snr_db",
        order=order,
        hue="group",
        hue_order=order,
        palette={k: palette[k] for k in order},
        dodge=False,
        ax=ax,
        inner="quart",
        cut=0,
        linewidth=0.8 if compact else 0.9,
        saturation=0.82,
        width=v_width,
        legend=False,
        zorder=2,
    )

    thr_fs = (8 if small_panel else (7 if compact else 9)) + fe
    thr_x = 1.04 if slim_violin else 1.03
    thr_labels = {t1: "fair", t2: "good", t3: "very good"}
    for tv in (t1, t2, t3):
        ax.axhline(tv, color="#333333", linewidth=1.0, linestyle="--", zorder=3)
        ax.text(
            thr_x,
            tv,
            thr_labels.get(tv, ""),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=thr_fs,
            fontweight="bold",
            color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.8 if slim_violin else 1.5),
            zorder=4,
            clip_on=False,
        )

    ax.set_ylim(y_lo, y_hi)
    ax.set_title(
        "SNR",
        fontsize=snr_title_fs,
        fontweight="bold",
        pad=6 if (small_panel or slim_violin) else (6 if compact else 10),
    )
    ax.set_xlabel("", fontsize=fs_l)
    ax.set_ylabel("SNR (dB)", fontsize=fs_l)
    ax.tick_params(axis="both", labelsize=fs_l)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    # (intentionally no n= annotation for this panel)
    return True



def plot_classification_overview_figure(
    *,
    output_dir: Path,
    data_dir: Path,
    n_classes: int,
    label_names: list,
    total_sorted: np.ndarray,
    names_sorted: list,
    n_train: int,
    n_test: int,
    df_all: pd.DataFrame,
    durations: np.ndarray,
    sequence_durations: np.ndarray | None,
    snr_csv: Path,
) -> None:
    """3×3 grid: row0 A,B,C; spacer row; row2 wide D (class), E, slim F (SNR)."""
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    from datasets_figures.scripts.pretraining_hf_snippets import (
        draw_segment_duration_hist_ax,
        load_snr_ok_detection_subframe,
    )

    iwi_csv = data_dir / "inter_detected_whistle_intervals.csv"
    iwis = load_inter_detected_whistle_intervals_csv(iwi_csv)
    snr_gold_all = load_snr_db_ok_only(snr_csv) if snr_csv.is_file() else None
    snr_det: np.ndarray | None = None
    det_df = load_snr_ok_detection_subframe(data_dir)
    if det_df is not None and len(det_df):
        snr_det = det_df["snr_db"].to_numpy(dtype=float)

    gold_span_title = "Expert-annotated dataset span\n(Nov 2019 - Mar 2020)"
    ox = CLASSIFICATION_OVERVIEW_FONT_EXTRA
    panel_lab_fs = 20 + ox

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.0, 0.05, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.32,
        wspace=0.22,
        left=0.05,
        right=0.99,
        top=0.97,
        bottom=0.05,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[2, :],
        width_ratios=[1.75, 1.0, 0.65],
        wspace=0.20,
    )
    ax_d = fig.add_subplot(gs_bot[0, 0])
    ax_e = fig.add_subplot(gs_bot[0, 1])
    ax_f = fig.add_subplot(gs_bot[0, 2])

    draw_inter_whistle_interval_hist_on_ax(
        ax_a, iwis, sns, compact=False, bin_width_s=1.0, font_extra=ox
    )

    if sequence_durations is not None and len(sequence_durations) > 0:
        draw_segment_duration_hist_ax(
            ax_b, sequence_durations, compact=False, font_extra=ox
        )
        ax_b.set_title(
            "Whistle sequence duration",
            fontsize=13 + ox,
            fontweight="bold",
            pad=10,
        )
    else:
        ax_b.text(
            0.5,
            0.5,
            "No sequence durations\n(audio_segment_durations.csv in data dir)",
            ha="center",
            va="center",
            transform=ax_b.transAxes,
            fontsize=10 + ox,
            color="#555555",
        )
        ax_b.set_axis_off()

    draw_monthly_whistles_on_ax(
        ax_c,
        df_all,
        compact=False,
        title=gold_span_title,
        xtick_every=2,
        font_extra=ox,
    )
    draw_class_distribution_coast_on_ax(
        ax_d,
        n_classes,
        label_names,
        total_sorted,
        names_sorted,
        n_train,
        n_test,
        compact=False,
        font_extra=ox,
    )
    if len(durations) > 0:
        draw_duration_histogram_on_ax(
            ax_e,
            durations,
            sns,
            compact=False,
            title="Whistle duration",
            font_extra=ox,
        )
    else:
        ax_e.text(
            0.5,
            0.5,
            "No whistle durations in dataset",
            ha="center",
            va="center",
            transform=ax_e.transAxes,
            fontsize=10 + ox,
            color="#555555",
        )
        ax_e.set_axis_off()
    draw_snr_detection_vs_gold_on_ax(
        ax_f,
        snr_det,
        snr_gold_all,
        sns,
        compact=False,
        small_panel=False,
        slim_violin=True,
        font_extra=ox,
    )

    for ax, lab in (
        (ax_a, "A"),
        (ax_b, "B"),
        (ax_c, "C"),
        (ax_d, "D"),
        (ax_e, "E"),
        (ax_f, "F"),
    ):
        _panel_label(ax, lab, fontsize=panel_lab_fs)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"fig_classification_overview.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[classification] Saved → {out}")
    plt.close(fig)
