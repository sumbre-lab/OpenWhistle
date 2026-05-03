"""Subset of DolphinDataset ``scripts/pretraining/figures.py`` for ``fig_dataset_overview`` only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from datasets_figures.scripts.paths import DATA_DIR


def load_confusion_matrix_csv(path: Path) -> tuple[np.ndarray, list[str], list[str]] | None:
    """Return (matrix, row_class_names, col_class_names)."""
    if not path.is_file():
        return None
    raw = pd.read_csv(path)
    if raw.shape[0] < 1 or raw.shape[1] < 2:
        return None

    df = raw.copy()
    first = str(df.columns[0])
    if first.startswith("Unnamed") and df.shape[1] >= 3:
        df = df.rename(columns={df.columns[0]: "row_class"})
        row_labels = df["row_class"].astype(str).tolist()
        df = df.drop(columns=["row_class"])
    else:
        row_labels = []

    drop_lbl = [c for c in df.columns if str(c).strip().lower() == "label"]
    if drop_lbl:
        df = df.drop(columns=drop_lbl)

    num = df.apply(pd.to_numeric, errors="coerce")
    if num.isna().any().any():
        num = df.select_dtypes(include=[np.number])
    if num.shape[0] < 1 or num.shape[1] < 1:
        return None

    mat = num.to_numpy(dtype=float)
    col_names = [str(c) for c in num.columns]

    if len(row_labels) == mat.shape[0]:
        row_names = row_labels
    elif mat.shape[0] == mat.shape[1] == len(col_names):
        row_names = list(col_names)
    else:
        row_names = [str(i) for i in range(mat.shape[0])]

    return mat, row_names, col_names


def draw_confusion_matrix_ax(
    ax: plt.Axes,
    cm: np.ndarray,
    row_names: list[str],
    col_names: list[str],
    *,
    small_panel: bool = False,
) -> None:
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.where(row_sums > 0, (cm / row_sums) * 100.0, 0.0)

    ann = {"size": 8 if small_panel else 10}
    cbar = {
        "shrink": 0.62 if small_panel else 0.82,
        "label": "Percent (%)",
        "aspect": 14 if small_panel else 20,
    }
    sns.heatmap(
        cm_pct,
        annot=True,
        annot_kws=ann,
        fmt=".1f",
        cmap="Blues",
        ax=ax,
        xticklabels=col_names,
        yticklabels=row_names,
        cbar_kws=cbar,
        linewidths=0.35 if small_panel else 0.5,
        linecolor="#e8e8e8",
        square=small_panel,
    )
    xlab, ylab, ttl = (8, 8, 10) if small_panel else (10, 10, 12)
    ax.set_xlabel("Predicted", fontsize=xlab)
    ax.set_ylabel("True", fontsize=ylab)
    ax.set_title("CNN test\nconfusion matrix", fontsize=ttl, fontweight="bold", pad=4 if small_panel else 8)
    ax.tick_params(axis="both", labelsize=8 if small_panel else 9)


# Backwards-compatible name (same as ``DATA_DIR``).
FIGURES_DATA_DIR = DATA_DIR

MONTH_ORDER  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUM    = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}

BAR_COLOR = "#5B7FA3"

# Daily schedule: human-interaction / public-hours band (9–16 UTC)
_SCHEDULE_BAND_COLOR_DEFAULT = "#d5e8d4"
_SCHEDULE_BAND_ALPHA_DEFAULT = 0.45
_SCHEDULE_BAND_COLOR_OVERVIEW = "#6eb85c"
_SCHEDULE_BAND_ALPHA_OVERVIEW = 0.52
_FEEDING_VLINE_COLOR = "#5c3d2e"
_FEEDING_HOURS = (10, 11, 12, 13, 14)

RECORDING_Y_BREAK    = 400
VOCALISATION_Y_BREAK = 5.0

# ── Dolphin group size ────────────────────────────────────────────────────────
DOLPHIN_INITIAL    = 5
DOLPHIN_COLOR      = "#E07B39"
DOLPHIN_DEPARTURES = [          # (name, exact date of departure)
    ("Neo",    pd.Timestamp("2021-04-16")),
    ("Yosefa", pd.Timestamp("2021-09-01")),
    ("Luna",   pd.Timestamp("2024-04-27")),
]


def _format_hours_compact(v: float) -> str:
    """Human-readable hours for annotations (comma for large values)."""
    if v >= 100:
        return f"{v:,.0f}"
    if abs(v - round(v)) < 1e-6:
        return f"{v:.0f}"
    return f"{v:.1f}"


def _monthly_hours_stats_annotation(values: np.ndarray) -> str:
    """Two-line summary: total h and mean h per plotted month."""
    n = int(len(values))
    if n == 0:
        return "Total: 0 h\nAvg / month: —"
    total = float(np.sum(values))
    avg = total / n
    return (
        f"Total: {_format_hours_compact(total)} h\n"
        f"Avg / month: {_format_hours_compact(avg)} h"
    )


def _panel_label(ax, letter: str) -> None:
    """Bold panel letter (A, B, …) at top-left, slightly outside the axes."""
    ax.text(
        -0.04,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
    )


def _annotate_monthly_stats(ax, values: np.ndarray) -> None:
    """Upper-left text box with total / average hours."""
    ax.text(
        0.02,
        0.97,
        _monthly_hours_stats_annotation(values),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#333333",
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.92,
        ),
        zorder=10,
    )


def _dolphin_count_for_month(year: int, month_num: int) -> int:
    """Number of dolphins present at the start of the given calendar month."""
    start = pd.Timestamp(year, month_num, 1)
    n_departed = sum(1 for _, d in DOLPHIN_DEPARTURES if d <= start)
    return DOLPHIN_INITIAL - n_departed


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load the stats CSV and return the deduplicated, valid subset."""
    df = pd.read_csv(csv_path)

    # Keep only found, non-duplicate rows with parseable metadata
    df = df[df["wav_found"] & ~df["is_duplicate"] & df["month"].notna() & df["time"].notna()].copy()

    # For multi-channel recordings keep only the lowest channel (simultaneous capture)
    df["channel_sort"] = pd.to_numeric(df["channel"], errors="coerce").fillna(0)
    df = (
        df.sort_values("channel_sort")
          .groupby("folder_name", sort=False)
          .first()
          .reset_index()
    )

    # Derived columns
    df["year"]      = df["year"].astype(int)
    df["month_num"] = df["month"].map(MONTH_NUM)
    df["hour"]      = (df["time"].astype(float) // 100).astype(int).clip(0, 23)

    if "detection_duration" not in df.columns:
        df["detection_duration"] = 0.0
    else:
        df["detection_duration"] = pd.to_numeric(df["detection_duration"], errors="coerce").fillna(0.0)

    return df


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_hourly_histogram(
    df: pd.DataFrame,
    ax,
    *,
    for_dataset_overview: bool = False,
) -> None:
    """Bar histogram of total recording hours per hour-of-day (x-axis = time)."""
    hourly = (
        df.groupby("hour")["duration_hours"]
          .sum()
          .reindex(range(24), fill_value=0)
    )

    hours = hourly.index.values
    values = hourly.values

    if for_dataset_overview:
        band_c, band_a = _SCHEDULE_BAND_COLOR_OVERVIEW, _SCHEDULE_BAND_ALPHA_OVERVIEW
    else:
        band_c, band_a = _SCHEDULE_BAND_COLOR_DEFAULT, _SCHEDULE_BAND_ALPHA_DEFAULT
    ax.axvspan(9, 16, color=band_c, alpha=band_a, zorder=0)

    ax.bar(hours, values, width=0.85, color=BAR_COLOR, edgecolor="white", linewidth=0.4, zorder=2)

    if for_dataset_overview:
        for h in _FEEDING_HOURS:
            ax.axvline(
                h,
                color=_FEEDING_VLINE_COLOR,
                linestyle="--",
                linewidth=1.05,
                alpha=0.9,
                zorder=3,
            )
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(
                    facecolor=band_c,
                    edgecolor="#3d6b35",
                    linewidth=0.6,
                    alpha=band_a,
                    label="Human\ninteractions",
                ),
                Line2D(
                    [0],
                    [0],
                    color=_FEEDING_VLINE_COLOR,
                    linestyle="--",
                    linewidth=1.2,
                    label="Feeding\ntime",
                ),
            ],
            loc="upper right",
            fontsize=9,
            framealpha=0.95,
            borderaxespad=0.0,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.5,
            borderpad=0.35,
        )

    ax.set_title("Daily recording schedule", fontsize=14, fontweight="bold", pad=12)
    xpad = 8 if for_dataset_overview else 28
    ax.set_xlabel("Hour of day (UTC)", fontsize=13, labelpad=xpad)
    ax.set_ylabel("Total recording hours", fontsize=13)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlim(-0.5, 23.5)
    ymax = float(values.max()) if len(values) else 1.0
    ax.set_ylim(0, ymax * 1.12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    if not for_dataset_overview:
        from matplotlib.transforms import blended_transform_factory as _btf

        # Reef opening hours: bracket + label ABOVE the x-axis label (axes fraction)
        _trans = _btf(ax.transData, ax.transAxes)
        _cap = 0.004
        _reef_y = -0.12
        _reef_color = "#2d5a2d"
        ax.plot([9, 16], [_reef_y, _reef_y], transform=_trans, color=_reef_color, lw=1.1, clip_on=False, zorder=5)
        for xv in (9, 16):
            ax.plot(
                [xv, xv],
                [_reef_y - _cap, _reef_y + _cap],
                transform=_trans,
                color=_reef_color,
                lw=1.1,
                clip_on=False,
                zorder=5,
            )
        ax.text(
            (9 + 16) / 2,
            _reef_y - 0.018,
            "Reef opening hours",
            transform=_trans,
            ha="center",
            va="top",
            fontsize=9,
            color=_reef_color,
            clip_on=False,
            zorder=5,
        )


def _recording_monthly(df: pd.DataFrame):
    """Chronological months with recording hours > 0."""
    counts = (
        df.groupby(["year", "month_num", "month"])["duration_hours"]
          .sum()
          .reset_index(name="hours")
          .sort_values(["year", "month_num"])
    )
    counts = counts[counts["hours"] > 0].reset_index(drop=True)
    labels = [row.month for row in counts.itertuples()]
    x      = np.arange(len(counts))
    values = counts["hours"].values
    return counts, labels, x, values


def _draw_recording_curve(ax, counts, labels, x, values) -> None:
    """Cumulative recording hours line chart with dolphin group size on a twin y-axis."""
    from matplotlib.transforms import blended_transform_factory

    cum_values = np.cumsum(values)

    # Cumulative recording hours — filled area + line
    ax.fill_between(x, cum_values, alpha=0.20, color=BAR_COLOR)
    ax.plot(x, cum_values, color=BAR_COLOR, linewidth=2.0,
            marker="o", markersize=3.5, zorder=3)

    # Dashed vertical lines bounding each year's data on both sides
    _seen_edges: set = set()
    for _yr, _grp in counts.groupby("year"):
        for _edge in (_grp.index[0] - 0.5, _grp.index[-1] + 0.5):
            if _edge not in _seen_edges:
                ax.axvline(_edge, color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=0)
                _seen_edges.add(_edge)

    ax.set_title("Recording hours", fontsize=16, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative recording hours", fontsize=13)
    ax.tick_params(axis="y")
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right", rotation_mode="anchor")

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for yr, grp in counts.groupby("year"):
        mid_x = (grp.index[0] + grp.index[-1]) / 2
        ax.text(mid_x, -0.22, str(int(yr)), transform=trans,
                ha="center", va="top", fontsize=12,
                fontweight="bold", color="#333333")

    ymax = float(cum_values.max()) if len(cum_values) else 0.0
    ax.set_ylim(0, max(ymax * 1.08, 1.0))
    step = 1000 if ymax > 3000 else 500 if ymax > 1000 else 200 if ymax > 400 else 100
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True)

    dolphin_counts = np.array(
        [_dolphin_count_for_month(int(r.year), int(r.month_num))
         for r in counts.itertuples()]
    )
    _add_dolphin_overlay(ax, counts, dolphin_counts)
    #_recording_hours_legend(ax, blue_label="Cumulative recording hours")
    _annotate_monthly_stats(ax, values)


def _add_dolphin_overlay(ax, counts: pd.DataFrame, dolphin_counts: np.ndarray) -> None:
    """Attach a dolphin-count step curve to a twin right y-axis of *ax*."""
    _alpha = 0.75
    _ann_color = (*tuple(int(DOLPHIN_COLOR.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)),
                  _alpha)

    ax2 = ax.twinx()
    ax2.step(np.arange(len(dolphin_counts)), dolphin_counts, where="pre",
             color=DOLPHIN_COLOR, linewidth=1.5, alpha=_alpha, zorder=4)
    ax2.set_ylabel("Number of dolphins", fontsize=13, color=DOLPHIN_COLOR, alpha=_alpha)
    ax2.set_ylim(0, DOLPHIN_INITIAL + 1)
    ax2.set_yticks(range(1, DOLPHIN_INITIAL + 1))
    ax2.tick_params(axis="y", colors=DOLPHIN_COLOR,
                    labelcolor=f"{DOLPHIN_COLOR}{int(_alpha * 255):02x}")
    sns.despine(ax=ax2, right=False, left=True, top=True)

    # Departure annotations — fall back to first visible drop if exact month absent
    for i, (name, dep_date) in enumerate(DOLPHIN_DEPARTURES):
        new_count = DOLPHIN_INITIAL - (i + 1)
        idx = None
        if dep_date.day == 1:
            drop_year, drop_month = dep_date.year, dep_date.month
        else:
            nxt = dep_date + pd.offsets.MonthBegin(1)
            drop_year, drop_month = nxt.year, nxt.month
        mask = (counts["year"] == drop_year) & (counts["month_num"] == drop_month)
        if mask.any():
            idx = int(counts[mask].index[0])
        else:
            for j in range(len(dolphin_counts)):
                if dolphin_counts[j] <= new_count and (j == 0 or dolphin_counts[j - 1] > new_count):
                    idx = j
                    break
        if idx is None:
            continue
        ax2.annotate(
            f"\u2212{name}",
            xy=(idx, new_count + 0.05),
            xytext=(idx + 0.5, new_count + 0.6),
            fontsize=10,
            color=_ann_color,
            ha="left",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color=_ann_color, lw=0.8),
            zorder=5,
        )


def _recording_hours_legend(ax, *, blue_label: str) -> None:
    """Legend for the blue recording series and orange dolphin-count overlay."""
    from matplotlib.lines import Line2D

    _alpha = 0.75
    h_blue = Line2D(
        [0],
        [0],
        color=BAR_COLOR,
        lw=2,
        marker="o",
        markersize=3.5,
        label=blue_label,
    )
    h_orange = Line2D(
        [0],
        [0],
        color=DOLPHIN_COLOR,
        lw=1.5,
        alpha=_alpha,
        label="Dolphin departure/death",
    )
    leg = ax.legend(
        handles=[h_blue, h_orange],
        loc="lower right",
        framealpha=0.92,
        fontsize=10,
    )
    for text in leg.get_texts():
        if text.get_text() == "Dolphin departure/death":
            text.set_color(DOLPHIN_COLOR)


def _draw_recording_curve_split(ax_top, ax_bot, counts, labels, x, values) -> None:
    """Broken-axis line chart (break at RECORDING_Y_BREAK) with dolphin overlay."""
    from matplotlib.transforms import blended_transform_factory

    Y_BREAK = RECORDING_Y_BREAK
    Y_MAX   = max(int(float(values.max()) * 1.08), Y_BREAK + 1) if len(values) else Y_BREAK + 1

    # Recording curve drawn on both panels — each clips to its own ylim
    for ax in (ax_top, ax_bot):
        ax.fill_between(x, values, alpha=0.20, color=BAR_COLOR)
        ax.plot(x, values, color=BAR_COLOR, linewidth=2.0,
                marker="o", markersize=3.5, zorder=3)

    ax_top.set_ylim(Y_BREAK, Y_MAX)
    ax_bot.set_ylim(0, Y_BREAK)

    # Break-spine decorations
    for sp in ("bottom", "top", "right"):
        ax_top.spines[sp].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    d = 0.012
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes,
                color="k", clip_on=False, linewidth=1.2)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes,
                color="k", clip_on=False, linewidth=1.2)

    # Year-boundary dashed lines on both panels
    _seen_edges: set = set()
    for _yr, _grp in counts.groupby("year"):
        for _edge in (_grp.index[0] - 0.5, _grp.index[-1] + 0.5):
            if _edge not in _seen_edges:
                for ax in (ax_top, ax_bot):
                    ax.axvline(_edge, color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=0)
                _seen_edges.add(_edge)

    ax_top.set_title("Recording hours per month", fontsize=16, fontweight="bold", pad=10)
    ax_bot.set_ylabel("Recording hours", fontsize=13)
    for ax in (ax_top, ax_bot):
        ax.tick_params(axis="y")
    ax_bot.set_xlabel("")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, fontsize=10, rotation=45, ha="right", rotation_mode="anchor")

    trans = blended_transform_factory(ax_bot.transData, ax_bot.transAxes)
    for yr, grp in counts.groupby("year"):
        mid_x = (grp.index[0] + grp.index[-1]) / 2
        ax_bot.text(mid_x, -0.22, str(int(yr)), transform=trans,
                    ha="center", va="top", fontsize=12,
                    fontweight="bold", color="#333333")

    ax_top.yaxis.set_major_locator(mticker.MultipleLocator(200))
    ax_bot.yaxis.set_major_locator(mticker.MultipleLocator(100))
    for ax in (ax_top, ax_bot):
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)

    dolphin_counts = np.array(
        [_dolphin_count_for_month(int(r.year), int(r.month_num))
         for r in counts.itertuples()]
    )
    _add_dolphin_overlay(ax_bot, counts, dolphin_counts)
    _recording_hours_legend(ax_bot, blue_label="Recording hours per month")
    _annotate_monthly_stats(ax_top, values)


def _vocalisation_monthly(df: pd.DataFrame):
    """Chronological months with vocalisation duration > 0 (hours)."""
    counts = (
        df.groupby(["year", "month_num", "month"])["detection_duration"]
          .sum()
          .reset_index(name="dur_s")
          .sort_values(["year", "month_num"])
    )
    counts = counts[counts["dur_s"] > 0].reset_index(drop=True)
    labels = [row.month for row in counts.itertuples()]
    x      = np.arange(len(counts))
    values = counts["dur_s"].values / 3600.0
    return counts, labels, x, values


def _draw_vocalisation_split(ax_top, ax_bot, counts, labels, x, values) -> None:
    """Broken y-axis at ``VOCALISATION_Y_BREAK`` h."""
    from matplotlib.transforms import blended_transform_factory

    Y_BREAK = VOCALISATION_Y_BREAK
    Y_MAX   = max(float(values.max()) * 1.08, Y_BREAK + 0.1) if len(values) else Y_BREAK + 0.1

    for ax in (ax_top, ax_bot):
        ax.bar(x, values, color=BAR_COLOR, width=0.75, edgecolor="white", linewidth=0.4)

    ax_top.set_ylim(Y_BREAK, Y_MAX)
    ax_bot.set_ylim(0, Y_BREAK)

    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    d = 0.012
    ax_top.plot((-d, +d), (-d, +d),
                transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.2)
    ax_bot.plot((-d, +d), (1 - d, 1 + d),
                transform=ax_bot.transAxes, color="k", clip_on=False, linewidth=1.2)

    year_boundaries = counts[counts["month_num"] == 1].index.tolist()
    for xb in year_boundaries[1:]:
        for ax in (ax_top, ax_bot):
            ax.axvline(xb - 0.5, color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=0)

    ax_top.set_title("Vocalisation duration per month", fontsize=16, fontweight="bold", pad=10)
    ax_bot.set_ylabel("Vocalisation duration (h)", fontsize=13)
    ax_bot.set_xlabel("")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, fontsize=10, rotation=45, ha="right", rotation_mode="anchor")

    trans = blended_transform_factory(ax_bot.transData, ax_bot.transAxes)
    for yr, grp in counts.groupby("year"):
        mid_x = (grp.index[0] + grp.index[-1]) / 2
        ax_bot.text(mid_x, -0.22, str(int(yr)), transform=trans,
                    ha="center", va="top", fontsize=12,
                    fontweight="bold", color="#333333")

    ax_top.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax_bot.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax_bot.set_xlim(-0.6, len(x) - 0.4)

    for ax in (ax_top, ax_bot):
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

    _annotate_monthly_stats(ax_top, values)


def _draw_vocalisation_flat(ax, counts, labels, x, values) -> None:
    """Single-axis vocalisation (all monthly values ≤ break threshold)."""
    from matplotlib.transforms import blended_transform_factory

    ax.bar(x, values, color=BAR_COLOR, width=0.75, edgecolor="white", linewidth=0.4)

    year_boundaries = counts[counts["month_num"] == 1].index.tolist()
    for xb in year_boundaries[1:]:
        ax.axvline(xb - 0.5, color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=0)

    ax.set_title("Vocalisation duration per month", fontsize=16, fontweight="bold", pad=12)
    ax.set_ylabel("Vocalisation duration (h)", fontsize=13)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right", rotation_mode="anchor")

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for yr, grp in counts.groupby("year"):
        mid_x = (grp.index[0] + grp.index[-1]) / 2
        ax.text(mid_x, -0.22, str(int(yr)), transform=trans,
                ha="center", va="top", fontsize=12,
                fontweight="bold", color="#333333")

    ymax = float(values.max()) if len(values) else 0.0
    ax.set_ylim(0, max(ymax * 1.12, 0.01))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    _annotate_monthly_stats(ax, values)


def _draw_vocalisation_curve(ax, counts, labels, x, values) -> None:
    """Cumulative vocalisation duration line chart (matches style of _draw_recording_curve)."""
    from matplotlib.transforms import blended_transform_factory

    cum_values = np.cumsum(values)

    ax.fill_between(x, cum_values, alpha=0.20, color=BAR_COLOR)
    ax.plot(x, cum_values, color=BAR_COLOR, linewidth=2.0,
            marker="o", markersize=3.5, zorder=3)

    _seen_edges: set = set()
    for _yr, _grp in counts.groupby("year"):
        for _edge in (_grp.index[0] - 0.5, _grp.index[-1] + 0.5):
            if _edge not in _seen_edges:
                ax.axvline(_edge, color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=0)
                _seen_edges.add(_edge)

    ax.set_title("Detected whistling hours", fontsize=16, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative detected whistling (h)", fontsize=13)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right", rotation_mode="anchor")

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for yr, grp in counts.groupby("year"):
        mid_x = (grp.index[0] + grp.index[-1]) / 2
        ax.text(mid_x, -0.22, str(int(yr)), transform=trans,
                ha="center", va="top", fontsize=12,
                fontweight="bold", color="#333333")

    ymax = float(cum_values.max()) if len(cum_values) else 0.0
    ax.set_ylim(0, max(ymax * 1.08, 0.01))
    step = 50 if ymax > 100 else 25 if ymax > 50 else 10 if ymax > 25 else 5
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    _annotate_monthly_stats(ax, values)

def _enlarge_panel_d_confusion_matrix_text(ax: plt.Axes) -> None:
    """Bump title, axis labels, ticks, and heatmap cell annotations for panel D."""
    ax.set_title(ax.get_title(), fontsize=15, fontweight="bold", pad=8)
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    for txt in ax.texts:
        fs = txt.get_fontsize()
        if fs and fs > 0:
            txt.set_fontsize(fs + 2)


def plot_combined(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    confusion_matrix_csv: Path | None = None,
) -> None:
    """2×2 combined overview (``fig_dataset_overview``)."""
    from matplotlib.gridspec import GridSpec

    rec_counts, rec_labels, rec_x, rec_values = _recording_monthly(df)
    voc_counts, voc_labels, voc_x, voc_values = _vocalisation_monthly(df)

    fig = plt.figure(figsize=(14, 9))
    outer = GridSpec(2, 1, figure=fig, hspace=0.42)
    top = outer[0].subgridspec(1, 2, width_ratios=[9.5, 5.5], wspace=0.28)
    ax_rec = fig.add_subplot(top[0, 0])
    ax_hist = fig.add_subplot(top[0, 1])
    bot = outer[1].subgridspec(1, 2, width_ratios=[9.5, 5.5], wspace=0.26)
    ax_voc = fig.add_subplot(bot[0, 0])
    ax_cm = fig.add_subplot(bot[0, 1])

    _draw_recording_curve(ax_rec, rec_counts, rec_labels, rec_x, rec_values)
    _draw_hourly_histogram(df, ax_hist, for_dataset_overview=True)
    _draw_vocalisation_curve(ax_voc, voc_counts, voc_labels, voc_x, voc_values)

    cm_path = (
        confusion_matrix_csv
        if confusion_matrix_csv is not None
        else (FIGURES_DATA_DIR / "CNN_test_confusion_matrix.csv")
    )
    loaded = load_confusion_matrix_csv(cm_path)
    if loaded is None:
        ax_cm.text(
            0.5,
            0.5,
            f"Missing or unreadable\n{cm_path.name}",
            ha="center",
            va="center",
            transform=ax_cm.transAxes,
            fontsize=14,
            color="#555555",
        )
        ax_cm.set_axis_off()
    else:
        cm, row_names, col_names = loaded
        draw_confusion_matrix_ax(ax_cm, cm, row_names, col_names, small_panel=False)
        _enlarge_panel_d_confusion_matrix_text(ax_cm)

    _panel_label(ax_rec, "A")
    _panel_label(ax_hist, "B")
    _panel_label(ax_voc, "C")
    _panel_label(ax_cm, "D")

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = output_dir / f"fig_dataset_overview.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"[figures] Saved → {out}")
    plt.close(fig)
