#!/usr/bin/env python3
"""Build Markdown and LaTeX benchmark tables from the structured results CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hf_collection_models import HF_COLLECTION_MODELS
from pipeline_config import EMBEDDING_PIPELINE_VERSION


def parse_args():
    benchmark_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=benchmark_root / "results" / "hf_collections_benchmark.csv",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=benchmark_root / "results" / "hf_collections_benchmark.md",
    )
    parser.add_argument(
        "--latex",
        type=Path,
        default=benchmark_root / "results" / "hf_collections_benchmark.tex",
    )
    return parser.parse_args()


def formatted_score(row: dict[str, str] | None) -> str:
    if row is None:
        return "—"
    mean = 100.0 * float(row["test_mean"])
    std = 100.0 * float(row["test_std"])
    return f"{mean:.2f} ± {std:.2f}"


def main():
    args = parse_args()
    with args.results_csv.open(newline="") as result_file:
        rows = [
            row
            for row in csv.DictReader(result_file)
            if row.get("embedding_pipeline_version")
            == EMBEDDING_PIPELINE_VERSION
        ]
    by_pair = {
        (
            row["model_id"],
            row["dataset_name"],
            row.get("dataset_config", ""),
        ): row
        for row in rows
    }

    table_rows = []
    for model in HF_COLLECTION_MODELS:
        table_rows.append(
            (
                model["label"],
                formatted_score(
                    by_pair.get((model["model_id"], "classification", "balanced"))
                ),
                formatted_score(
                    by_pair.get((model["model_id"], "classification", "unbalanced"))
                ),
                formatted_score(
                    by_pair.get((model["model_id"], "classification", "all"))
                ),
                formatted_score(
                    by_pair.get((model["model_id"], "detection", "default"))
                ),
            )
        )

    markdown_lines = [
        "| Frozen backbone | Macro-F1 balanced ↑ | Macro-F1 unbalanced ↑ | Macro-F1 all ↑ | Detection mAP ↑ |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {label} | {balanced} | {unbalanced} | {all_score} | {detection} |"
            for label, balanced, unbalanced, all_score, detection in table_rows
        ],
        "",
    ]
    latex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Frozen backbone & Balanced F1 $\uparrow$ & Unbalanced F1 $\uparrow$ & All F1 $\uparrow$ & Detection mAP $\uparrow$ \\",
        r"\midrule",
        *[
            f"{label.replace('%', r'\%')} & {balanced} & {unbalanced} & "
            f"{all_score} & {detection} \\\\"
            for label, balanced, unbalanced, all_score, detection in table_rows
        ],
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(markdown_lines))
    args.latex.write_text("\n".join(latex_lines))
    print(f"Saved {args.markdown}")
    print(f"Saved {args.latex}")


if __name__ == "__main__":
    main()
