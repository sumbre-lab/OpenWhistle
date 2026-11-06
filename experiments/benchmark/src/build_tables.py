"""Validate benchmark scores and generate Markdown, LaTeX and optional PDF tables.

Published results use the experiment definitions and aggregate split counts in
benchmark_config; local evaluations use their saved run settings and labels.
"""

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from benchmark_config import (
    CACHE_FINGERPRINT,
    CLASSIFICATION_CONFIGS,
    EVALUATIONS,
    read_run_config,
)
from benchmark_models import BENCHMARK_MODELS, result_model_id


def label_permutation_macro_f1(labels, num_classes):
    """Exact expected macro-F1 of a random permutation of the test labels.

    Permutation preserves every class count n. E[TP] = n*n/N, hence
    E[F1] = E[TP]/n = n/N. Averaging over K present classes gives 1/K.
    """
    counts = np.bincount(labels, minlength=num_classes)
    return label_permutation_macro_f1_from_counts(counts, num_classes)


def label_permutation_macro_f1_from_counts(counts, num_classes):
    counts = np.asarray(counts)
    if (
        num_classes < 1
        or counts.ndim != 1
        or len(counts) != num_classes
        or not np.isfinite(counts).all()
        or np.any(counts <= 0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError(
            "Label-permutation chance requires every declared class in test."
        )
    return 1.0 / num_classes


def random_ranking_map(labels):
    """Exact expected AP of a uniform random ranking, averaged across labels."""
    return random_ranking_map_from_counts(len(labels), labels.sum(axis=0))


def random_ranking_map_from_counts(total, counts):
    counts = np.asarray(counts)
    if (
        total < 1
        or total != int(total)
        or counts.ndim != 1
        or not len(counts)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts > total)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError("Invalid detection split counts.")
    if total == 1:
        return float(np.mean(counts > 0))
    harmonic = np.sum(1 / np.arange(1, total + 1))
    prevalence = counts / total
    expectation = prevalence + (1 - prevalence) * (harmonic - 1) / (total - 1)
    return float(np.where(counts > 0, expectation, 0).mean())


def split_statistics(run_dir, manifest):
    """Use aggregate counts for summary releases, or labels from a local run."""
    if "split_statistics" in manifest:
        return manifest["split_statistics"]
    statistics = {"classification": {}, "detection": {}}
    for config in CLASSIFICATION_CONFIGS:
        train = cached_labels(run_dir, manifest, "classification", config, "train")
        test = cached_labels(run_dir, manifest, "classification", config, "test")
        classes = int(train.max()) + 1
        statistics["classification"][config] = {
            "train_class_counts": np.bincount(train, minlength=classes).tolist(),
            "test_class_counts": np.bincount(test, minlength=classes).tolist(),
        }
    labels = cached_labels(run_dir, manifest, "detection", "default", "test")
    statistics["detection"]["default"] = {
        "test_examples": len(labels),
        "positives_per_label": labels.sum(axis=0).tolist(),
    }
    return statistics


def cached_labels(run_dir, manifest, task, config, split):
    exported = run_dir / "labels" / task / config / f"{split}.npz"
    if exported.is_file():
        with np.load(exported) as arrays:
            return arrays["y"]
    model_key = f"spectral_features__strict_batch{manifest['batch_size']}_fp32"
    cache_dir = Path(manifest.get("embedding_cache_dir", run_dir / "embedding_cache"))
    path = (
        cache_dir
        / manifest.get("cache_fingerprint", CACHE_FINGERPRINT)
        / model_key
        / task
        / config
        / f"{split}.npz"
    )
    with np.load(path) as arrays:
        return arrays["y"]


def score(row, latex=False, bold=False):
    mean, std = 100 * float(row["test_mean"]), 100 * float(row["test_std"])
    if latex:
        text = f"{mean:.2f}\\pm{std:.2f}"
        return "$" + (r"\mathbf{" + text + "}" if bold else text) + "$"
    text = f"{mean:.2f} ± {std:.2f}"
    return f"**{text}**" if bold else text


def build(run_dir, output_dir=None):
    manifest = read_run_config(run_dir)
    models = manifest.get("models", BENCHMARK_MODELS)
    if [model["key"] for model in models] != [
        model["key"] for model in BENCHMARK_MODELS
    ]:
        raise ValueError(
            "Run configuration must contain the nine benchmark representations in order."
        )
    num_seeds = int(manifest.get("num_seeds", 1))
    with (run_dir / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {
        (row["model_id"], row["dataset_name"], row["dataset_config"]): row
        for row in rows
    }
    expected = {
        (result_model_id(model), task, config)
        for model in models
        for task, config in EVALUATIONS
    }
    if len(by_key) != len(rows):
        raise ValueError("Duplicate model/task/config rows in results.csv.")
    if set(by_key) - expected:
        raise ValueError("Unexpected model/task/config rows in results.csv.")
    for model in models:
        for task, config in EVALUATIONS:
            key = (result_model_id(model), task, config)
            if key not in by_key:
                raise ValueError(f"Incomplete rerun: missing {key}")
            row = by_key[key]
            if row.get("normalize_data") != "True":
                raise ValueError(f"Benchmark tables require normalize_data=True: {key}")
            neural = model["backend"] not in {
                "spectral_features",
                "mfcc",
                "spectrogram",
            }
            if (
                row.get("metric") != ("Macro-F1" if task == "classification" else "mAP")
                or row.get("embedding_mode") != ("batched" if neural else "individual")
                or row.get("strict_embedding_batches") != "True"
                or row.get("mixed_precision") != "False"
                or int(row["embedding_batch_size"]) != manifest["batch_size"]
                or int(row["num_bootstrap"]) != manifest["num_bootstrap"]
                or (
                    "seed" in manifest
                    and int(row.get("seed") or -1) != manifest["seed"]
                )
                or int(row["num_seeds"]) != num_seeds
            ):
                raise ValueError(
                    f"Incompatible extraction or uncertainty settings: {key}"
                )
            if (
                not np.isfinite([float(row["test_mean"]), float(row["test_std"])]).all()
                or not 0 <= float(row["test_mean"]) <= 1
                or float(row["test_std"]) < 0
            ):
                raise ValueError(f"Invalid test score or uncertainty: {key}")
    statistics = split_statistics(run_dir, manifest)
    detection = statistics["detection"]["default"]
    detection_chance = random_ranking_map_from_counts(
        detection["test_examples"], detection["positives_per_label"]
    )
    classification_chances = {}
    overview = [
        "| Method | Pretraining | Macro-F1 unbalanced | Macro-F1 balanced | Detection mAP |",
        "|---|---|---:|---:|---:|",
    ]
    for model in models:
        values = [
            score(by_key[(result_model_id(model), "classification", config)])
            for config in CLASSIFICATION_CONFIGS
        ]
        values.append(score(by_key[(result_model_id(model), "detection", "default")]))
        overview.append(
            f"| {model['label']} | {model['pretraining']} | "
            + " | ".join(values)
            + " |"
        )
    tables = output_dir or run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for config in CLASSIFICATION_CONFIGS:
        counts = statistics["classification"][config]
        num_classes = len(counts["train_class_counts"])
        classification_chance = label_permutation_macro_f1_from_counts(
            counts["test_class_counts"], num_classes
        )
        classification_chances[config] = classification_chance
        selected = [
            (
                model,
                by_key[(result_model_id(model), "classification", config)],
                by_key[(result_model_id(model), "detection", "default")],
            )
            for model in models
        ]
        best_f1 = max(float(row["test_mean"]) for _, row, _ in selected)
        best_map = max(float(row["test_mean"]) for _, _, row in selected)
        markdown = [
            f"Classification configuration: `{config}`. Scores in percent.",
            "",
            "| Method | Pretraining | Classification Macro-F1 | Detection mAP |",
            "|---|---|---:|---:|",
            f"| Chance level | Label permutation / random ranking | {100 * classification_chance:.2f} | {100 * detection_chance:.2f} |",
        ]
        latex = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \resizebox{\linewidth}{!}{",
            r"  \begin{tabular}{l l cc}",
            r"    \toprule",
            r"    Method & Pretraining & Classification (macro-F1) & Detection (mAP) \\",
            r"    \midrule",
            f"    Chance level & -- & {100 * classification_chance:.2f} & {100 * detection_chance:.2f} \\\\",
        ]
        previous_group = None
        for model, classification, detection in selected:
            if model["group"] != previous_group:
                latex.extend(
                    [
                        r"    \midrule",
                        r"    \multicolumn{4}{l}{\textit{" + model["group"] + r"}} \\",
                    ]
                )
                previous_group = model["group"]
            best_classification = float(classification["test_mean"]) == best_f1
            best_detection = float(detection["test_mean"]) == best_map
            markdown.append(
                f"| {model['label']} | {model['pretraining']} | {score(classification, bold=best_classification)} | {score(detection, bold=best_detection)} |"
            )
            label = model.get("latex_label", model["label"])
            pretraining = model["pretraining"].replace(
                " checkpoint to ", r" checkpoint $\rightarrow$ "
            )
            latex.append(
                f"    {label} & {pretraining} & {score(classification, latex=True, bold=best_classification)} & {score(detection, latex=True, bold=best_detection)} \\\\"
            )
        configuration = "macro-F1" if config == "unbalanced" else f"{config}, macro-F1"
        description = (
            f"Performance on whistle-type classification ({configuration}) and whistle-type detection (mAP). "
            "Models are grouped by representation and pretraining strategy. All use linear probing with "
            "a logistic regression classifier on frozen representations. Scores are reported as percentages. "
            "Each entry gives the full-test metric $\\pm$ the standard deviation over "
            f"{num_seeds * manifest['num_bootstrap']:,} bootstrap resamples of the test set. "
        )
        markdown.extend(
            [
                "",
                description.rstrip(),
                "Continual AVES uses three 16 kHz band views concatenated into one embedding; scratch AVES uses native 44.1 kHz audio.",
                "",
            ]
        )
        latex.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"  }",
                "",
                r"  \caption{"
                + description
                + r"$^{\dagger}$Continual pretraining at 16\,kHz with multi-band encoding; "
                r"$^{\ddagger}$trained natively at 44.1\,kHz (Sec.~\ref{app:pretraining}).}",
                r"  \label{tab:results"
                + ("" if config == "unbalanced" else "_" + config)
                + "}",
                r"\end{table}",
                "",
            ]
        )
        (tables / f"benchmark_{config}.md").write_text("\n".join(markdown))
        (tables / f"benchmark_{config}.tex").write_text("\n".join(latex))
    overview.insert(
        2,
        "| Chance level | Label permutation / random ranking | "
        + " | ".join(
            f"{100 * classification_chances[config]:.2f}"
            for config in CLASSIFICATION_CONFIGS
        )
        + f" | {100 * detection_chance:.2f} |",
    )
    (tables / "benchmark_overview.md").write_text("\n".join(overview) + "\n")
    print(f"Saved tables to {tables}")
    return tables


def compile_pdfs(tables, tectonic=None):
    """Compile standalone PDFs and keep only the LaTeX snippets and PDFs."""
    if tectonic is None and not shutil.which("pdflatex"):
        raise RuntimeError(
            "PDF generation requires pdflatex or --tectonic_bin /path/to/tectonic."
        )
    for config in CLASSIFICATION_CONFIGS:
        name = f"benchmark_{config}"
        snippet = (tables / f"{name}.tex").read_text()
        # The appendix reference is resolved only in the full paper.
        snippet = snippet.replace(r" (Sec.~\ref{app:pretraining})", "")
        source = "\n".join(
            [
                r"\documentclass[11pt,a4paper,landscape]{article}",
                r"\usepackage[margin=18mm]{geometry}",
                r"\usepackage{booktabs,graphicx,caption}",
                r"\captionsetup{font=small}",
                r"\pagestyle{empty}",
                r"\begin{document}",
                snippet,
                r"\end{document}",
            ]
        )
        with tempfile.TemporaryDirectory(prefix="openwhistle-table-") as directory:
            temporary = Path(directory)
            (temporary / f"{name}.tex").write_text(source)
            command = (
                [str(tectonic), f"{name}.tex"]
                if tectonic
                else [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"{name}.tex",
                ]
            )
            result = subprocess.run(
                command, cwd=temporary, capture_output=True, text=True
            )
            if result.returncode:
                raise RuntimeError(
                    f"Failed to compile {name}:\n{result.stdout}\n{result.stderr}"
                )
            shutil.copy2(temporary / f"{name}.pdf", tables / f"{name}.pdf")
        print(f"Saved PDF: {tables / (name + '.pdf')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Directory containing results.csv: published results or a local run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Write tables here; defaults to run_dir/tables.",
    )
    parser.add_argument(
        "--pdf", action="store_true", help="Also compile standalone PDFs."
    )
    parser.add_argument(
        "--tectonic_bin",
        type=Path,
        default=None,
        help="Use this Tectonic executable instead of pdflatex.",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve() if args.output_dir else None
    tables = build(args.run_dir.expanduser().resolve(), output_dir=output)
    if args.pdf:
        compile_pdfs(
            tables,
            args.tectonic_bin.expanduser().resolve() if args.tectonic_bin else None,
        )


if __name__ == "__main__":
    main()
