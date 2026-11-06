# ruff: noqa: E402  # Benchmark scripts are imported after adding their source directory.
"""Verify the benchmark matrix, batching and chance-level definitions."""

import csv
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

SRC = Path(__file__).resolve().parents[1] / "experiments/benchmark/src"
sys.path.insert(0, str(SRC))
import benchmark_config
import run_benchmark as launcher
import train_lr_splits
from benchmark_config import CACHE_FINGERPRINT
from build_tables import (
    build,
    label_permutation_macro_f1,
    random_ranking_map,
    split_statistics,
)
from benchmark_models import BENCHMARK_MODELS, result_model_id
from models import Aves, BioLingual
from run_benchmark import make_command
from train_lr_splits import RESULT_FIELDS, embed_split


class BenchmarkProtocolTests(unittest.TestCase):
    def test_publication_tree_contains_only_experiment_sources_and_results(self):
        benchmark = SRC.parent
        self.assertEqual(
            {path.name for path in benchmark.iterdir()},
            {
                "README.md",
                "requirements.txt",
                "fig_tasks.png",
                "src",
                "results",
            },
        )
        for path in benchmark.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            self.assertNotIn(path.suffix, {".npz", ".log", ".safetensors", ".pt"})

    def test_dependency_change_invalidates_cached_embeddings_automatically(self):
        original = benchmark_config.embedding_cache_fingerprint()
        installed_version = benchmark_config.importlib.metadata.version

        def changed_version(name):
            return "different-build" if name == "numpy" else installed_version(name)

        with patch.object(
            benchmark_config.importlib.metadata, "version", side_effect=changed_version
        ):
            self.assertNotEqual(
                benchmark_config.embedding_cache_fingerprint(), original
            )

    def test_single_model_defaults_match_the_published_protocol(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_lr_splits.py",
                "--model",
                "mfcc",
                "--dataset_name",
                "classification",
            ],
        ):
            args = train_lr_splits.get_args()
        self.assertEqual(args.num_seeds, 1)
        self.assertEqual(args.num_bootstrap, 1000)
        self.assertEqual(args.embedding_batch_size, 1)
        self.assertEqual(args.inverse_regs, [0.1, 1, 10])
        self.assertTrue(args.strict_embedding_batches)
        self.assertFalse(args.mixed_precision)

    def test_published_artifacts_form_a_complete_reproducible_release(self):
        release = SRC.parent / "results"
        with (release / "results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = {
            (result_model_id(model), task, config)
            for model in BENCHMARK_MODELS
            for task, config in (
                ("classification", "unbalanced"),
                ("classification", "balanced"),
                ("detection", "default"),
            )
        }
        self.assertEqual(len(rows), 27)
        self.assertEqual(
            {
                (row["model_id"], row["dataset_name"], row["dataset_config"])
                for row in rows
            },
            expected,
        )
        expected_files = {"results.csv", "tables/benchmark_overview.md"}
        expected_files.update(
            f"tables/benchmark_{config}.{extension}"
            for config in ("unbalanced", "balanced")
            for extension in ("md", "tex", "pdf")
        )
        self.assertEqual(
            {
                p.relative_to(release).as_posix()
                for p in release.rglob("*")
                if p.is_file()
            },
            expected_files,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build(release, output_dir=output)
            for path in output.iterdir():
                self.assertEqual(
                    path.read_bytes(), (release / "tables" / path.name).read_bytes()
                )

    def test_tables_reject_disabled_or_missing_normalization_before_writing(self):
        release = SRC.parent / "results"
        with (release / "results.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            original = list(reader)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = benchmark_config.read_run_config(release)
            (root / "run_config.json").write_text(json.dumps(manifest))
            for index in (0, 2):  # Classification and detection.
                for value in ("False", "", None):
                    with self.subTest(
                        task=original[index]["dataset_name"], value=value
                    ):
                        rows = [dict(row) for row in original]
                        columns = fields
                        if value is None:
                            columns = [
                                name for name in fields if name != "normalize_data"
                            ]
                            for row in rows:
                                row.pop("normalize_data")
                        else:
                            rows[index]["normalize_data"] = value
                        with (root / "results.csv").open("w", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=columns)
                            writer.writeheader()
                            writer.writerows(rows)
                        with self.assertRaisesRegex(ValueError, "normalize_data=True"):
                            build(root)
                        self.assertFalse((root / "tables").exists())

    def test_resume_rejects_changed_environment_without_running_or_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            argv = ["run_benchmark.py", "--output_dir", str(root)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    launcher, "environment_record", return_value={"python": "original"}
                ),
                patch.object(launcher.subprocess, "run") as run,
            ):
                launcher.main()
                self.assertEqual(run.call_count, 28)
            original = (root / "environment.json").read_bytes()
            with (
                patch.object(sys, "argv", argv + ["--resume"]),
                patch.object(
                    launcher, "environment_record", return_value={"python": "changed"}
                ),
                patch.object(launcher.subprocess, "run") as run,
            ):
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
            self.assertEqual((root / "environment.json").read_bytes(), original)

    def test_complete_continual_cache_runs_without_model_or_dataset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache_key = "dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz__btb3_concat__strict_batch64_fp32"
            paths = []
            for split in ("train", "validation", "test"):
                path = train_lr_splits.embedding_cache_path(
                    cache, cache_key, "classification", "unbalanced", split
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                labels = np.array([0, 1] * 10)
                np.savez(path, x=np.column_stack((labels, 1 - labels)), y=labels)
                paths.append(path)
            argv = [
                "train_lr_splits.py",
                "--model",
                "hf",
                "--hf_model_id",
                "dolphinteam/AVES-bio-OpenWhistle-100pct-44kHz",
                "--hf_sample_rate",
                "16000",
                "--hf_feature_mode",
                "btb3_concat",
                "--dataset_name",
                "classification",
                "--dataset_config",
                "unbalanced",
                "--strict_embedding_batches",
                "--embedding_batch_size",
                "64",
                "--embedding_cache_dir",
                str(cache),
                "--results_csv",
                str(root / "results.csv"),
                "--num_seeds",
                "1",
                "--num_bootstrap",
                "20",
                "--predictions_dir",
                str(root / "predictions"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    train_lr_splits,
                    "HuggingFaceAudioBackbone",
                    side_effect=AssertionError("encoder must not load"),
                ) as encoder,
                patch.object(
                    train_lr_splits,
                    "load_classification_splits",
                    side_effect=AssertionError("dataset must not load"),
                ) as dataset,
                patch(
                    "transformers.AutoFeatureExtractor.from_pretrained",
                    side_effect=AssertionError(
                        "explicit sample rate needs no preprocessor"
                    ),
                ),
            ):
                train_lr_splits.main()
                encoder.assert_not_called()
                dataset.assert_not_called()
                with (root / "results.csv").open() as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(float(row["test_mean"]), 1.0)
                self.assertEqual(row["target_sample_rate"], "16000")
                self.assertEqual(row["feature_mode"], "btb3_concat")
                self.assertEqual(row["mixed_precision"], "False")
                self.assertEqual(row["seed"], "42")
                with np.load(
                    root
                    / "predictions/dolphinteam__AVES-bio-OpenWhistle-100pct-44kHz/classification/unbalanced.npz"
                ) as data:
                    np.testing.assert_array_equal(data["y"], labels)
                    np.testing.assert_array_equal(data["predictions"], labels)
                for split in ("train", "validation", "test"):
                    self.assertTrue(
                        (root / "labels/classification/unbalanced" / f"{split}.npz").is_file()
                    )
                # A partial cache must still use the actual model to extract
                # the missing split, rather than silently omitting examples.
                paths[-1].unlink()
                with self.assertRaisesRegex(AssertionError, "encoder must not load"):
                    train_lr_splits.main()

    def test_launcher_has_27_jobs_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "run_benchmark.py"),
                    "--output_dir",
                    directory,
                    "--dry_run",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            commands = result.stdout.splitlines()
            self.assertEqual(len(commands), 28)
            self.assertEqual(len(list(Path(directory).iterdir())), 0)
            for command in commands[:-1]:
                self.assertIn("--embedding_batch_size 1", command)
                self.assertIn("--batch_native_models", command)
                self.assertIn("--strict_embedding_batches", command)
                self.assertIn("--num_seeds 1 --num_bootstrap 1000", command)
                self.assertIn("--no-mixed_precision", command)
                self.assertNotIn("--model dolph2vec", command)
                self.assertNotIn("--hf_local_dir", command)
                self.assertNotIn("--hf_local_files_only", command)
            self.assertEqual(sum("--dataset_config unbalanced" in c for c in commands), 9)
            self.assertEqual(sum("--dataset_config balanced" in c for c in commands), 9)
            self.assertEqual(sum("--dataset_name detection" in c for c in commands), 9)

    def test_output_and_cache_overrides_preserve_the_published_protocol(self):
        args = SimpleNamespace(
            output_dir=Path("/tmp/first-run"),
            embedding_cache_dir=Path("/tmp/first-cache"),
        )
        for model in BENCHMARK_MODELS:
            with self.subTest(model=model["key"]):
                first = make_command(args, model, "classification", "unbalanced")
                args.output_dir = Path("/tmp/second-run")
                args.embedding_cache_dir = Path("/tmp/second-cache")
                second = make_command(args, model, "classification", "unbalanced")
                args.output_dir = Path("/tmp/first-run")
                args.embedding_cache_dir = Path("/tmp/first-cache")
                expected = [
                    part.replace("/tmp/first-run", "/tmp/second-run").replace(
                        "/tmp/first-cache", "/tmp/second-cache"
                    )
                    for part in first
                ]
                self.assertEqual(second, expected)

    def test_native_aves_forwards_all_64_clips_once_and_excludes_padded_frames(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.zeros(1))
                self.calls = 0

            def extract_features(self, inputs, lengths):
                self.calls += 1
                self.batch_size = len(inputs)
                return [inputs.unsqueeze(-1)], lengths

        model = Aves.__new__(Aves)
        encoder = Encoder()
        model.feature_extractor = SimpleNamespace(model=encoder)
        model.feature_mode = "standard"
        model.batch_native_models = True
        model.sample_rate = 44100
        audio = [
            {"array": np.full(16 + i, i + 1, dtype=np.float32), "sampling_rate": 44100}
            for i in range(64)
        ]
        x = model.embed_batch(audio)
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(encoder.batch_size, 64)
        np.testing.assert_array_equal(x[:, 0].numpy(), np.arange(1, 65))

    def test_biolingual_forwards_64_clips_in_one_model_call(self):
        model = BioLingual.__new__(BioLingual)
        model.batch_native_models = True
        model.device = "cpu"
        extractor = Mock(return_value={"input_features": torch.ones(64, 3)})
        model.processor = SimpleNamespace(feature_extractor=extractor)
        forward = Mock(return_value=torch.ones(64, 8))
        model.model = SimpleNamespace(get_audio_features=forward)
        audio = [{"array": np.ones(16, dtype=np.float32), "sampling_rate": 48000}] * 64
        self.assertEqual(tuple(model.embed_batch(audio).shape), (64, 8))
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(len(extractor.call_args.args[0]), 64)

    def test_strict_batch_failure_does_not_change_batch_composition(self):
        forward = Mock(side_effect=RuntimeError("out of memory"))
        model = SimpleNamespace(embed_batch=forward)
        rows = [{"audio": {}, "label": 0}] * 64
        with self.assertRaisesRegex(RuntimeError, "strict batching refuses"):
            embed_split(rows, "train", "classification", model, 64, strict_batches=True)
        self.assertEqual(forward.call_count, 1)

    def test_label_permutation_chance_matches_enumerated_predictions(self):
        labels = np.array([0, 0, 1])
        brute_force = np.mean(
            [
                f1_score(
                    labels, prediction, labels=[0, 1], average="macro", zero_division=0
                )
                for prediction in itertools.permutations(labels)
            ]
        )
        self.assertAlmostEqual(label_permutation_macro_f1(labels, 2), brute_force)
        self.assertAlmostEqual(brute_force, 0.5)
        with self.assertRaisesRegex(ValueError, "every declared class"):
            label_permutation_macro_f1(np.array([0, 0]), 2)

    def test_exact_random_ranking_ap_matches_enumerated_rankings(self):
        labels = np.array([0, 0, 1, 1])
        brute_force = np.mean(
            [
                average_precision_score(labels, scores)
                for scores in itertools.permutations(range(4))
            ]
        )
        self.assertAlmostEqual(random_ranking_map(labels[:, None]), brute_force)

    def test_tables_require_matching_protocol_and_include_all_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(
                cache_fingerprint=CACHE_FINGERPRINT,
                batch_size=64,
                num_bootstrap=1000,
                num_seeds=10,
            )
            (root / "run_config.json").write_text(json.dumps(manifest))
            rows = []
            for index, model in enumerate(BENCHMARK_MODELS):
                for task, config in [
                    ("classification", c) for c in ("unbalanced", "balanced")
                ] + [("detection", "default")]:
                    rows.append(
                        dict(
                            model_id=result_model_id(model),
                            dataset_name=task,
                            dataset_config=config,
                            metric="Macro-F1" if task == "classification" else "mAP",
                            embedding_mode="individual" if index < 3 else "batched",
                            strict_embedding_batches=True,
                            embedding_batch_size=64,
                            mixed_precision=False,
                            normalize_data=True,
                            num_seeds=10,
                            num_bootstrap=1000,
                            test_mean=0.501234 + index / 100,
                            test_std=0.02,
                        )
                    )

            def write_rows():
                with (root / "results.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)

            write_rows()
            for task, config in [("classification", c) for c in ("unbalanced", "balanced")] + [
                ("detection", "default")
            ]:
                path = (
                    root
                    / "embedding_cache"
                    / CACHE_FINGERPRINT
                    / "spectral_features__strict_batch64_fp32"
                    / task
                    / config
                )
                path.mkdir(parents=True)
                for split in ("train", "test"):
                    np.savez(
                        path / f"{split}.npz",
                        y=np.array([0, 0, 1])
                        if task == "classification"
                        else np.array([[0, 1], [0, 0], [1, 0]]),
                    )
            build(root)
            latex = (root / "tables/benchmark_unbalanced.tex").read_text()
            self.assertIn(r"\label{tab:results}", latex)
            self.assertIn(r"\mathbf{58.12\pm2.00}", latex)
            self.assertIn("10,000 bootstrap resamples", latex)
            self.assertIn("Scores are reported as percentages", latex)
            self.assertNotIn("RNG seed", latex)
            statistics = split_statistics(root, manifest)
            self.assertEqual(
                statistics["classification"]["unbalanced"]["test_class_counts"], [2, 1]
            )
            self.assertEqual(statistics["detection"]["default"]["test_examples"], 3)
            self.assertIn("50.00", latex)
            self.assertEqual(len(list((root / "tables").glob("*.tex"))), 2)
            rows.append(dict(rows[-1]))
            write_rows()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build(root)
            rows.pop()
            rows[-1]["num_seeds"] = 1
            write_rows()
            with self.assertRaisesRegex(ValueError, "Incompatible extraction"):
                build(root)
            rows.append(dict(rows[-1]))
            write_rows()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build(root)
            rows.pop()
            rows[-1]["num_seeds"] = 10
            rows[-1]["embedding_mode"] = "individual"
            write_rows()
            with self.assertRaisesRegex(ValueError, "Incompatible extraction"):
                build(root)
            rows[-1]["embedding_mode"] = "batched"
            manifest["batch_size"] = 1
            (root / "run_config.json").write_text(json.dumps(manifest))
            write_rows()
            with self.assertRaisesRegex(ValueError, "Incompatible extraction"):
                build(root)
            for row in rows:
                row["embedding_batch_size"] = 1
            write_rows()
            cache = root / "embedding_cache" / CACHE_FINGERPRINT
            (cache / "spectral_features__strict_batch64_fp32").rename(
                cache / "spectral_features__strict_batch1_fp32"
            )
            build(root)
            regenerated_latex = (root / "tables/benchmark_unbalanced.tex").read_text()
            self.assertEqual(regenerated_latex, latex)

            # A replacement checkpoint must be selected from this run's manifest,
            # with chance-level labels resolved through the shared cache.
            manifest["models"] = [dict(model) for model in BENCHMARK_MODELS]
            replacement = "test/alternative-wav2vec2-checkpoint"
            model_index = next(
                i
                for i, model in enumerate(BENCHMARK_MODELS)
                if model["key"] == "wav2vec2"
            )
            manifest["models"][model_index]["model_id"] = replacement
            for row in rows[3 * model_index : 3 * model_index + 3]:
                row["model_id"] = replacement
                row["test_mean"] = 0.91
            shared_cache = root / "shared_cache"
            (root / "embedding_cache").rename(shared_cache)
            manifest["embedding_cache_dir"] = str(shared_cache)
            (root / "run_config.json").write_text(json.dumps(manifest))
            write_rows()
            build(root)
            self.assertIn(
                r"\mathbf{91.00\pm2.00}",
                (root / "tables/benchmark_unbalanced.tex").read_text(),
            )
            rows[3 * model_index]["model_id"] = BENCHMARK_MODELS[model_index][
                "model_id"
            ]
            write_rows()
            with self.assertRaisesRegex(ValueError, "Unexpected model/task/config"):
                build(root)


if __name__ == "__main__":
    unittest.main()
