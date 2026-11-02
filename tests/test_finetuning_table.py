# ruff: noqa: E402
"""Verify one-seed fine-tuning, common evaluation and test bootstrap provenance."""
import csv
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/finetuning/src"))
import run_finetuning as table
import finetuning_config as config
from metrics import bootstrap_metric_scores, score_predictions
from hf_datasets import CLASSIFICATION_LABEL_NAMES, DETECTION_ONE_HOT_COLUMNS


class FinetuningTableTests(unittest.TestCase):
    def test_single_job_uses_the_campaign_recipe_and_rejects_invalid_task_configs(self):
        import finetune as training

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            for model in config.MODELS:
                for task in config.TASKS:
                    dataset_name, dataset_config = config.TASK_DATASETS[task]
                    command = ["finetune", "--model", model[0], "--dataset_name", dataset_name,
                               "--dataset_config", dataset_config, "--output_dir", str(output),
                               "--refit_train_validation"]
                    with patch.object(sys, "argv", command):
                        args = training.parse_args()
                    self.assertEqual(args.model_id, model[3])
                    for name, value in config.recipe(model[0], task).items():
                        self.assertEqual(getattr(args, name), value)
            for dataset_name, dataset_config in (("classification", "default"), ("detection", "all")):
                command = ["finetune", "--model", "wav2vec2", "--dataset_name", dataset_name,
                           "--dataset_config", dataset_config, "--output_dir", str(output)]
                with patch.object(sys, "argv", command), redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as stopped:
                    training.parse_args()
                self.assertEqual(stopped.exception.code, 2)
            self.assertFalse(output.exists())

    def test_models_and_evaluations_share_the_benchmark_registry(self):
        from benchmark_config import EVALUATIONS, EVALUATION_SETTINGS
        from benchmark_models import BENCHMARK_MODELS, result_model_id

        neural = [model for model in BENCHMARK_MODELS if model["key"] in config.NEURAL_KEYS]
        self.assertEqual([model[0] for model in config.MODELS], [model["key"] for model in neural])
        self.assertEqual(config.RESULT_MODEL_IDS,
                         {model["key"]: result_model_id(model) for model in neural})
        self.assertEqual(tuple(config.TASK_DATASETS.values()), EVALUATIONS)
        self.assertEqual((config.SEED, config.NUM_BOOTSTRAP),
                         (EVALUATION_SETTINGS["seed"], EVALUATION_SETTINGS["num_bootstrap"]))
        with (ROOT / "experiments/benchmark/results/results.csv").open() as handle:
            benchmark_ids = {row["model_id"] for row in csv.DictReader(handle)}
        self.assertTrue(set(config.RESULT_MODEL_IDS.values()).issubset(benchmark_ids))

    def test_launcher_rejects_scientific_overrides_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            for override in (["--seed", "43"], ["--num_bootstrap", "20"]):
                with self.subTest(override=override), \
                        patch.object(sys, "argv", ["run_finetuning", "--output_dir", str(output)] + override), \
                        redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    table.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertFalse(output.exists())

    def test_default_outputs_use_the_repository_runs_directory(self):
        self.assertEqual(config.RUNS_DIR, ROOT / "runs/finetuning")
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "finetuning"
            output = io.StringIO()
            with patch.object(table, "RUNS_DIR", run_root), \
                    patch.object(sys, "argv", ["run_finetuning", "--dry_run"]), redirect_stdout(output):
                table.main()
            self.assertIn(str(run_root / "run-"), output.getvalue())
            self.assertFalse(run_root.exists())

    def test_saved_predictions_rebuild_tables_after_checkpoint_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            for path in paths:
                path.with_name("final.pt").unlink()
            with redirect_stdout(io.StringIO()):
                table.write_results(root, 42, 30)
            self.assertTrue((root / "results.csv").is_file())
            self.assertFalse((root / "checkpoint_hashes.json").exists())

    def test_resume_keeps_completed_refits_and_archives_an_interrupted_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            base = ["table", "--output_dir", str(output)]
            with patch.object(sys, "argv", base + ["--prepare_only"]), redirect_stdout(io.StringIO()):
                table.main()
            manifest = json.loads((output / "run_config.json").read_text())
            self.assertEqual(manifest["protocol"], "supervised_end_to_end_finetuning")
            paths = self.write_fixture(output)
            interrupted = paths[0]
            report = interrupted.read_text()
            interrupted.unlink()

            def restart(command, check):
                self.assertTrue(check)
                self.assertFalse(interrupted.parent.exists())
                interrupted.parent.mkdir(parents=True)
                interrupted.write_text(report)

            with patch.object(sys, "argv", base + ["--resume"]), redirect_stdout(io.StringIO()), \
                    patch.object(table.subprocess, "run", side_effect=restart) as run, \
                    patch.object(table, "write_results"):
                table.main()
            self.assertEqual(run.call_count, 1)
            archived = list((output / "interrupted").glob("*/*/*/final.pt"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), b"fixture checkpoint")

    def test_dry_run_prepares_eighteen_jobs_with_one_seed_and_common_evaluation(self):
        import finetune as training

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            result = subprocess.run([
                sys.executable, str(ROOT / "experiments/finetuning/src/run_finetuning.py"),
                "--output_dir", str(output), "--dry_run"],
                check=True, capture_output=True, text=True)
            commands = [line for line in result.stdout.splitlines() if " -u " in line]
            self.assertEqual(len(commands), 18)
            for command, (model, task) in zip(commands, ((m, t) for m in config.MODELS for t in config.TASKS)):
                tokens = shlex.split(command)
                self.assertEqual(Path(tokens[2]).name, "finetune.py")
                with patch.object(sys, "argv", tokens[2:]):
                    args = training.parse_args()
                self.assertEqual(args.model_id, model[3])
                self.assertEqual((args.dataset_name, args.dataset_config), config.TASK_DATASETS[task])
                for name, value in config.recipe(model[0], task).items():
                    self.assertEqual(getattr(args, name), value)
                self.assertIn("--seed 42 ", command)
                self.assertIn("--eval_batch_size 1 ", command)
                self.assertIn("--eval_precision none", command)
                self.assertIn("--refit_train_validation", command)
                self.assertNotIn("--seed 43", command)
                self.assertNotIn("--class_weight balanced", command)
            balanced = [command for command in commands if "--dataset_config balanced" in command]
            self.assertEqual(len(balanced), 6)
            self.assertTrue(all("/balanced/seed_42" in command for command in balanced))
            configs = [command.split("--dataset_config ")[1].split()[0]
                       if "--dataset_config " in command else "default" for command in commands]
            self.assertEqual(configs, ["all", "balanced", "default"] * 6)
            wav2vec2_class = next(c for c in commands if "--model wav2vec2 " in c and "--dataset_config all" in c)
            self.assertEqual(config.MODEL_REVISIONS[config.MODEL_ID],
                             "b5f156048fc4da1d8d8b7345b61ebef1257b365c")
            self.assertIn("/wav2vec2/classification/seed_42", wav2vec2_class)
            self.assertIn("--batch_size 32", wav2vec2_class)
            self.assertIn("--gradient_accumulation_steps 2", wav2vec2_class)
            continual = [c for c in commands if "--feature_mode btb3_concat" in c]
            self.assertEqual(len(continual), 3)
            self.assertTrue(all("--sample_rate 16000" in c and "--batch_size 4" in c for c in continual))
            scratch = [c for c in commands if "--model aves_scratch" in c]
            self.assertEqual(len(scratch), 3)
            self.assertFalse(output.exists())

    def write_fixture(self, output):
        paths = []
        for model in config.MODELS:
            key, _, _, model_id = model
            for task in config.TASKS:
                path = output / key / task / "seed_42/results.json"
                path.parent.mkdir(parents=True)
                if task != "detection":
                    num_labels = 6 if task == "balanced" else 10
                    y_true = np.tile(np.arange(num_labels), 4)
                    y_pred = y_true.copy()
                    y_pred[:5] = (y_pred[:5] + 1) % num_labels
                else:
                    y_true = (np.arange(168).reshape(24, 7) % 4 == 0).astype(int)
                    y_pred = np.random.default_rng(7).random(y_true.shape).astype(np.float32)
                metric_task = "detection" if task == "detection" else "classification"
                score = score_predictions(metric_task, y_true, y_pred)
                record = dict(protocol="supervised_end_to_end_finetuning", model_id=model_id,
                              model_revision=config.MODEL_REVISIONS.get(model_id), seed=42,
                              dataset_config={"classification": "all", "balanced": "balanced",
                                              "detection": "default"}[task],
                              best_epoch=1, test={"map" if task == "detection" else "macro_f1": score},
                              training_args={**config.recipe(key, task), "seed": 42,
                                             "output_dir": str(path.parent),
                                             "model_id": "unused CLI default",
                                             "max_duration_seconds": None,
                                             "class_weight": "none", "num_bootstrap": 0},
                              final_training=dict(split="train+validation", checkpoint="final.pt", epochs=1,
                                                  schedule_horizon_epochs=config.recipe(key, task)["epochs"]),
                              refit_history=[dict(epoch=1, train_loss=0.1)])
                metric = "map" if task == "detection" else "macro_f1"
                record[f"validation_{metric}"] = score
                record["history"] = [dict(epoch=1, train_loss=0.1, validation={metric: score})]
                record["label_names"] = list(DETECTION_ONE_HOT_COLUMNS if task == "detection" else
                                              CLASSIFICATION_LABEL_NAMES["balanced" if task == "balanced" else "all"])
                if task != "detection":
                    cm = np.zeros((num_labels, num_labels), dtype=int)
                    np.add.at(cm, (y_true, y_pred), 1)
                    record["test_confusion_matrix"] = cm.tolist()
                    record["test"]["accuracy"] = float(cm.trace() / cm.sum())
                path.write_text(json.dumps(record))
                path.with_name("final.pt").write_bytes(b"fixture checkpoint")
                np.savez_compressed(path.with_name("test_predictions.npz"), y_true=y_true, y_pred=y_pred)
                paths.append(path)
        return paths

    def test_table_uses_full_test_and_shared_bootstrap_not_seed_variation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.write_fixture(output)
            table.write_results(output, 42, 30)
            with (output / "results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(row["num_training_seeds"] == "1" for row in rows))
            with np.load(output / "wav2vec2/classification/seed_42/test_predictions.npz") as data:
                score = score_predictions("classification", data["y_true"], data["y_pred"])
                samples = bootstrap_metric_scores("classification", data["y_true"], data["y_pred"], 30, 42)
            wav2vec2 = next(r for r in rows if r["model_id"] == config.MODEL_ID
                            and r["dataset_name"] == "classification" and r["dataset_config"] == "all")
            self.assertEqual(float(wav2vec2["test_mean"]), score)
            self.assertEqual(float(wav2vec2["test_std"]), np.std(samples, ddof=1))
            with np.load(output / "wav2vec2/classification/seed_42/bootstrap_scores.npz") as data:
                np.testing.assert_array_equal(data["scores"], samples)
            self.assertFalse((output / "finetuning.tex").exists())
            self.assertFalse((output / "finetuning_per_task.csv").exists())
            self.assertFalse((output / "table_inputs.json").exists())
            with np.load(output / "wav2vec2/balanced/seed_42/test_predictions.npz") as data:
                samples = bootstrap_metric_scores("classification", data["y_true"], data["y_pred"], 30, 42)
                balanced = next(r for r in rows if r["model_id"] == config.MODEL_ID
                                and r["dataset_config"] == "balanced")
                self.assertEqual(float(balanced["test_std"]), np.std(samples, ddof=1))
            with np.load(output / "wav2vec2/balanced/seed_42/bootstrap_scores.npz") as data:
                np.testing.assert_array_equal(data["scores"], samples)

    def test_release_export_keeps_only_csv_and_reports(self):
        from export_finetuning_results import export, validate_results

        with tempfile.TemporaryDirectory() as directory:
            source, release = Path(directory) / "run", Path(directory) / "release"
            with patch.object(sys, "argv", ["table", "--output_dir", str(source), "--prepare_only"]), \
                    redirect_stdout(io.StringIO()):
                table.main()
            manifest_path = source / "run_config.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertNotIn("model_revisions", manifest)
            self.assertNotIn("dataset_revisions", manifest)
            self.assertEqual(manifest, config.run_manifest())
            invalid = json.loads(json.dumps(manifest))
            invalid["recipes"]["wav2vec2"]["classification"]["learning_rate"] *= 2
            manifest_path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "six-model"):
                export(source, release)
            self.assertFalse(release.exists())
            manifest_path.write_text(json.dumps(manifest))
            paths = self.write_fixture(source)
            with redirect_stdout(io.StringIO()):
                table.write_results(source, 42, 1000)
                export(source, release)
            self.assertFalse(list(release.rglob("*.pt")))
            self.assertFalse(list(release.rglob("*.npz")))
            self.assertEqual({str(p.relative_to(release)) for p in release.rglob("*") if p.is_file()},
                             {"results.csv"} |
                             {str(p.relative_to(source)) for p in paths})
            for path in paths:
                relative = path.relative_to(source)
                original = json.loads(path.read_text())
                published = json.loads((release / relative).read_text())
                self.assertEqual(published["model_id"], original["model_id"])
                self.assertEqual(published["test"], original["test"])
                self.assertEqual(published["refit_history"], original["refit_history"])
                for key in ("output_dir", "model_id", "max_duration_seconds", "class_weight", "num_bootstrap"):
                    self.assertNotIn(key, published["training_args"])
                self.assertIn("output_dir", original["training_args"])
            with self.assertRaisesRegex(ValueError, "not overwritten"):
                export(source, release)
            with (release / "results.csv").open("a") as handle:
                handle.write("modified\n")
            with self.assertRaisesRegex(ValueError, "Missing or duplicate evaluations"):
                validate_results(release)

    def test_training_entrypoint_rejects_removed_weighting_flag(self):
        import finetune as training

        for model in (config.MODELS[0], next(model for model in config.MODELS if model[0] == "wav2vec2")):
            for task in ("classification", "detection"):
                command = table.make_command(Path("/tmp/unused-finetuning-test"), model, task, 42)
                error = io.StringIO()
                with patch.object(sys, "argv", command[2:] + ["--class_weight", "balanced"]), \
                        redirect_stderr(error), self.assertRaises(SystemExit) as stopped:
                    training.parse_args()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("unrecognized arguments", error.getvalue())

    def test_runner_checks_label_order_selected_epoch_and_confusion_counts(self):
        from run_finetuning import validate_report_details

        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            original = json.loads(paths[0].read_text())
            task = paths[0].parts[-3]
            validate_report_details(original, task)
            changed = json.loads(json.dumps(original))
            changed["label_names"] = list(reversed(changed["label_names"]))
            with self.assertRaisesRegex(ValueError, "label order"):
                validate_report_details(changed, task)
            changed = json.loads(json.dumps(original))
            changed["validation_macro_f1"] = 0.0
            with self.assertRaisesRegex(ValueError, "validation selection"):
                validate_report_details(changed, task)
            changed = json.loads(json.dumps(original))
            changed["test_confusion_matrix"][0][0] += 1
            with self.assertRaisesRegex(ValueError, "confusion matrix"):
                validate_report_details(changed, task)
            paths[0].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "confusion matrix"):
                table.load_result(paths[0], config.MODELS[0], task, 42)

    def test_old_model_identity_or_batched_evaluation_cannot_enter_table(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = self.write_fixture(output)
            path = next(p for p in paths if "wav2vec2" in p.parts)
            record = json.loads(path.read_text())
            record["model_id"] = "OpenWhistleNeurIPS26/OpenWhistle-Wav2Vec2.0"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Incompatible model"):
                table.write_results(output, 42, 2)
            record["model_id"] = config.MODEL_ID
            record["training_args"]["eval_batch_size"] = 8
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "evaluation settings"):
                table.write_results(output, 42, 2)
            record["training_args"]["class_weight"] = "none"
            record["training_args"]["eval_batch_size"] = 1
            record["training_args"]["refit_train_validation"] = False
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "evaluation settings"):
                table.write_results(output, 42, 2)
            record["training_args"]["refit_train_validation"] = True
            record["training_args"]["eval_batch_size"] = 1
            record["training_args"]["class_weight"] = "balanced"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "evaluation settings"):
                table.write_results(output, 42, 2)

    def test_unified_training_keeps_all_eighteen_recipes_and_exports_test_predictions(self):
        import finetune as module

        class TinyModel(torch.nn.Module):
            def __init__(self, num_labels):
                super().__init__()
                self.encoder = torch.nn.Linear(4, 4)
                self.classifier = torch.nn.Linear(4, num_labels)

            def forward(self, input_values):
                return self.classifier(self.encoder(input_values))

            def backbone_parameters(self):
                return self.encoder.parameters()

            def head_parameters(self):
                return self.classifier.parameters()

            def checkpoint_state(self):
                return self.state_dict()

            def load_checkpoint_state(self, state):
                self.load_state_dict(state)

        def collate(examples):
            return dict(input_values=torch.stack([item[0] for item in examples]),
                        labels=torch.tensor([item[2] for item in examples]))

        with tempfile.TemporaryDirectory() as directory:
            for model in config.MODELS:
                for task in config.TASKS:
                    with self.subTest(model=model[0], task=task), ExitStack() as stack:
                        is_classification = task != "detection"
                        command = table.make_command(Path(directory), model, task, 42)
                        with patch.object(sys, "argv", command[2:]):
                            args = module.parse_args()
                        args.epochs, args.patience, args.num_workers = 1, 1, 0
                        args.batch_size, args.gradient_accumulation_steps = 2, 1
                        args.mixed_precision = "none"
                        num_labels = 2 if is_classification else 7
                        examples = [(torch.arange(4).float() + i, 44100,
                                     i % 2 if is_classification else
                                     [int(j == i % 7) for j in range(7)]) for i in range(6)]
                        splits = dict.fromkeys(("train", "validation", "test"), examples)
                        labels = [str(i) for i in range(num_labels)]
                        stack.enter_context(patch.object(module, "parse_args", return_value=args))
                        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
                        loader = module.DataLoader
                        batches = []

                        def make_loader(*loader_args, **kwargs):
                            batches.append(kwargs["batch_size"])
                            return loader(*loader_args, **kwargs)

                        stack.enter_context(patch.object(module, "DataLoader", side_effect=make_loader))
                        if is_classification:
                            stack.enter_context(patch.object(module, "load_classification_splits",
                                                             return_value=(splits, SimpleNamespace(names=labels))))
                        else:
                            stack.enter_context(patch.object(module, "load_detection_splits", return_value=(splits, labels)))
                        stack.enter_context(patch.object(module, "AudioDataset", side_effect=lambda x, _task: x))
                        factory = stack.enter_context(patch.object(
                            module, "make_model_and_collator",
                            side_effect=lambda *_args: (TinyModel(num_labels), collate, 44100)))
                        refit_factory = stack.enter_context(patch.object(
                            module, "make_refit_model", side_effect=lambda *_args: TinyModel(num_labels)))
                        evaluation = stack.enter_context(patch.object(module, "evaluate", wraps=module.evaluate))
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            module.main()
                        self.assertEqual(batches, [2, 1, 1])
                        self.assertEqual([call.args[-1] for call in evaluation.call_args_list], ["none", "none"])
                        result = json.loads((args.output_dir / "results.json").read_text())
                        self.assertEqual(factory.call_count, 1)
                        self.assertEqual(refit_factory.call_count, int(args.refit_train_validation))
                        self.assertEqual(result["final_training"]["split"],
                                         "train+validation" if args.refit_train_validation else "train")
                        self.assertEqual(len(result["refit_history"]), int(args.refit_train_validation))
                        if args.refit_train_validation:
                            self.assertTrue((args.output_dir / "final.pt").is_file())
                        self.assertEqual(result["training_args"]["batch_size"], 2)
                        self.assertNotIn("output_dir", result["training_args"])
                        self.assertNotIn("model_id", result["training_args"])
                        self.assertNotIn("predictions", result["test"])
                        with np.load(args.output_dir / "test_predictions.npz") as data:
                            score = score_predictions("classification" if is_classification else "detection",
                                                      data["y_true"], data["y_pred"])
                        metric = "macro_f1" if is_classification else "map"
                        self.assertAlmostEqual(score, result["test"][metric])


if __name__ == "__main__":
    unittest.main()
