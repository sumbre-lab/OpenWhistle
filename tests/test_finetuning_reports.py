"""Keep published fine-tuning reports portable without changing scientific data."""

import copy
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/finetuning/src"))
from report_io import clean_report  # noqa: E402


class FinetuningReportTests(unittest.TestCase):
    def test_cleanup_preserves_effective_values_and_does_not_mutate_source(self):
        report = {
            "model_id": "aves_bio", "model_revision": None,
            "training_args": {
                "model_id": "unused model", "output_dir": "/home/researcher/run",
                "max_duration_seconds": None, "class_weight": "none", "num_bootstrap": 0,
                "eval_precision": "none", "label_smoothing": 0.0,
                "refit_train_validation": False, "gradient_accumulation_steps": 1,
            },
            "final_training": {"checkpoint": "final.pt", "class_weighting": None},
            "test": {"macro_f1": 0.0},
            "history": [{"epoch": 1, "validation": {"loss": 0.0, "optional": None}}],
        }
        original = copy.deepcopy(report)
        expected = {
            "model_id": "aves_bio",
            "training_args": {
                "eval_precision": "none", "label_smoothing": 0.0,
                "refit_train_validation": False, "gradient_accumulation_steps": 1,
            },
            "final_training": {"checkpoint": "final.pt"},
            "test": {"macro_f1": 0.0},
            "history": [{"epoch": 1, "validation": {"loss": 0.0}}],
        }
        self.assertEqual(clean_report(report), expected)
        self.assertEqual(report, original)
        self.assertEqual(clean_report(expected), expected)

    def test_active_settings_are_never_hidden_as_unused(self):
        report = {"training_args": {"max_duration_seconds": 3.0, "class_weight": "balanced"},
                  "class_weighting": {"weights": [1.0, 2.0]}}
        self.assertEqual(clean_report(report), report)

    def test_all_published_reports_have_no_null_fields_or_local_paths(self):
        reports = list((ROOT / "experiments/finetuning/results").rglob("results.json"))
        self.assertEqual(len(reports), 18)

        def check(value):
            self.assertIsNotNone(value)
            if isinstance(value, dict):
                self.assertFalse({"sha256", "model_revisions", "dataset_revisions"}.intersection(value))
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(value.startswith(("/", "~", "file:", "\\\\")), value)
                self.assertIsNone(re.match(r"^[A-Za-z]:[\\/]", value), value)

        for path in reports:
            with self.subTest(report=str(path.relative_to(ROOT))):
                report = json.loads(path.read_text())
                check(report)
                self.assertEqual(clean_report(report), report)

    def test_release_inventory_contains_only_documentation_code_and_results(self):
        root = ROOT / "experiments/finetuning"
        expected = {"README.md", "requirements.txt"}
        source = {
            "checkpoint_io.py", "finetuning_config.py", "export_finetuning_results.py",
            "finetune.py", "finetuning_models.py",
            "refit_protocol.py", "report_io.py", "run_finetuning.py",
        }
        expected.update(f"src/{name}" for name in source)
        models = ("aves_core", "biolingual", "aves_bio", "wav2vec2", "aves_scratch", "aves_continual")
        expected.add("results/results.csv")
        expected.update(f"results/{model}/{task}/seed_42/results.json"
                        for model in models for task in ("classification", "balanced", "detection"))
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
                  and not {"runs", "__pycache__"}.intersection(path.relative_to(root).parts)}
        self.assertEqual(actual, expected)

    def test_exporter_validates_all_published_results(self):
        from contextlib import redirect_stdout
        import io
        from export_finetuning_results import validate_results

        with redirect_stdout(io.StringIO()):
            validate_results(ROOT / "experiments/finetuning/results")
