import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "experiments/benchmark/src"
sys.path.insert(0, str(SRC))
from hf_collection_models import AVES_SCRATCH_MODELS, REBUTTAL_PRETRAINING_MODELS


class PretrainingBackboneTests(unittest.TestCase):
    def test_native_export_uses_versioned_files_without_modifying_download(self):
        models = importlib.import_module("models")
        spec = AVES_SCRATCH_MODELS[1]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "original.json"
            original = '{"sample_rate":44100,"encoder_num_layers":12}'
            config.write_text(original)
            weights = Path(directory) / "weights.pt"
            weights.write_bytes(b"test stub")

            def download(**kwargs):
                self.assertEqual(kwargs["repo_id"], spec["model_id"])
                self.assertEqual(kwargs["revision"], spec["revision"])
                return str(config if kwargs["filename"] == spec["config_filename"] else weights)

            with patch("huggingface_hub.hf_hub_download", side_effect=download) as fetch, \
                    patch.object(models.Aves, "__init__", return_value=None) as initialize:
                backbone = models.HuggingFaceAvesBackbone(
                    spec["model_id"], hf_sample_rate=44100, sample_rate=44100,
                    aves_model_path="", aves_config_path="")
                self.assertEqual(fetch.call_count, 2)
                arguments = initialize.call_args.kwargs
                self.assertEqual(arguments["aves_model_path"], str(weights))
                self.assertEqual(arguments["sample_rate"], 44100)
                self.assertEqual(json.loads(Path(arguments["aves_config_path"]).read_text()),
                                 {"encoder_num_layers": 12})
                self.assertEqual(config.read_text(), original)
                backbone._config_directory.cleanup()

    def test_rebuttal_runner_uses_native_aves_and_correct_full_corpus_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, str(SRC / "run_hf_collections.py"), "--collection", "rebuttal_pretraining",
                "--classification_configs", "all", "--datasets", "classification", "--dry_run",
                "--results_csv", str(Path(directory) / "new.csv")],
                check=True, capture_output=True, text=True)
        commands = [line for line in result.stdout.splitlines() if "--model " in line]
        self.assertEqual(len(commands), 5)
        self.assertIn("--model aves_hf", commands[0])
        self.assertIn("dolphinteam/AVES-OpenWhistle-Stage-2-320-100pct", commands[0])
        self.assertIn("dolphinteam/OpenWhistle-Wav2Vec2.0", commands[-1])
        self.assertNotIn("stride960-transformers", result.stdout)
        self.assertEqual(REBUTTAL_PRETRAINING_MODELS[-1]["model_id"],
                         "dolphinteam/OpenWhistle-Wav2Vec2.0")


if __name__ == "__main__":
    unittest.main()
