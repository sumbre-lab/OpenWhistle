# ruff: noqa: E402  # Benchmark scripts are imported after adding their source directory.
import hashlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "experiments/benchmark/src"
sys.path.insert(0, str(SRC))
from aves_models import AVES_SCRATCH_MODELS


class PretrainingBackboneTests(unittest.TestCase):
    def test_official_aves_downloads_are_verified_and_source_config_is_preserved(self):
        artifacts = importlib.import_module("aves_artifacts")
        payloads = {
            "model-url": b"official AVES weights",
            "config-url": b'{"sample_rate":44100,"encoder_num_layers":12}',
        }
        spec = dict(
            model_filename="model.pt",
            config_filename="config.json",
            model_url="model-url",
            config_url="config-url",
            sample_rate=44100,
            model_sha256=hashlib.sha256(payloads["model-url"]).hexdigest(),
            config_sha256=hashlib.sha256(payloads["config-url"]).hexdigest(),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(artifacts, "AVES_CACHE_DIR", Path(directory)),
            patch.object(artifacts, "AVES_VARIANTS", {"bio": spec}),
            patch.object(
                artifacts,
                "urlopen",
                side_effect=lambda url, **kw: io.BytesIO(payloads[url]),
            ) as fetch,
        ):
            model, config = artifacts.get_aves_paths("bio")
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(Path(model).read_bytes(), payloads["model-url"])
            self.assertNotIn("sample_rate", json.loads(Path(config).read_text()))
            self.assertEqual(
                (Path(directory) / "config.json").read_bytes(), payloads["config-url"]
            )
            self.assertEqual(artifacts.get_aves_paths("bio"), (model, config))
            self.assertEqual(fetch.call_count, 2)

    def test_corrupt_aves_download_does_not_replace_an_existing_cache_entry(self):
        artifacts = importlib.import_module("aves_artifacts")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.pt"
            destination.write_bytes(b"existing weights")
            with patch.object(
                artifacts, "urlopen", return_value=io.BytesIO(b"corrupt download")
            ):
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    artifacts._download_file(
                        "model-url",
                        destination,
                        hashlib.sha256(b"expected").hexdigest(),
                    )
            self.assertEqual(destination.read_bytes(), b"existing weights")
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_training_cli_accepts_native_aves_backend_emitted_by_runner(self):
        train = importlib.import_module("train_lr_splits")
        with patch.object(
            sys,
            "argv",
            [
                "train_lr_splits.py",
                "--model",
                "aves_hf",
                "--hf_model_id",
                AVES_SCRATCH_MODELS[0]["model_id"],
                "--hf_local_dir",
                "/tmp/native-aves-export",
                "--dataset_name",
                "classification",
                "--dataset_config",
                "all",
            ],
        ):
            args = train.get_args()
        self.assertEqual(args.model, "aves_hf")
        self.assertEqual(args.dataset_config, "all")
        self.assertEqual(args.hf_local_dir, Path("/tmp/native-aves-export"))

    def test_native_export_uses_versioned_files_without_modifying_download(self):
        models = importlib.import_module("models")
        spec = AVES_SCRATCH_MODELS[0]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "original.json"
            original = '{"sample_rate":44100,"encoder_num_layers":12}'
            config.write_text(original)
            weights = Path(directory) / "weights.pt"
            weights.write_bytes(b"test stub")

            def download(**kwargs):
                self.assertEqual(kwargs["repo_id"], spec["model_id"])
                self.assertEqual(kwargs["revision"], spec["revision"])
                return str(
                    config if kwargs["filename"] == spec["config_filename"] else weights
                )

            with (
                patch("huggingface_hub.hf_hub_download", side_effect=download) as fetch,
                patch.object(models.Aves, "__init__", return_value=None) as initialize,
            ):
                backbone = models.HuggingFaceAvesBackbone(
                    spec["model_id"],
                    hf_sample_rate=44100,
                    sample_rate=44100,
                    aves_model_path="",
                    aves_config_path="",
                )
                self.assertEqual(fetch.call_count, 2)
                arguments = initialize.call_args.kwargs
                self.assertEqual(arguments["aves_model_path"], str(weights))
                self.assertEqual(arguments["sample_rate"], 44100)
                self.assertEqual(
                    json.loads(Path(arguments["aves_config_path"]).read_text()),
                    {"encoder_num_layers": 12},
                )
                self.assertEqual(config.read_text(), original)
                backbone._config_directory.cleanup()

    def test_local_native_export_checks_both_hashes_without_hub_access(self):
        models = importlib.import_module("models")
        registry = importlib.import_module("aves_models")
        original_spec = AVES_SCRATCH_MODELS[0]
        for tamper in (None, "weights", "config"):
            with (
                self.subTest(tamper=tamper),
                tempfile.TemporaryDirectory() as directory,
            ):
                spec = dict(original_spec)
                weights = Path(directory) / spec["model_filename"]
                config = Path(directory) / spec["config_filename"]
                weights.parent.mkdir(parents=True)
                config.parent.mkdir(parents=True)
                weights.write_bytes(b"test weights")
                original_config = '{"sample_rate":44100,"encoder_num_layers":12}'
                config.write_text(original_config)
                spec["weights_sha256"] = hashlib.sha256(
                    weights.read_bytes()
                ).hexdigest()
                spec["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
                if tamper == "weights":
                    weights.write_bytes(b"different checkpoint")
                elif tamper == "config":
                    config.write_text('{"sample_rate":16000}')
                with (
                    patch.object(registry, "AVES_SCRATCH_MODELS", (spec,)),
                    patch(
                        "huggingface_hub.hf_hub_download",
                        side_effect=AssertionError("must stay local"),
                    ),
                    patch.object(
                        models.Aves, "__init__", return_value=None
                    ) as initialize,
                ):
                    if tamper:
                        with self.assertRaisesRegex(
                            ValueError, "Local file does not match"
                        ):
                            models.HuggingFaceAvesBackbone(
                                spec["model_id"], hf_local_dir=directory
                            )
                        initialize.assert_not_called()
                    else:
                        backbone = models.HuggingFaceAvesBackbone(
                            spec["model_id"], hf_local_dir=directory
                        )
                        self.assertEqual(backbone.revision, spec["revision"])
                        self.assertEqual(
                            initialize.call_args.kwargs["aves_model_path"], str(weights)
                        )
                        self.assertEqual(config.read_text(), original_config)
                        backbone._config_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
