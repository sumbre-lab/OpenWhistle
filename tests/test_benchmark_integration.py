"""Integration checks for official splits, audio decoding and benchmark metrics."""

import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from sklearn.metrics import f1_score

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments/benchmark/src")
)
import hf_datasets
from models import BioLingual, SpectralFeatures, get_waveform, load_waveform
from train_lr_splits import embed_split, score_predictions


class BenchmarkIntegrationTests(unittest.TestCase):
    def test_unsupported_classification_config_is_rejected_before_dataset_access(self):
        for loader in (hf_datasets.load_classification_splits,):
            with (
                self.subTest(loader=loader.__name__),
                patch.object(hf_datasets, "load_dataset") as load,
            ):
                with self.assertRaisesRegex(ValueError, "only unbalanced and balanced"):
                    loader(config_name="unsupported")
                load.assert_not_called()

    def test_embedded_audio_takes_precedence_over_unavailable_original_path(self):
        samples = np.stack(
            [np.linspace(-0.5, 0.5, 4096), np.linspace(0.2, -0.2, 4096)], axis=1
        )
        buffer = io.BytesIO()
        sf.write(buffer, samples, 44100, format="WAV", subtype="PCM_16")
        decoded, rate = sf.read(
            io.BytesIO(buffer.getvalue()), dtype="float32", always_2d=True
        )
        payload = {"bytes": buffer.getvalue(), "path": "/missing/hub-original.wav"}
        with patch(
            "models._load_waveform_from_file",
            side_effect=AssertionError("must use embedded bytes"),
        ):
            waveform, actual_rate = load_waveform(payload)
            actual = get_waveform(payload, 16000)
        np.testing.assert_array_equal(waveform, decoded.mean(axis=1))
        self.assertEqual(actual_rate, rate)
        expected = get_waveform({"array": decoded, "sampling_rate": rate}, 16000)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hub_loaders_and_feature_extraction_do_not_require_torchcodec(self):
        samples = np.sin(2 * np.pi * 6000 * np.arange(8192) / 44100).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, samples, 44100, format="WAV", subtype="PCM_16")
        payload = {"bytes": buffer.getvalue(), "path": "/missing/hub-original.wav"}
        for task in ("classification", "detection"):
            label_name = "label" if task == "classification" else "labels"
            label_feature = (
                ClassLabel(names=["a", "b"])
                if task == "classification"
                else Sequence(feature=Value("int64"), length=7)
            )
            labels = [0] if task == "classification" else [[1, 0, 0, 0, 0, 0, 0]]
            source = Dataset.from_dict(
                {"audio": [payload], label_name: labels},
                features=Features(
                    {
                        "audio": {"bytes": Value("binary"), "path": Value("string")},
                        label_name: label_feature,
                    }
                ),
            ).cast_column("audio", Audio(decode=True))
            splits = DatasetDict(
                {name: source for name in ("train", "validation", "test")}
            )
            loaders = (
                (hf_datasets.load_classification_splits,)
                if task == "classification"
                else (hf_datasets.load_detection_splits,)
            )
            for loader in loaders:
                with (
                    self.subTest(loader=loader.__name__),
                    patch.object(hf_datasets, "load_dataset", return_value=splits),
                    patch.dict(
                        sys.modules,
                        {"torchcodec": None, "datasets.features._torchcodec": None},
                    ),
                ):
                    dataset, _ = loader()
                    if isinstance(dataset, DatasetDict):
                        dataset = dataset["train"]
                    x, y = embed_split(dataset, "test", task, SpectralFeatures())
                    self.assertEqual(len(x), len(dataset))
                    self.assertEqual(len(y), len(dataset))
                    self.assertTrue(np.isfinite(x).all())

    def test_classification_uses_macro_f1(self):
        truth = np.array([0, 0, 0, 1, 2])
        predicted = np.zeros(5, dtype=int)
        actual = score_predictions("classification", truth, predicted)
        self.assertAlmostEqual(actual, f1_score(truth, predicted, average="macro"))
        self.assertNotAlmostEqual(actual, float(np.mean(truth == predicted)))

    def test_failed_audio_cannot_silently_shrink_test_set(self):
        def failing_model(audio):
            raise ValueError("cannot decode")

        with self.assertRaisesRegex(RuntimeError, "refusing a partial benchmark"):
            embed_split(
                [{"audio": {"path": "broken.wav"}, "label": 0}],
                "test",
                "classification",
                failing_model,
            )

    def test_biolingual_accepts_tensor_and_pooled_feature_outputs(self):
        backbone = BioLingual.__new__(BioLingual)
        backbone.device = torch.device("cpu")
        backbone.processor = SimpleNamespace(
            feature_extractor=lambda *args, **kwargs: {
                "input_features": torch.ones(1, 2)
            }
        )
        for features in (
            torch.ones(1, 3),
            SimpleNamespace(pooler_output=torch.ones(1, 3)),
        ):
            backbone.model = SimpleNamespace(
                get_audio_features=lambda **kwargs: features
            )
            with patch("models.get_waveform", return_value=torch.ones(1, 100)):
                self.assertEqual(backbone("audio.wav").shape, torch.Size([3]))


if __name__ == "__main__":
    unittest.main()
