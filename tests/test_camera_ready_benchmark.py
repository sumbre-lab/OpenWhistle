"""Small regression checks for the integrated camera-ready benchmark."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/benchmark/src"))
import hf_datasets
from models import BioLingual
from plot_pr_micro import micro_pr_curve
from train_lr_splits import embed_split, score_predictions


class BenchmarkIntegrationTests(unittest.TestCase):
    def test_classification_uses_macro_f1(self):
        truth = np.array([0, 0, 0, 1, 2])
        predicted = np.zeros(5, dtype=int)
        actual = score_predictions("classification", truth, predicted)
        self.assertAlmostEqual(actual, f1_score(truth, predicted, average="macro"))
        self.assertNotAlmostEqual(actual, float(np.mean(truth == predicted)))

    def test_multiclass_pr_matches_binarized_targets(self):
        truth = np.array([0, 1, 2, 0, 1, 2])
        scores = np.array([[.8, .1, .1], [.3, .5, .2], [.2, .2, .6],
                           [.2, .6, .2], [.1, .8, .1], [.4, .4, .2]])
        _, _, actual = micro_pr_curve(truth, scores, np.arange(3))
        self.assertAlmostEqual(actual, average_precision_score(np.eye(3)[truth], scores, average="micro"))

    def test_umap_can_include_validation_without_changing_default(self):
        labels = ClassLabel(names=["a", "b"])
        splits = DatasetDict({name: Dataset.from_dict({"audio": [None], "label": [i % 2]}).cast_column("label", labels)
                              for i, name in enumerate(("train", "validation", "test"))})
        with patch.object(hf_datasets, "load_dataset", return_value=splits):
            default, _ = hf_datasets.load_classification_examples()
            all_rows, _ = hf_datasets.load_classification_examples(splits=("train", "validation", "test"))
        self.assertEqual(len(default), 2)
        self.assertEqual(len(all_rows), 3)

    def test_failed_audio_cannot_silently_shrink_test_set(self):
        def failing_model(audio):
            raise ValueError("cannot decode")
        with self.assertRaisesRegex(RuntimeError, "refusing a partial benchmark"):
            embed_split([{"audio": {"path": "broken.wav"}, "label": 0}], "test", "classification", failing_model)

    def test_biolingual_accepts_tensor_and_pooled_feature_outputs(self):
        backbone = BioLingual.__new__(BioLingual)
        backbone.device = torch.device("cpu")
        backbone.processor = SimpleNamespace(feature_extractor=lambda *args, **kwargs: {"input_features": torch.ones(1, 2)})
        for features in (torch.ones(1, 3), SimpleNamespace(pooler_output=torch.ones(1, 3))):
            backbone.model = SimpleNamespace(get_audio_features=lambda **kwargs: features)
            with patch("models.get_waveform", return_value=torch.ones(1, 100)):
                self.assertEqual(backbone("audio.wav").shape, torch.Size([3]))


if __name__ == "__main__":
    unittest.main()
