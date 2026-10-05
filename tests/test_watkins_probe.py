import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/benchmark/src/train_lr_watkins.py"
spec = importlib.util.spec_from_file_location("watkins", MODULE_PATH)
watkins = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watkins)


class WatkinsProbeTests(unittest.TestCase):
    def test_selection_uses_macro_f1_even_when_accuracy_is_lower(self):
        truth = np.array([0] * 80 + [1] * 20)
        predictions = [np.zeros(100, dtype=int), np.array([0] * 50 + [1] * 50)]
        sweep = [{"C": c, "valid_accuracy": accuracy_score(truth, pred),
                  "valid_macro_f1": f1_score(truth, pred, average="macro")}
                 for c, pred in zip([.1, 1], predictions)]
        self.assertEqual(watkins.select_candidate(sweep)["C"], 1)
        self.assertEqual(watkins.select_candidate(sweep, "valid_accuracy")["C"], .1)

    def test_bootstrap_keeps_absent_classes_in_macro_average(self):
        truth = np.zeros(10, dtype=int)
        samples = watkins.bootstrap_macro_f1(truth, truth, [0, 1], 10, 42)
        np.testing.assert_array_equal(samples, np.full(10, .5))

    def test_cache_rejects_duplicate_audio_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            pd.DataFrame({"audio_path": ["a/x.wav", "a/y.wav", "a/x.wav"],
                          "label": ["a"] * 3, "split": ["train", "valid", "test"]}).to_csv(
                              cache / "embedding_index.csv", index=False)
            np.save(cache / "embeddings.npy", np.zeros((3, 2)))
            with self.assertRaisesRegex(ValueError, "split leakage"):
                watkins.load_cache(cache)


if __name__ == "__main__":
    unittest.main()
