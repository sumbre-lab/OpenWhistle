"""Check independent refitting, split isolation and test uncertainty without a GPU."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/finetuning/src"))
import finetune as training  # noqa: E402
from metrics import bootstrap_metric_scores, score_predictions  # noqa: E402


class RefitTests(unittest.TestCase):
    def test_refit_restarts_weights_and_optimizer_and_tests_only_final_model(self):
        class TinyModel(torch.nn.Module):
            model_revision = training.MODEL_REVISIONS["dolphinteam/OpenWhistle-Wav2Vec2.0"]

            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Parameter(torch.tensor(0.0))
                self.head = torch.nn.Parameter(torch.tensor(0.0))

            def backbone_parameters(self):
                return [self.encoder]

            def head_parameters(self):
                return [self.head]

            def checkpoint_state(self):
                return self.state_dict()

            def load_checkpoint_state(self, state):
                self.load_state_dict(state)

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "refit"
            command = ["finetune", "--model", "wav2vec2", "--dataset_name", "classification",
                       "--output_dir", str(output), "--dataset_config", "all",
                       "--epochs", "6", "--patience", "1", "--batch_size", "2",
                       "--eval_batch_size", "1", "--eval_precision", "none",
                       "--mixed_precision", "none", "--num_workers", "0",
                       "--refit_train_validation", "--num_bootstrap", "20"]
            with patch.object(sys, "argv", command):
                args = training.parse_args()
            splits = dict(train=["train0", "train1"], validation=["validation0"],
                          test=["test0", "test1", "test2"])
            models, train_calls, evaluations = [], [], []

            def make_model(*_args):
                model = TinyModel()
                models.append(model)
                return model

            def train(model, loader, optimizer, scheduler, scaler, device, args, **kwargs):
                is_selection = model is models[0]
                if not is_selection and len(train_calls) == 3:
                    self.assertEqual(float(model.encoder.detach()), 0.0)
                    self.assertIsNot(optimizer, train_calls[0][1])
                    self.assertEqual(optimizer.state, {})
                    self.assertEqual(scheduler.last_epoch, 0)
                self.assertEqual(scheduler.lr_lambdas[0].keywords["num_training_steps"],
                                 math.ceil(len(loader) / args.gradient_accumulation_steps) * args.epochs)
                data = list(loader.dataset)
                self.assertEqual(data, splits["train"] if is_selection else
                                 splits["train"] + splits["validation"])
                train_calls.append((model, optimizer))
                with torch.no_grad():
                    model.encoder.add_(10 if is_selection else 1)
                return 0.1

            def evaluate(model, loader, device, precision, *, dataset_name):
                self.assertEqual(dataset_name, "classification")
                data = list(loader.dataset)
                evaluations.append(data)
                self.assertEqual(precision, "none")
                if data == splits["validation"]:
                    self.assertIs(model, models[0])
                    return dict(loss=0.1, macro_f1=[0.7, 0.8, 0.6][len(evaluations) - 1], accuracy=0.5)
                self.assertEqual(data, splits["test"])
                self.assertIs(model, models[1])
                self.assertEqual(len(train_calls), 5)
                self.assertTrue((output / "final.pt").is_file())
                self.assertEqual(float(model.encoder.detach()), 2.0)
                y, pred = np.array([0, 1, 0]), np.array([0, 1, 1])
                return dict(loss=0.1, macro_f1=score_predictions("classification", y, pred),
                            accuracy=2 / 3, labels=y.tolist(), predictions=pred.tolist())

            stack.enter_context(patch.object(training, "parse_args", return_value=args))
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            stack.enter_context(patch.object(training, "load_classification_splits",
                                             return_value=(splits, SimpleNamespace(names=["a", "b"]))))
            stack.enter_context(patch.object(training, "AudioDataset", side_effect=lambda x, _task: x))
            stack.enter_context(patch.object(training, "make_model_and_collator",
                                             side_effect=lambda *_args: (make_model(), lambda x: x, 44100)))
            stack.enter_context(patch.object(training, "make_refit_model", side_effect=make_model))
            stack.enter_context(patch.object(training, "train_epoch", side_effect=train))
            stack.enter_context(patch.object(training, "evaluate", side_effect=evaluate))
            with redirect_stdout(io.StringIO()):
                training.main()
            self.assertEqual(len(models), 2)
            self.assertEqual(evaluations, [splits["validation"]] * 3 + [splits["test"]])
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["validation_macro_f1"], 0.8)
            self.assertEqual(result["final_training"]["split"], "train+validation")
            self.assertEqual(result["final_training"]["schedule_horizon_epochs"], 6)
            self.assertEqual(result["final_training"]["num_examples"], 3)
            self.assertEqual(len(result["refit_history"]), 2)
            selected = torch.load(output / "best.pt", weights_only=False)
            final = torch.load(output / "final.pt", weights_only=False)
            self.assertEqual(float(selected["model"]["encoder"]), 20.0)
            self.assertEqual(float(final["model"]["encoder"]), 2.0)
            with np.load(output / "test_predictions.npz") as data:
                expected = bootstrap_metric_scores("classification", data["y_true"], data["y_pred"], 20, 42)
            with np.load(output / "bootstrap_scores.npz") as data:
                np.testing.assert_array_equal(data["scores"], expected)
            self.assertEqual(result["test_bootstrap"]["std"], float(np.std(expected, ddof=1)))
            with self.assertRaisesRegex(ValueError, "new or empty"):
                training.main()


if __name__ == "__main__":
    unittest.main()
