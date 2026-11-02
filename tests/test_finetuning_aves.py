"""Validate verified native AVES weights and end-to-end backpropagation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/finetuning/src"))
import finetune_classification as training  # noqa: E402


class AvesFinetuningTests(unittest.TestCase):
    def test_native_export_is_verified_loaded_for_training_and_backpropagates(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 4)

            def extract_features(self, values, lengths):
                return [self.linear(values.unsqueeze(-1))], lengths

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, config = root / "weights.pt", root / "config.json"
            weights.write_bytes(b"native pretrained weights")
            config.write_text(json.dumps(dict(sample_rate=44100, encoder_embed_dim=4)))
            spec = dict(model_id="test/native-aves", revision="pinned-revision", sample_rate=44100,
                        model_filename=weights.name, config_filename=config.name,
                        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())

            def load(config_path, model_path, device, for_inference):
                self.assertFalse(for_inference)
                self.assertEqual(Path(model_path), weights)
                self.assertNotIn("sample_rate", json.loads(Path(config_path).read_text()))
                return SimpleNamespace(model=Encoder(), config=dict(encoder_embed_dim=4))

            with patch.object(training, "AVES_SCRATCH_MODELS", [spec]), \
                    patch("huggingface_hub.hf_hub_download",
                          side_effect=lambda **kw: str(root / kw["filename"])) as download, \
                    patch("aves.load_feature_extractor", side_effect=load):
                model = training.NativeAvesClassifier("test/native-aves", 3, 0.1, torch.device("cpu"))
                self.assertEqual(model.model_revision, "pinned-revision")
                self.assertTrue(all(p.requires_grad for p in model.backbone_parameters()))
                logits = model(torch.randn(2, 8), torch.tensor([8, 5]))
                self.assertEqual(tuple(logits.shape), (2, 3))
                logits.square().mean().backward()
                self.assertIsNotNone(model.encoder.linear.weight.grad)
                self.assertIsNotNone(model.classifier.weight.grad)
                self.assertTrue(all(c.kwargs["revision"] == spec["revision"] for c in download.call_args_list))
                self.assertEqual(json.loads(config.read_text())["sample_rate"], 44100)
                weights.write_bytes(b"wrong checkpoint")
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    training.NativeAvesClassifier("test/native-aves", 3, 0.1, torch.device("cpu"))



if __name__ == "__main__":
    unittest.main()
