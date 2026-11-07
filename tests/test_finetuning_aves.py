"""Validate pretrained encoder adapters, pooling and end-to-end backpropagation."""
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
import finetuning_models as training  # noqa: E402


class AvesFinetuningTests(unittest.TestCase):
    def test_hf_adapter_keeps_masked_pooling_and_three_view_order(self):
        class Encoder(torch.nn.Module):
            config = SimpleNamespace(hidden_size=2)

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_values, attention_mask):
                hidden = torch.stack((input_values, 2 * input_values), dim=-1) * self.scale
                return SimpleNamespace(last_hidden_state=hidden)

            def _get_feature_vector_attention_mask(self, _length, mask):
                return mask

        model_id = "dolphinteam/OpenWhistle-Wav2Vec2.0"
        with patch.object(training.AutoModel, "from_pretrained", side_effect=lambda *_a, **_kw: Encoder()) as load:
            standard = training.HFAudioClassifier(model_id, "standard", 2, 0.0)
            with torch.no_grad():
                standard.classifier.weight.copy_(torch.eye(2))
                standard.classifier.bias.zero_()
            output = standard(input_values=torch.tensor([[1.0, 3.0, 100.0]]),
                              attention_mask=torch.tensor([[1, 1, 0]]))
            torch.testing.assert_close(output, torch.tensor([[2.0, 4.0]]))
            output.sum().backward()
            self.assertIsNotNone(standard.encoder.scale.grad)
            views = training.HFAudioClassifier(model_id, "btb3_concat", 2, 0.0)
            with torch.no_grad():
                views.classifier.weight.zero_()
                views.classifier.weight[0, 0] = 1
                views.classifier.weight[1, 4] = 1
                views.classifier.bias.zero_()
            values = torch.arange(1, 7).float().reshape(6, 1).repeat(1, 2)
            output = views(input_values=values, attention_mask=torch.ones_like(values))
            torch.testing.assert_close(output, torch.tensor([[1.0, 3.0], [4.0, 6.0]]))
            self.assertTrue(all(call.kwargs["revision"] == training.MODEL_REVISIONS[model_id]
                                for call in load.call_args_list))

    def test_biolingual_adapter_trains_audio_and_projection_and_freezes_text(self):
        class Clap(torch.nn.Module):
            config = SimpleNamespace(projection_dim=2)

            def __init__(self):
                super().__init__()
                self.audio_model = torch.nn.Linear(3, 4)
                self.audio_projection = torch.nn.Linear(4, 2)
                self.text_model = torch.nn.Linear(3, 4)
                self.text_projection = torch.nn.Linear(4, 2)

            def get_audio_features(self, input_features):
                return self.audio_projection(self.audio_model(input_features))

        with patch.object(training.ClapModel, "from_pretrained", return_value=Clap()):
            model = training.BioLingualClassifier("davidrrobinson/biolingual", 3, 0.0)
            output = model(input_features=torch.randn(2, 3))
            self.assertEqual(tuple(output.shape), (2, 3))
            output.square().mean().backward()
            for parameter in model.backbone_parameters():
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad)
            for tower in (model.backbone.text_model, model.backbone.text_projection):
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in tower.parameters()))

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
