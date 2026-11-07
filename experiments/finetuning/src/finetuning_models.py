"""Trainable audio encoders and collators matching the frozen benchmark frontends."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel, ClapModel, ClapProcessor

from finetuning_config import HF_MODELS
from aves_artifacts import get_aves_paths, get_aves_sample_rate
from aves_models import AVES_SCRATCH_MODELS
from benchmark_revisions import MODEL_REVISIONS
from models import make_btb3_views


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    max_duration_seconds: float | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(waveform, dtype=torch.float32)
    if sample_rate != target_sample_rate:
        tensor = torchaudio.functional.resample(
            tensor,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
    if max_duration_seconds is not None:
        max_samples = int(round(max_duration_seconds * target_sample_rate))
        tensor = tensor[:max_samples]
    return tensor


class AvesCollator:
    def __init__(self, sample_rate: int, max_duration_seconds: float | None):
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds

    def __call__(self, examples):
        waveforms = [
            prepare_waveform(
                waveform,
                sample_rate,
                self.sample_rate,
                self.max_duration_seconds,
            )
            for waveform, sample_rate, _ in examples
        ]
        lengths = torch.tensor([len(waveform) for waveform in waveforms])
        return {
            "input_values": nn.utils.rnn.pad_sequence(
                waveforms,
                batch_first=True,
            ),
            "input_lengths": lengths,
            "labels": torch.tensor([example[2] for example in examples]),
        }


class BioLingualCollator:
    def __init__(
        self,
        processor: ClapProcessor,
        sample_rate: int,
        max_duration_seconds: float | None,
    ):
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds

    def __call__(self, examples):
        waveforms = [
            prepare_waveform(
                waveform,
                sample_rate,
                self.sample_rate,
                self.max_duration_seconds,
            ).numpy()
            for waveform, sample_rate, _ in examples
        ]
        inputs = self.processor(
            audio=waveforms,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        batch = {
            key: value
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        batch["labels"] = torch.tensor([example[2] for example in examples])
        return batch


class AvesClassifier(nn.Module):
    def __init__(
        self,
        variant: str,
        num_labels: int,
        dropout: float,
        device: torch.device,
        model_path=None,
        config_path=None,
    ):
        super().__init__()
        from aves import load_feature_extractor

        if model_path is None or config_path is None:
            model_path, config_path = get_aves_paths(variant)
        wrapper = load_feature_extractor(
            config_path=config_path,
            model_path=model_path,
            device=device.type,
            for_inference=False,
        )
        self.encoder = wrapper.model
        hidden_size = int(wrapper.config["encoder_embed_dim"])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        layer_outputs, output_lengths = self.encoder.extract_features(
            input_values,
            input_lengths,
        )
        hidden = layer_outputs[-1]
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask = positions < output_lengths.unsqueeze(1)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))

    def backbone_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        return self.classifier.parameters()

    def checkpoint_state(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.classifier.load_state_dict(state["classifier"])


class NativeAvesClassifier(AvesClassifier):
    """Train the same pinned native OpenWhistle export as the frozen benchmark."""

    def __init__(self, model_id, num_labels, dropout, device):
        from huggingface_hub import hf_hub_download

        spec = next((s for s in AVES_SCRATCH_MODELS if s["model_id"] == model_id), None)
        if spec is None:
            raise ValueError(f"Unknown native AVES export: {model_id}")
        paths = {}
        for kind in ("model", "config"):
            path = Path(hf_hub_download(repo_id=model_id, revision=spec["revision"],
                                        filename=spec[f"{kind}_filename"]))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = spec["weights_sha256" if kind == "model" else "config_sha256"]
            if digest != expected:
                raise ValueError(f"Native AVES {kind} hash differs from the benchmark: {path}")
            paths[kind] = path
        config = json.loads(paths["config"].read_text())
        if config.pop("sample_rate", spec["sample_rate"]) != spec["sample_rate"]:
            raise ValueError("Unexpected native AVES sample rate.")
        with tempfile.TemporaryDirectory(prefix="openwhistle-aves-finetune-") as directory:
            config_path = Path(directory) / "model_config.json"
            config_path.write_text(json.dumps(config))
            super().__init__("scratch", num_labels, dropout, device,
                             model_path=str(paths["model"]), config_path=str(config_path))
        self.model_revision = spec["revision"]


class BioLingualClassifier(nn.Module):
    def __init__(self, model_id: str, num_labels: int, dropout: float):
        super().__init__()
        self.backbone = ClapModel.from_pretrained(model_id, revision=MODEL_REVISIONS.get(model_id))
        # The text tower is not part of audio classification. Keeping it frozen
        # avoids presenting unused CLAP parameters as fine-tuned parameters.
        self.backbone.text_model.requires_grad_(False)
        self.backbone.text_projection.requires_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            int(self.backbone.config.projection_dim),
            num_labels,
        )

    def forward(self, **inputs) -> torch.Tensor:
        outputs = self.backbone.get_audio_features(**inputs)
        features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        return self.classifier(self.dropout(features))

    def backbone_parameters(self):
        yield from self.backbone.audio_model.parameters()
        yield from self.backbone.audio_projection.parameters()

    def head_parameters(self):
        return self.classifier.parameters()

    def checkpoint_state(self) -> dict:
        return {
            "audio_model": self.backbone.audio_model.state_dict(),
            "audio_projection": self.backbone.audio_projection.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.backbone.audio_model.load_state_dict(state["audio_model"])
        self.backbone.audio_projection.load_state_dict(state["audio_projection"])
        self.classifier.load_state_dict(state["classifier"])


class HFAudioCollator:
    def __init__(
        self,
        feature_extractor,
        sample_rate: int,
        feature_mode: str,
        max_duration_seconds: float | None,
    ):
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.feature_mode = feature_mode
        self.max_duration_seconds = max_duration_seconds

    def __call__(self, examples):
        waveforms = []
        if self.feature_mode == "standard":
            waveforms = [
                prepare_waveform(
                    waveform,
                    source_rate,
                    self.sample_rate,
                    self.max_duration_seconds,
                ).numpy()
                for waveform, source_rate, _ in examples
            ]
        else:
            max_samples = (
                round(self.max_duration_seconds * self.sample_rate)
                if self.max_duration_seconds is not None
                else None
            )
            for waveform, source_rate, _ in examples:
                views = make_btb3_views(waveform, source_rate)
                waveforms.extend(
                    view[:max_samples] if max_samples is not None else view
                    for view in views
                )

        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        batch = {
            key: value
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        batch["labels"] = torch.tensor([example[2] for example in examples])
        return batch


class HFAudioClassifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        feature_mode: str,
        num_labels: int,
        dropout: float,
    ):
        super().__init__()
        self.model_revision = MODEL_REVISIONS.get(model_id)
        self.encoder = AutoModel.from_pretrained(model_id, revision=self.model_revision)
        self.feature_mode = feature_mode
        hidden_size = int(self.encoder.config.hidden_size)
        head_size = hidden_size * 3 if feature_mode == "btb3_concat" else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_size, num_labels)

    def _masked_pool(self, hidden, attention_mask):
        if attention_mask is None:
            return hidden.mean(dim=1)
        if hasattr(self.encoder, "_get_feature_vector_attention_mask"):
            attention_mask = self.encoder._get_feature_vector_attention_mask(
                hidden.shape[1],
                attention_mask,
            )
        elif attention_mask.shape[1] != hidden.shape[1]:
            return hidden.mean(dim=1)
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def forward(self, **inputs):
        attention_mask = inputs.get("attention_mask")
        outputs = self.encoder(**inputs)
        pooled = self._masked_pool(outputs.last_hidden_state, attention_mask)
        if self.feature_mode.startswith("btb3_"):
            pooled = pooled.reshape(-1, 3, pooled.shape[-1])
            if self.feature_mode == "btb3_concat":
                pooled = pooled.reshape(pooled.shape[0], -1)
            else:
                pooled = pooled.mean(dim=1)
        return self.classifier(self.dropout(pooled))

    def backbone_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        return self.classifier.parameters()

    def checkpoint_state(self):
        return {
            "encoder": self.encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_checkpoint_state(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.classifier.load_state_dict(state["classifier"])


def make_model_and_collator(
    args: argparse.Namespace,
    num_labels: int,
    device: torch.device,
):
    if args.model in HF_MODELS:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.model_id, revision=MODEL_REVISIONS.get(args.model_id))
        sample_rate = int(args.sample_rate if args.sample_rate is not None else
                          getattr(feature_extractor, "sampling_rate", 44100))
        if args.sample_rate is not None:
            feature_extractor.sampling_rate = sample_rate
        if args.feature_mode.startswith("btb3_") and sample_rate != 16000:
            raise ValueError("BTB3 feature modes require --sample_rate 16000.")
        collator = HFAudioCollator(feature_extractor, sample_rate, args.feature_mode,
                                   args.max_duration_seconds)
        # Preserve selection initialization after frontend setup for HF classification.
        if args.dataset_name == "classification" and args.refit_train_validation:
            set_seed(args.seed)
        model = HFAudioClassifier(args.model_id, args.feature_mode, num_labels, args.dropout)
        return model, collator, sample_rate
    if args.model == "aves_scratch":
        model = NativeAvesClassifier(args.model_id, num_labels, args.dropout, device)
        return model, AvesCollator(44100, args.max_duration_seconds), 44100
    if args.model.startswith("aves_"):
        variant = args.model.removeprefix("aves_")
        sample_rate = get_aves_sample_rate(variant)
        model = AvesClassifier(
            variant=variant,
            num_labels=num_labels,
            dropout=args.dropout,
            device=device,
        )
        collator = AvesCollator(sample_rate, args.max_duration_seconds)
        return model, collator, sample_rate

    processor = ClapProcessor.from_pretrained(args.model_id, revision=MODEL_REVISIONS.get(args.model_id))
    model = BioLingualClassifier(args.model_id, num_labels, args.dropout)
    sample_rate = int(processor.feature_extractor.sampling_rate)
    collator = BioLingualCollator(
        processor,
        sample_rate,
        args.max_duration_seconds,
    )
    return model, collator, sample_rate


def make_refit_model(args, num_labels, device):
    """Restore pretrained weights with the original per-backend initialization order."""
    if args.model in HF_MODELS:
        return HFAudioClassifier(args.model_id, args.feature_mode, num_labels, args.dropout)
    return make_model_and_collator(args, num_labels, device)[0]
