#!/usr/bin/env python3
"""Supervised end-to-end fine-tuning on OpenWhistle Classification."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor, get_linear_schedule_with_warmup


BENCHMARK_SRC = Path(__file__).resolve().parents[2] / "benchmark" / "src"
sys.path.insert(0, str(BENCHMARK_SRC))

from conf import get_aves_paths, get_aves_sample_rate  # noqa: E402
from hf_datasets import (  # noqa: E402
    CLASSIFICATION_BALANCED_CONFIG_NAME,
    load_classification_splits,
)
from models import load_waveform  # noqa: E402


DEFAULT_BIOLINGUAL_ID = "davidrrobinson/biolingual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an AVES or BioLingual audio encoder and a classification "
            "head on OpenWhistle Classification."
        )
    )
    parser.add_argument(
        "--backbone",
        choices=("aves_bio", "aves_core", "biolingual"),
        required=True,
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_BIOLINGUAL_ID,
        help="BioLingual Hugging Face model ID (ignored for AVES).",
    )
    parser.add_argument(
        "--dataset_config",
        default=CLASSIFICATION_BALANCED_CONFIG_NAME,
        choices=("balanced", "unbalanced", "all"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--max_duration_seconds",
        type=float,
        default=None,
        help="Optionally truncate clips after resampling; default keeps full clips.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mixed_precision",
        choices=("none", "float16", "bfloat16"),
        default="float16",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClassificationDataset(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        row = self.split[index]
        waveform, sample_rate = load_waveform(row["audio"])
        return waveform, sample_rate, int(row["label"])


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
    ):
        super().__init__()
        from aves import load_feature_extractor

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


class BioLingualClassifier(nn.Module):
    def __init__(self, model_id: str, num_labels: int, dropout: float):
        super().__init__()
        self.backbone = ClapModel.from_pretrained(model_id)
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
        return self.classifier(self.dropout(outputs.pooler_output))

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


def move_batch(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    labels = batch.pop("labels").to(device, non_blocking=True)
    inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }
    return inputs, labels


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision != "none"
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
) -> dict:
    model.eval()
    losses = []
    labels = []
    predictions = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        inputs, target = move_batch(batch, device)
        with autocast_context(device, precision):
            logits = model(**inputs)
            loss = F.cross_entropy(logits, target)
        losses.append(float(loss))
        labels.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "labels": labels,
        "predictions": predictions,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        inputs, labels = move_batch(batch, device)
        with autocast_context(device, args.mixed_precision):
            logits = model(**inputs)
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=args.label_smoothing,
            )
            scaled_loss = loss / args.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        losses.append(float(loss.detach()))

        should_step = (
            step % args.gradient_accumulation_steps == 0 or step == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def make_model_and_collator(
    args: argparse.Namespace,
    num_labels: int,
    device: torch.device,
):
    if args.backbone.startswith("aves_"):
        variant = args.backbone.removeprefix("aves_")
        sample_rate = get_aves_sample_rate(variant)
        model = AvesClassifier(
            variant=variant,
            num_labels=num_labels,
            dropout=args.dropout,
            device=device,
        )
        collator = AvesCollator(sample_rate, args.max_duration_seconds)
        return model, collator, sample_rate

    processor = ClapProcessor.from_pretrained(args.model_id)
    model = BioLingualClassifier(args.model_id, num_labels, args.dropout)
    sample_rate = int(processor.feature_extractor.sampling_rate)
    collator = BioLingualCollator(
        processor,
        sample_rate,
        args.max_duration_seconds,
    )
    return model, collator, sample_rate


def public_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"labels", "predictions"}
    }


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be at least 1.")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, label_feature = load_classification_splits(args.dataset_config)
    label_names = list(label_feature.names)
    model, collator, sample_rate = make_model_and_collator(
        args,
        len(label_names),
        device,
    )
    model.to(device)

    loaders = {
        split_name: DataLoader(
            ClassificationDataset(splits[split_name]),
            batch_size=args.batch_size,
            shuffle=split_name == "train",
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split_name in ("train", "validation", "test")
    }

    backbone_parameters = list(model.backbone_parameters())
    head_parameters = list(model.head_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(loaders["train"]) / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    use_scaler = (
        device.type == "cuda" and args.mixed_precision == "float16"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    trainable_backbone = sum(
        parameter.numel()
        for parameter in backbone_parameters
        if parameter.requires_grad
    )
    trainable_head = sum(parameter.numel() for parameter in head_parameters)
    print(
        f"device={device} sample_rate={sample_rate} labels={label_names}\n"
        f"trainable audio backbone parameters={trainable_backbone:,}; "
        f"classification head parameters={trainable_head:,}"
    )
    if trainable_backbone == 0:
        raise RuntimeError("The audio backbone is frozen; end-to-end fine-tuning aborted.")

    best_path = args.output_dir / "best.pt"
    history = []
    best_validation_f1 = -math.inf
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scheduler,
            scaler,
            device,
            args,
        )
        validation = evaluate(
            model,
            loaders["validation"],
            device,
            args.mixed_precision,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": public_metrics(validation),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        if validation["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.checkpoint_state(),
                    "epoch": epoch,
                    "validation_macro_f1": best_validation_f1,
                    "label_names": label_names,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_checkpoint_state(checkpoint["model"])
    test = evaluate(model, loaders["test"], device, args.mixed_precision)
    results = {
        "protocol": "supervised_end_to_end_finetuning",
        "backbone": args.backbone,
        "model_id": args.model_id if args.backbone == "biolingual" else args.backbone,
        "dataset_config": args.dataset_config,
        "seed": args.seed,
        "sample_rate": sample_rate,
        "label_names": label_names,
        "trainable_backbone_parameters": trainable_backbone,
        "trainable_head_parameters": trainable_head,
        "best_epoch": checkpoint["epoch"],
        "validation_macro_f1": checkpoint["validation_macro_f1"],
        "test": public_metrics(test),
        "test_confusion_matrix": confusion_matrix(
            test["labels"],
            test["predictions"],
            labels=list(range(len(label_names))),
        ).tolist(),
        "history": history,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Test macro-F1: {test['macro_f1']:.4f}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
