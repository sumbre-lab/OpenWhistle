#!/usr/bin/env python3
"""End-to-end mAP-only detection fine-tuning for one HF audio backbone."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SRC = SCRIPT_DIR.parents[1] / "benchmark" / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BENCHMARK_SRC))

from finetune_classification import set_seed  # noqa: E402
from finetune_detection import train_epoch, worker_init_fn  # noqa: E402
from finetune_detection_map import DetectionDataset, evaluate  # noqa: E402
from finetune_hf_classification import (  # noqa: E402
    HFAudioClassifier,
    HFAudioCollator,
)
from hf_datasets import load_detection_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument(
        "--feature_mode",
        choices=("standard", "btb3_concat", "btb3_mean"),
        default="standard",
    )
    parser.add_argument("--sample_rate", type=int, default=None)
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
    parser.add_argument("--max_duration_seconds", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--mixed_precision",
        choices=("none", "float16", "bfloat16"),
        default="float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be at least 1.")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, label_names = load_detection_splits()
    label_names = list(label_names)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    sample_rate = int(
        args.sample_rate
        if args.sample_rate is not None
        else getattr(feature_extractor, "sampling_rate", 44100)
    )
    if args.sample_rate is not None:
        feature_extractor.sampling_rate = sample_rate
    if args.feature_mode.startswith("btb3_") and sample_rate != 16000:
        raise ValueError("BTB3 feature modes require --sample_rate 16000.")

    collator = HFAudioCollator(
        feature_extractor,
        sample_rate,
        args.feature_mode,
        args.max_duration_seconds,
    )
    model = HFAudioClassifier(
        args.model_id,
        args.feature_mode,
        len(label_names),
        args.dropout,
    ).to(device)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": args.prefetch_factor,
                "worker_init_fn": worker_init_fn,
            }
        )
    loaders = {
        split_name: DataLoader(
            DetectionDataset(splits[split_name]),
            shuffle=split_name == "train",
            **loader_kwargs,
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
    use_scaler = device.type == "cuda" and args.mixed_precision == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    trainable_backbone = sum(
        parameter.numel()
        for parameter in backbone_parameters
        if parameter.requires_grad
    )
    trainable_head = sum(parameter.numel() for parameter in head_parameters)
    print(
        f"device={device} model={args.model_id} sample_rate={sample_rate} "
        f"feature_mode={args.feature_mode} labels={label_names}\n"
        f"trainable audio backbone parameters={trainable_backbone:,}; "
        f"detection head parameters={trainable_head:,}"
    )

    best_path = args.output_dir / "best.pt"
    history = []
    best_validation_map = -math.inf
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
            "validation": validation,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        if validation["map"] > best_validation_map:
            best_validation_map = validation["map"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.checkpoint_state(),
                    "epoch": epoch,
                    "validation_map": best_validation_map,
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
        "task": "multilabel_detection",
        "backbone": "hf",
        "model_id": args.model_id,
        "feature_mode": args.feature_mode,
        "dataset_config": "default",
        "seed": args.seed,
        "sample_rate": sample_rate,
        "label_names": label_names,
        "trainable_backbone_parameters": trainable_backbone,
        "trainable_head_parameters": trainable_head,
        "best_epoch": checkpoint["epoch"],
        "validation_map": checkpoint["validation_map"],
        "test": test,
        "history": history,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Test mAP: {test['map']:.4f}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
