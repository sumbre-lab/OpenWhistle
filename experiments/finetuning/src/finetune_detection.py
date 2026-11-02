#!/usr/bin/env python3
"""Supervised end-to-end fine-tuning on OpenWhistle Detection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SRC = SCRIPT_DIR.parents[1] / "benchmark" / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BENCHMARK_SRC))

from finetune_classification import (  # noqa: E402
    autocast_context,
    make_model_and_collator,
    move_batch,
    public_metrics,
    set_seed,
    save_test_predictions,
    model_identity,
)
from hf_datasets import (  # noqa: E402
    get_detection_label_vector,
    load_detection_splits,
)
from metrics import MeanAveragePrecision  # noqa: E402
from models import load_waveform  # noqa: E402
from refit_protocol import add_refit_argument, refit_model, validate_refit_output  # noqa: E402
from checkpoint_io import save_checkpoint  # noqa: E402
from report_io import write_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        choices=("aves_bio", "aves_core", "aves_scratch", "biolingual"),
        required=True,
    )
    parser.add_argument("--model_id", default="davidrrobinson/biolingual")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Validation/test batch size; defaults to training batch size.")
    parser.add_argument("--eval_precision", choices=("none", "float16", "bfloat16"),
                        default=None, help="Validation/test autocast; defaults to training precision.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_duration_seconds", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--mixed_precision",
        choices=("none", "float16", "bfloat16"),
        default="float16",
    )
    add_refit_argument(parser)
    return parser.parse_args()


class DetectionDataset(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        row = self.split[index]
        waveform, sample_rate = load_waveform(row["audio"])
        labels = get_detection_label_vector(row)
        return waveform, sample_rate, labels


def worker_init_fn(_worker_id: int) -> None:
    torch.set_num_threads(1)


@torch.no_grad()
def evaluate(model, loader, device, precision) -> dict:
    model.eval()
    losses = []
    targets = []
    scores = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        inputs, target = move_batch(batch, device)
        target = target.float()
        with autocast_context(device, precision):
            logits = model(**inputs)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        losses.append(float(loss))
        targets.append(target.cpu())
        scores.append(logits.sigmoid().float().cpu())

    map_metric = MeanAveragePrecision()
    map_metric.update(torch.cat(scores), torch.cat(targets))
    return {
        "loss": float(np.mean(losses)),
        "map": float(map_metric.get_primary_metric()),
        "labels": torch.cat(targets).tolist(),
        "predictions": torch.cat(scores).tolist(),
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        inputs, labels = move_batch(batch, device)
        labels = labels.float()
        with autocast_context(device, args.mixed_precision):
            logits = model(**inputs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
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


def main() -> None:
    args = parse_args()
    validate_refit_output(args)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be at least 1.")
    if args.batch_size < 1 or (args.eval_batch_size is not None and args.eval_batch_size < 1):
        raise ValueError("Batch sizes must be positive.")
    eval_precision = args.eval_precision or args.mixed_precision
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, label_names = load_detection_splits()
    label_names = list(label_names)
    model, collator, sample_rate = make_model_and_collator(
        args,
        len(label_names),
        device,
    )
    model.to(device)

    loader_kwargs = {
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
            batch_size=(args.batch_size if split_name == "train"
                        else args.eval_batch_size or args.batch_size),
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
    if trainable_backbone == 0:
        raise RuntimeError("The audio backbone is frozen; end-to-end fine-tuning aborted.")
    print(
        f"device={device} sample_rate={sample_rate} labels={label_names}\n"
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
            eval_precision,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": public_metrics(validation),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        if validation["map"] > best_validation_map:
            best_validation_map = validation["map"]
            epochs_without_improvement = 0
            save_checkpoint(
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

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    best_epoch = checkpoint["epoch"]
    final_training = dict(split="train", epochs=best_epoch, checkpoint="best.pt")
    refit_history = []
    if args.refit_train_validation:
        del model, optimizer, scheduler, scaler, backbone_parameters, head_parameters, checkpoint
        model, final_training, refit_history = refit_model(
            args, best_epoch, lambda: make_model_and_collator(args, len(label_names), device)[0],
            loaders, loader_kwargs, train_epoch, device, set_seed, label_names,
            "map", best_validation_map)
    else:
        model.load_checkpoint_state(checkpoint["model"])
    test = evaluate(model, loaders["test"], device, eval_precision)
    save_test_predictions(args.output_dir, test)
    results = {
        "protocol": "supervised_end_to_end_finetuning",
        "task": "multilabel_detection",
        "backbone": args.backbone,
        "model_id": model_identity(args)[0],
        "model_revision": model_identity(args)[1],
        "dataset_config": "default",
        "training_args": json.loads(json.dumps(vars(args), default=str)),
        "seed": args.seed,
        "sample_rate": sample_rate,
        "label_names": label_names,
        "trainable_backbone_parameters": trainable_backbone,
        "trainable_head_parameters": trainable_head,
        "best_epoch": best_epoch,
        "validation_map": best_validation_map,
        "validation_score_scope": "Selection model trained on train only.",
        "final_training": final_training,
        "refit_history": refit_history,
        "test": public_metrics(test),
        "history": history,
    }
    results_path = args.output_dir / "results.json"
    write_report(results_path, results)
    print(f"Test mAP: {test['map']:.4f}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
