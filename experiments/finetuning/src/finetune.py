"""Fine-tune any benchmark encoder on classification or multilabel detection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetuning_config import HF_MODELS, MODELS, MODEL_REVISIONS, TASK_DATASETS, recipe
from finetuning_models import make_model_and_collator, make_refit_model, set_seed
from hf_datasets import get_detection_label_vector, load_classification_splits, load_detection_splits
from metrics import MeanAveragePrecision, bootstrap_metric_scores
from models import load_waveform
from checkpoint_io import save_checkpoint
from refit_protocol import (
    add_refit_argument, make_training_components, refit_model, validate_refit_output,
)
from report_io import write_report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[model[0] for model in MODELS], required=True)
    parser.add_argument("--dataset_name", choices=("classification", "detection"), required=True)
    parser.add_argument("--dataset_config", choices=("unbalanced", "balanced", "default"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    for name in ("epochs", "patience", "batch_size", "eval_batch_size",
                 "gradient_accumulation_steps", "num_workers", "sample_rate"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("learning_rate", "head_learning_rate", "weight_decay", "warmup_ratio",
                 "max_grad_norm", "dropout", "label_smoothing", "max_duration_seconds"):
        parser.add_argument(f"--{name}", type=float)
    for name in ("mixed_precision", "eval_precision"):
        parser.add_argument(f"--{name}", choices=("none", "float16", "bfloat16"))
    parser.add_argument("--feature_mode", choices=("standard", "btb3_concat", "btb3_mean"))
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--num_bootstrap", type=int, default=0,
                        help="Optional per-job test bootstrap; the campaign aggregates it by default.")
    add_refit_argument(parser)
    args = parser.parse_args()
    args.dataset_config = args.dataset_config or ("unbalanced" if args.dataset_name == "classification" else "default")
    if (args.dataset_name, args.dataset_config) not in TASK_DATASETS.values():
        parser.error("Use unbalanced/balanced for classification or default for detection.")
    if args.model not in HF_MODELS and (args.feature_mode is not None or args.sample_rate is not None):
        parser.error("--feature_mode and --sample_rate apply only to Hugging Face audio encoders.")
    task = "balanced" if args.dataset_config == "balanced" else args.dataset_name
    for name, value in recipe(args.model, task).items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    args.feature_mode = args.feature_mode or "standard"
    args.label_smoothing = args.label_smoothing or 0.0
    args.model_id = next(model[3] for model in MODELS if model[0] == args.model)
    return args


class AudioDataset(Dataset):
    def __init__(self, split, dataset_name):
        self.split = split
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        row = self.split[index]
        waveform, sample_rate = load_waveform(row["audio"])
        labels = (get_detection_label_vector(row) if self.dataset_name == "detection"
                  else int(row["label"]))
        return waveform, sample_rate, labels


def worker_init_fn(_worker_id):
    torch.set_num_threads(1)


def move_batch(batch, device):
    labels = batch.pop("labels").to(device, non_blocking=True)
    inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return inputs, labels


def autocast_context(device, precision):
    enabled = device.type == "cuda" and precision != "none"
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def task_loss(logits, labels, dataset_name, label_smoothing=0.0):
    if dataset_name == "detection":
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model, loader, device, precision, *, dataset_name):
    model.eval()
    losses, labels, predictions = [], [], []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        inputs, target = move_batch(batch, device)
        if dataset_name == "detection":
            target = target.float()
        with autocast_context(device, precision):
            logits = model(**inputs)
            loss = task_loss(logits, target, dataset_name)
        losses.append(float(loss))
        if dataset_name == "detection":
            labels.append(target.cpu())
            predictions.append(logits.sigmoid().float().cpu())
        else:
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
    result = {"loss": float(np.mean(losses))}
    if dataset_name == "detection":
        labels, predictions = torch.cat(labels), torch.cat(predictions)
        metric = MeanAveragePrecision()
        metric.update(predictions, labels)
        result["map"] = float(metric.get_primary_metric())
        labels, predictions = labels.tolist(), predictions.tolist()
    else:
        result.update(macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
                      accuracy=float(accuracy_score(labels, predictions)))
    return dict(result, labels=labels, predictions=predictions)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        inputs, labels = move_batch(batch, device)
        if args.dataset_name == "detection":
            labels = labels.float()
        with autocast_context(device, args.mixed_precision):
            logits = model(**inputs)
            loss = task_loss(logits, labels, args.dataset_name, args.label_smoothing)
            scaled_loss = loss / args.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        losses.append(float(loss.detach()))
        if step % args.gradient_accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def public_metrics(metrics):
    return {key: value for key, value in metrics.items() if key not in {"labels", "predictions"}}


def main():
    args = parse_args()
    validate_refit_output(args)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be at least 1.")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if args.num_bootstrap != 0 and args.num_bootstrap < 2:
        raise ValueError("Use --num_bootstrap 0 or at least 2.")
    eval_precision = args.eval_precision or args.mixed_precision
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification = args.dataset_name == "classification"
    metric = "macro_f1" if classification else "map"
    if classification:
        splits, label_feature = load_classification_splits(args.dataset_config)
        label_names = list(label_feature.names)
    else:
        splits, label_names = load_detection_splits()
        label_names = list(label_names)
    model, collator, sample_rate = make_model_and_collator(args, len(label_names), device)
    model.to(device)
    loader_kwargs = dict(collate_fn=collator, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    # Retain the original native-classification worker lifecycle and prefetching elsewhere.
    persistent_workers = args.dataset_name == "detection" or args.model in HF_MODELS
    if args.num_workers > 0 and persistent_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor,
                             worker_init_fn=worker_init_fn)
    loaders = {
        name: DataLoader(AudioDataset(splits[name], args.dataset_name), shuffle=name == "train",
                         batch_size=args.batch_size if name == "train" else args.eval_batch_size,
                         **loader_kwargs)
        for name in ("train", "validation", "test")
    }
    backbone_parameters = list(model.backbone_parameters())
    head_parameters = list(model.head_parameters())
    optimizer, scheduler, scaler = make_training_components(model, loaders["train"], args, device)
    trainable_backbone = sum(p.numel() for p in backbone_parameters if p.requires_grad)
    trainable_head = sum(p.numel() for p in head_parameters)
    if trainable_backbone == 0:
        raise RuntimeError("The audio backbone is frozen; end-to-end fine-tuning aborted.")
    print(f"device={device} model={args.model_id} sample_rate={sample_rate} labels={label_names}\n"
          f"trainable audio backbone parameters={trainable_backbone:,}; head parameters={trainable_head:,}",
          flush=True)
    best_path = args.output_dir / "best.pt"
    history, best_validation, epochs_without_improvement = [], -math.inf, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, scheduler, scaler, device, args)
        validation = evaluate(model, loaders["validation"], device, eval_precision, dataset_name=args.dataset_name)
        epoch_result = dict(epoch=epoch, train_loss=train_loss, validation=public_metrics(validation))
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        if validation[metric] > best_validation:
            best_validation, epochs_without_improvement = validation[metric], 0
            save_checkpoint(dict(model=model.checkpoint_state(), epoch=epoch, label_names=label_names,
                                 args=vars(args), **{f"validation_{metric}": best_validation}), best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.", flush=True)
                break
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    best_epoch = checkpoint["epoch"]
    final_training = dict(split="train", epochs=best_epoch, checkpoint="best.pt")
    refit_history = []
    if args.refit_train_validation:
        del model, optimizer, scheduler, scaler, backbone_parameters, head_parameters, checkpoint
        model, final_training, refit_history = refit_model(
            args, best_epoch, lambda: make_refit_model(args, len(label_names), device),
            loaders, loader_kwargs, train_epoch, device, set_seed, label_names, metric, best_validation)
    else:
        model.load_checkpoint_state(checkpoint["model"])
    test = evaluate(model, loaders["test"], device, eval_precision, dataset_name=args.dataset_name)
    np.savez_compressed(args.output_dir / "test_predictions.npz",
                        y_true=np.asarray(test["labels"]), y_pred=np.asarray(test["predictions"]))
    training_args = vars(args).copy()
    if args.model not in HF_MODELS:
        training_args.pop("feature_mode")
    if classification and args.model not in HF_MODELS:
        training_args.pop("prefetch_factor")
    if not classification:
        training_args.pop("label_smoothing")
    results = dict(
        protocol="supervised_end_to_end_finetuning",
        backbone="hf" if args.model in HF_MODELS else args.model,
        model_id=args.model_id, model_revision=MODEL_REVISIONS.get(args.model_id),
        dataset_config=args.dataset_config, training_args=json.loads(json.dumps(training_args, default=str)),
        seed=args.seed, sample_rate=sample_rate, label_names=label_names,
        trainable_backbone_parameters=trainable_backbone, trainable_head_parameters=trainable_head,
        best_epoch=best_epoch, validation_score_scope="Selection model trained on train only.",
        final_training=final_training, refit_history=refit_history, test=public_metrics(test), history=history,
        **{f"validation_{metric}": best_validation})
    if args.model in HF_MODELS:
        results["feature_mode"] = args.feature_mode
    if classification:
        results["test_confusion_matrix"] = confusion_matrix(
            test["labels"], test["predictions"], labels=list(range(len(label_names)))).tolist()
    else:
        results["task"] = "multilabel_detection"
    if args.num_bootstrap:
        scores = bootstrap_metric_scores(args.dataset_name, np.asarray(test["labels"]),
                                         np.asarray(test["predictions"]), args.num_bootstrap, args.seed)
        np.savez_compressed(args.output_dir / "bootstrap_scores.npz", scores=scores)
        results["test_bootstrap"] = dict(num_resamples=args.num_bootstrap, seed=args.seed, ddof=1,
                                         std=float(np.std(scores, ddof=1)), metric=metric)
    path = args.output_dir / "results.json"
    write_report(path, results)
    print(f"Test {metric}: {test[metric]:.4f}\nResults: {path}", flush=True)


if __name__ == "__main__":
    main()
