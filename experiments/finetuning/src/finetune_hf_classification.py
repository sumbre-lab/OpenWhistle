#!/usr/bin/env python3
"""End-to-end fine-tuning for one Hugging Face audio backbone."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    get_linear_schedule_with_warmup,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SRC = SCRIPT_DIR.parents[1] / "benchmark" / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BENCHMARK_SRC))

from finetune_classification import (  # noqa: E402
    ClassificationDataset,
    evaluate,
    prepare_waveform,
    public_metrics,
    set_seed,
    save_test_predictions,
    train_epoch,
)
from hf_datasets import (  # noqa: E402
    CLASSIFICATION_BALANCED_CONFIG_NAME,
    load_classification_splits,
)
from models import make_btb3_views  # noqa: E402
from benchmark_revisions import MODEL_REVISIONS  # noqa: E402
from metrics import bootstrap_metric_scores  # noqa: E402
from checkpoint_io import save_checkpoint  # noqa: E402
from report_io import write_report  # noqa: E402
from refit_protocol import refit_model, validate_refit_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument(
        "--feature_mode",
        choices=("standard", "btb3_concat", "btb3_mean"),
        default="standard",
    )
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument(
        "--dataset_config",
        choices=("balanced", "all"),
        default=CLASSIFICATION_BALANCED_CONFIG_NAME,
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Validation/test batch size; defaults to training batch size.")
    parser.add_argument("--eval_precision", choices=("none", "float16", "bfloat16"),
                        default=None, help="Validation/test autocast; defaults to training precision.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--refit_train_validation", action="store_true",
                        help="Select an epoch on validation, then restart on train+validation for that many epochs.")
    parser.add_argument("--num_bootstrap", type=int, default=0,
                        help="Test bootstrap resamples; 0 disables, otherwise require at least 2.")
    parser.add_argument("--max_duration_seconds", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--mixed_precision",
        choices=("none", "float16", "bfloat16"),
        default="float16",
    )
    return parser.parse_args()


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


def worker_init_fn(_worker_id: int) -> None:
    torch.set_num_threads(1)


def make_training_components(model, train_loader, args, device):
    optimizer = torch.optim.AdamW(
        [{"params": list(model.backbone_parameters()), "lr": args.learning_rate},
         {"params": list(model.head_parameters()), "lr": args.head_learning_rate}],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    # Preserve the schedule as a function of epoch across selection and refit.
    # Selection determines when to stop, rather than changing the LR schedule.
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.mixed_precision == "float16")
    return optimizer, scheduler, scaler


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be at least 1.")
    if args.batch_size < 1 or (args.eval_batch_size is not None and args.eval_batch_size < 1):
        raise ValueError("Batch sizes must be positive.")
    validate_refit_output(args)
    if args.num_bootstrap != 0 and args.num_bootstrap < 2:
        raise ValueError("Use --num_bootstrap 0 or at least 2.")
    eval_precision = args.eval_precision or args.mixed_precision
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, label_feature = load_classification_splits(args.dataset_config)
    label_names = list(label_feature.names)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_id, revision=MODEL_REVISIONS.get(args.model_id))
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
    if args.refit_train_validation:
        set_seed(args.seed)
    model = HFAudioClassifier(
        args.model_id,
        args.feature_mode,
        len(label_names),
        args.dropout,
    ).to(device)

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
            ClassificationDataset(splits[split_name]),
            shuffle=split_name == "train",
            batch_size=(args.batch_size if split_name == "train"
                        else args.eval_batch_size or args.batch_size),
            **loader_kwargs,
        )
        for split_name in ("train", "validation", "test")
    }

    backbone_parameters = list(model.backbone_parameters())
    head_parameters = list(model.head_parameters())
    optimizer, scheduler, scaler = make_training_components(model, loaders["train"], args, device)

    trainable_backbone = sum(
        parameter.numel()
        for parameter in backbone_parameters
        if parameter.requires_grad
    )
    trainable_head = sum(parameter.numel() for parameter in head_parameters)
    if trainable_backbone == 0:
        raise RuntimeError("The audio backbone is frozen; end-to-end fine-tuning aborted.")
    print(
        f"device={device} model={args.model_id} sample_rate={sample_rate} "
        f"feature_mode={args.feature_mode} labels={label_names}\n"
        f"trainable audio backbone parameters={trainable_backbone:,}; "
        f"classification head parameters={trainable_head:,}"
    )

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
            eval_precision,
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
            save_checkpoint(
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

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    best_epoch = checkpoint["epoch"]
    final_training = dict(split="train", epochs=best_epoch, checkpoint="best.pt")
    refit_history = []
    if args.refit_train_validation:
        # Release selection weights and optimizer state before creating a fresh model.
        del model, optimizer, scheduler, scaler, backbone_parameters, head_parameters, checkpoint
        model, final_training, refit_history = refit_model(
            args, best_epoch,
            lambda: HFAudioClassifier(args.model_id, args.feature_mode, len(label_names), args.dropout),
            loaders, loader_kwargs, train_epoch, device, set_seed, label_names,
            "macro_f1", best_validation_f1)
    else:
        model.load_checkpoint_state(checkpoint["model"])
    test = evaluate(model, loaders["test"], device, eval_precision)
    save_test_predictions(args.output_dir, test)
    results = {
        "protocol": "supervised_end_to_end_finetuning",
        "backbone": "hf",
        "model_id": args.model_id,
        "model_revision": model.model_revision,
        "training_args": json.loads(json.dumps(vars(args), default=str)),
        "feature_mode": args.feature_mode,
        "dataset_config": args.dataset_config,
        "seed": args.seed,
        "sample_rate": sample_rate,
        "label_names": label_names,
        "trainable_backbone_parameters": trainable_backbone,
        "trainable_head_parameters": trainable_head,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_validation_f1,
        "validation_score_scope": "Selection model trained on train only.",
        "final_training": final_training,
        "test": public_metrics(test),
        "test_confusion_matrix": confusion_matrix(
            test["labels"],
            test["predictions"],
            labels=list(range(len(label_names))),
        ).tolist(),
        "history": history,
        "refit_history": refit_history,
    }
    if args.num_bootstrap:
        bootstrap_scores = bootstrap_metric_scores(
            "classification", np.asarray(test["labels"]), np.asarray(test["predictions"]),
            args.num_bootstrap, args.seed)
        np.savez_compressed(args.output_dir / "bootstrap_scores.npz", scores=bootstrap_scores)
        results["test_bootstrap"] = dict(
            num_resamples=args.num_bootstrap, seed=args.seed, ddof=1,
            std=float(np.std(bootstrap_scores, ddof=1)), metric="macro_f1")
    results_path = args.output_dir / "results.json"
    write_report(results_path, results)
    print(f"Test macro-F1: {test['macro_f1']:.4f}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
