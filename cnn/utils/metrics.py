from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.config import TrainConfig
from utils.model_runtime import amp_enabled

def compute_classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if labels.size == 0:
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'positive_prediction_rate': 0.0,
        }

    return {
        'accuracy': float(np.mean(labels == preds)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'positive_prediction_rate': float(np.mean(preds == 1)),
    }

@dataclass(frozen=True)
class EpochMetric:
    name: str
    display_label: str

TRACKED_EPOCH_METRICS = (
    EpochMetric('loss', 'loss'),
    EpochMetric('accuracy', 'acc'),
    EpochMetric('f1', 'f1'),
    EpochMetric('precision', 'prec'),
    EpochMetric('recall', 'rec'),
    EpochMetric('positive_prediction_rate', 'ppr'),
)

def new_history() -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    for metric in TRACKED_EPOCH_METRICS:
        history[metric.name] = []
        history[f'val_{metric.name}'] = []
    return history

def metric_value(metrics: dict[str, np.ndarray | float], metric_name: str) -> float:
    return float(metrics[metric_name])

def append_epoch_history(
    history: dict[str, list[float]],
    train_metrics: dict[str, np.ndarray | float],
    validation_metrics: dict[str, np.ndarray | float],
) -> None:
    for metric in TRACKED_EPOCH_METRICS:
        history[metric.name].append(metric_value(train_metrics, metric.name))
        history[f'val_{metric.name}'].append(
            metric_value(validation_metrics, metric.name)
        )

def format_epoch_metrics(
    metrics: dict[str, np.ndarray | float],
    prefix: str = '',
) -> str:
    parts = [
        f'{prefix}{metric.display_label}={metric_value(metrics, metric.name):.4f}'
        for metric in TRACKED_EPOCH_METRICS
    ]
    return '  '.join(parts)

def build_epoch_wandb_payload(
    epoch: int,
    train_metrics: dict[str, np.ndarray | float],
    validation_metrics: dict[str, np.ndarray | float],
    learning_rate: float,
    best_val_loss: float,
) -> dict[str, object]:
    payload: dict[str, object] = {'epoch': epoch}
    for metric in TRACKED_EPOCH_METRICS:
        payload[f'train/{metric.name}'] = metric_value(train_metrics, metric.name)
        payload[f'validation/{metric.name}'] = metric_value(
            validation_metrics,
            metric.name,
        )

    payload['train/learning_rate'] = float(learning_rate)
    payload['validation/best_loss_so_far'] = min(
        best_val_loss,
        metric_value(validation_metrics, 'loss'),
    )
    return payload

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
    device: torch.device,
    pin_memory: bool,
    description: str,
    optimizer: Adam | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, np.ndarray | float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_scores = []
    all_labels = []
    all_preds = []
    use_autocast = amp_enabled(config, device)

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=description, leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            autocast_manager = (
                torch.amp.autocast(device_type='cuda', enabled=use_autocast)
                if device.type == 'cuda'
                else nullcontext()
            )
            with autocast_manager:
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size
            all_scores.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
            progress.set_postfix(
                loss=f'{total_loss / total_samples:.4f}',
                acc=f'{total_correct / total_samples:.4f}',
            )

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()
    metrics = compute_classification_metrics(labels, preds)
    return {
        'loss': total_loss / total_samples,
        **metrics,
        'scores': scores,
        'labels': labels,
        'predictions': preds,
    }