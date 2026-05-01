import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch import nn

from utils.config import TrainConfig
from utils.metrics import compute_classification_metrics
from utils.runtime_utils import extract_session_id

def save_checkpoint(
    path: str,
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
    test_split_name: str | None,
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'img_size': config.img_size,
            'class_to_idx': {'0': 0, '1': 1},
            'normalization': config.normalization_metadata(),
            'model_class': model.__class__.__name__,
            'pretrained_backbone': config.pretrained_backbone,
            'freeze_backbone': config.freeze_backbone,
            'dataset_source': config.dataset_source,
            'train_input_source': config.train_input_source,
            'validation_split': config.validation_split,
            'test_split': test_split_name,
            'spectrogram_config': config.spectrogram_config.to_metadata(),
            'checkpoint_selection_metric': 'val_loss',
        },
        path,
    )

def plot_training_curves(history: dict[str, list[float]], config: TrainConfig) -> None:
    if not history['loss']:
        print('Training curves skipped: no training history available.')
        return

    fig, (ax_loss, ax_acc, ax_f1) = plt.subplots(1, 3, figsize=(18, 4))

    ax_loss.plot(history['loss'], label='train')
    ax_loss.plot(history['val_loss'], label='val')
    ax_loss.set_title('Model loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right')

    ax_acc.plot(history['accuracy'], label='train')
    ax_acc.plot(history['val_accuracy'], label='val')
    ax_acc.set_title('Model accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')

    ax_f1.plot(history['f1'], label='train')
    ax_f1.plot(history['val_f1'], label='val')
    ax_f1.set_title('Model F1 score')
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('F1')
    ax_f1.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(config.figs_dir, 'metrics_training.png'))
    plt.close()

def maybe_build_roc_curve(
    split_name: str,
    metrics: dict[str, np.ndarray | float],
) -> tuple[str, np.ndarray, np.ndarray, float] | None:
    labels = metrics['labels']
    scores = metrics['scores']
    if np.unique(labels).size < 2:
        print(f'ROC skipped for {split_name}: split contains a single class.')
        return None

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return split_name, fpr, tpr, roc_auc

def plot_roc_curves(
    curves: list[tuple[str, np.ndarray, np.ndarray, float]],
    config: TrainConfig,
) -> None:
    if not curves:
        print('No ROC curve generated because no evaluation split contained both classes.')
        return

    plt.figure(figsize=(8, 6))
    for split_name, fpr, tpr, roc_auc in curves:
        plt.plot(fpr, tpr, lw=2, label=f'{split_name.capitalize()} ROC (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - validation and test')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(config.figs_dir, 'roc_validation_test.png'))
    plt.close()

def save_confusion_matrix_artifacts(
    split_name: str,
    metrics: dict[str, np.ndarray | float],
    config: TrainConfig,
) -> tuple[str, str, dict[str, int]]:
    labels = np.asarray(metrics['labels'], dtype=np.int64)
    preds = np.asarray(metrics['predictions'], dtype=np.int64)
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    counts = {
        'tn': int(matrix[0, 0]),
        'fp': int(matrix[0, 1]),
        'fn': int(matrix[1, 0]),
        'tp': int(matrix[1, 1]),
    }

    csv_path = os.path.join(config.reports_dir, f'{split_name}_confusion_matrix.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['actual\\predicted', 'noise', 'whistle'])
        writer.writerow(['noise', counts['tn'], counts['fp']])
        writer.writerow(['whistle', counts['fn'], counts['tp']])

    fig_path = os.path.join(config.figs_dir, f'{split_name}_confusion_matrix.png')
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['noise', 'whistle'],
        yticklabels=['noise', 'whistle'],
        ylabel='Actual',
        xlabel='Predicted',
        title=f'{split_name.capitalize()} confusion matrix',
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                f'{value}',
                ha='center',
                va='center',
                color='white' if value > threshold else 'black',
            )
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return csv_path, fig_path, counts

def write_run_summary_json(
    config: TrainConfig,
    best_epoch: int,
    best_val_loss: float,
    best_model_path: str,
    split_summary: dict[str, dict[str, int]],
    validation_metrics: dict[str, np.ndarray | float],
    test_metrics: dict[str, np.ndarray | float] | None,
    confusion_artifacts: dict[str, dict[str, object]],
    validation_session_report_path: str | None,
    test_session_report_path: str | None,
) -> str:
    payload: dict[str, object] = {
        'dataset_source': config.dataset_source,
        'train_input_source': config.train_input_source,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_model_path': best_model_path,
        'split_summary': split_summary,
        'metrics': {
            config.validation_split: {
                'loss': float(validation_metrics['loss']),
                'accuracy': float(validation_metrics['accuracy']),
                'f1': float(validation_metrics['f1']),
                'precision': float(validation_metrics['precision']),
                'recall': float(validation_metrics['recall']),
                'positive_prediction_rate': float(
                    validation_metrics['positive_prediction_rate']
                ),
            },
        },
        'artifacts': {
            'model': best_model_path,
            'figures_dir': config.figs_dir,
            'reports_dir': config.reports_dir,
            'validation_session_report_path': validation_session_report_path,
            'test_session_report_path': test_session_report_path,
            'confusion_matrices': confusion_artifacts,
        },
    }
    if test_metrics is not None:
        payload['metrics'][config.test_split] = {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'f1': float(test_metrics['f1']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'positive_prediction_rate': float(test_metrics['positive_prediction_rate']),
        }

    summary_path = os.path.join(config.reports_dir, 'run_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    return summary_path

def write_test_only_summary_json(
    config: TrainConfig,
    checkpoint_path: str,
    split_summary: dict[str, dict[str, int]],
    test_metrics: dict[str, np.ndarray | float],
    confusion_artifacts: dict[str, dict[str, object]],
    test_session_report_path: str | None,
) -> str:
    payload: dict[str, object] = {
        'dataset_source': config.dataset_source,
        'train_input_source': config.train_input_source,
        'checkpoint_path': checkpoint_path,
        'split_summary': split_summary,
        'metrics': {
            config.test_split: {
                'loss': float(test_metrics['loss']),
                'accuracy': float(test_metrics['accuracy']),
                'f1': float(test_metrics['f1']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'positive_prediction_rate': float(test_metrics['positive_prediction_rate']),
            },
        },
        'artifacts': {
            'model': checkpoint_path,
            'figures_dir': config.figs_dir,
            'reports_dir': config.reports_dir,
            'test_session_report_path': test_session_report_path,
            'confusion_matrices': confusion_artifacts,
        },
    }
    summary_path = os.path.join(config.reports_dir, 'test_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    return summary_path

def build_session_report_rows(
    split_name: str,
    split_dataset: HFDataset,
    metrics: dict[str, np.ndarray | float],
) -> list[dict[str, object]]:
    if 'recording' not in split_dataset.column_names:
        return []

    recordings = split_dataset['recording']
    labels = np.asarray(metrics['labels'], dtype=np.int64)
    preds = np.asarray(metrics['predictions'], dtype=np.int64)
    scores = np.asarray(metrics['scores'], dtype=np.float64)

    if not (len(recordings) == len(labels) == len(preds) == len(scores)):
        raise RuntimeError(
            f'Per-session report mismatch for {split_name}: '
            f'{len(recordings)=} {len(labels)=} {len(preds)=} {len(scores)=}'
        )

    grouped: dict[str, dict[str, object]] = {}
    for recording, label, pred, score in zip(recordings, labels, preds, scores):
        session_id = extract_session_id(recording)
        stats = grouped.setdefault(
            session_id,
            {
                'split': split_name,
                'session_id': session_id,
                'rows': 0,
                'positives': 0,
                'negatives': 0,
                'predicted_positives': 0,
                'predicted_negatives': 0,
                '_labels': [],
                '_preds': [],
                '_scores': [],
            },
        )
        stats['rows'] += 1
        stats['positives'] += int(label)
        stats['negatives'] += int(1 - label)
        stats['predicted_positives'] += int(pred)
        stats['predicted_negatives'] += int(1 - pred)
        stats['_labels'].append(int(label))
        stats['_preds'].append(int(pred))
        stats['_scores'].append(float(score))

    rows = []
    for session_id, stats in grouped.items():
        session_labels = np.asarray(stats.pop('_labels'), dtype=np.int64)
        session_preds = np.asarray(stats.pop('_preds'), dtype=np.int64)
        session_scores = np.asarray(stats.pop('_scores'), dtype=np.float64)
        session_metrics = compute_classification_metrics(session_labels, session_preds)
        rows.append(
            {
                **stats,
                **session_metrics,
                'mean_positive_score': float(np.mean(session_scores)),
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row['rows']),
            str(row['session_id']),
        )
    )
    return rows

def write_session_report_csv(
    split_name: str,
    rows: list[dict[str, object]],
    config: TrainConfig,
) -> str | None:
    if not rows:
        return None

    path = os.path.join(config.reports_dir, f'{split_name}_session_metrics.csv')
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

def print_session_report_preview(
    split_name: str,
    rows: list[dict[str, object]],
    path: str | None,
    limit: int = 5,
) -> None:
    if not rows:
        return

    if path is not None:
        print(f'  {split_name.capitalize()} session report: {path}')
    preview = rows[:limit]
    for row in preview:
        print(
            f"    {row['session_id']}  rows={row['rows']}  "
            f"acc={row['accuracy']:.4f}  f1={row['f1']:.4f}  "
            f"prec={row['precision']:.4f}  rec={row['recall']:.4f}  "
            f"ppr={row['positive_prediction_rate']:.4f}"
        )
