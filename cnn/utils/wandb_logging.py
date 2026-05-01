import os

import torch
import wandb
from torch import nn

from utils.config import TrainConfig
from utils.model_runtime import count_parameters

def init_wandb_run(config: TrainConfig, device: torch.device) -> object | None:
    if not config.wandb_enabled:
        print('Weights & Biases logging disabled.')
        return None
    if wandb is None:
        print('Weights & Biases not installed, skipping wandb logging.')
        return None

    init_kwargs = {
        'project': config.wandb_project,
        'config': config.wandb_config(device),
        'mode': config.wandb_mode,
        'tags': ['torch', 'cnn', 'dolphin', 'whistle'],
        'settings': wandb.Settings(console=config.wandb_console),
    }
    if config.wandb_entity:
        init_kwargs['entity'] = config.wandb_entity
    if config.wandb_run_name:
        init_kwargs['name'] = config.wandb_run_name

    try:
        run = wandb.init(**init_kwargs)
        if run is not None:
            run.define_metric('epoch')
            run.define_metric('train/*', step_metric='epoch')
            run.define_metric('validation/*', step_metric='epoch')
            run.define_metric('test/*', step_metric='epoch')
        return run
    except Exception as exc:
        print(f'Warning: unable to initialize wandb ({exc}). Continuing without wandb.')
        return None

def update_wandb_run_config(
    run: object | None,
    model: nn.Module,
    split_summary: dict[str, dict[str, int]],
) -> None:
    if run is None:
        return

    param_counts = count_parameters(model)
    config_update = {
        'parameter_count_total': param_counts['total'],
        'parameter_count_trainable': param_counts['trainable'],
        'split_summary': split_summary,
    }
    try:
        run.config.update(config_update, allow_val_change=True)
    except Exception as exc:
        print(f'Warning: unable to update wandb config ({exc}).')

def wandb_log(run: object | None, payload: dict[str, object]) -> None:
    if run is None or wandb is None:
        return
    try:
        run.log(payload)
    except Exception as exc:
        print(f'Warning: wandb logging failed ({exc}).')

def wandb_log_artifact_images(run: object | None, config: TrainConfig) -> None:
    if run is None or wandb is None:
        return

    payload = {}
    metrics_path = os.path.join(config.figs_dir, 'metrics_training.png')
    roc_path = os.path.join(config.figs_dir, 'roc_validation_test.png')
    if os.path.exists(metrics_path):
        payload['figures/metrics_training'] = wandb.Image(metrics_path)
    if os.path.exists(roc_path):
        payload['figures/roc_validation_test'] = wandb.Image(roc_path)
    for split_name in (config.validation_split, config.test_split):
        confusion_path = os.path.join(config.figs_dir, f'{split_name}_confusion_matrix.png')
        if os.path.exists(confusion_path):
            payload[f'figures/{split_name}_confusion_matrix'] = wandb.Image(confusion_path)
    if payload:
        wandb_log(run, payload)

def wandb_log_table(
    run: object | None,
    key: str,
    rows: list[dict[str, object]],
) -> None:
    if run is None or wandb is None or not rows:
        return

    columns = list(rows[0].keys())
    data = [[row[column] for column in columns] for row in rows]
    try:
        run.log({key: wandb.Table(columns=columns, data=data)})
    except Exception as exc:
        print(f'Warning: unable to log wandb table {key} ({exc}).')