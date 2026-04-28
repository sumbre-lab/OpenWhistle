import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config import TrainConfig
from utils.model import build_vgg16_whistle_classifier

def build_model(config: TrainConfig, device: torch.device) -> nn.Module:
    return build_vgg16_whistle_classifier(
        pretrained_backbone=config.pretrained_backbone,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

def build_optimizer(model: nn.Module, config: TrainConfig) -> Adam:
    return Adam(model.parameters(), lr=config.learning_rate)

def build_scheduler(optimizer: Adam, config: TrainConfig) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )

def amp_enabled(config: TrainConfig, device: torch.device) -> bool:
    return bool(config.use_amp and device.type == 'cuda')

def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {'total': int(total), 'trainable': int(trainable)}
