import random
import re

import numpy as np
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(cpu_only: bool = False) -> torch.device:
    if cpu_only:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def extract_session_id(recording_name: str) -> str:
    """Strip _channel_N so simultaneous multi-channel recordings share one ID."""
    return re.sub(r'_channel_\d+$', '', str(recording_name).strip())

def configure_torch_matmul() -> None:
    try:
        torch.set_float32_matmul_precision('high')
    except (AttributeError, RuntimeError):
        pass