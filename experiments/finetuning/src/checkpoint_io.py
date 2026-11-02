"""Preserve the previous checkpoint when serialization or disk writes fail."""
import os
from pathlib import Path
import tempfile

import torch


def save_checkpoint(state, path):
    """Write on the destination filesystem, then replace only after a full save."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
