#!/usr/bin/env python3
"""mAP-only entry point for supervised OpenWhistle detection fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SRC = SCRIPT_DIR.parents[1] / "benchmark" / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BENCHMARK_SRC))

import finetune_detection as detection  # noqa: E402
from finetune_classification import (  # noqa: E402
    autocast_context,
    move_batch,
)
from hf_datasets import get_detection_label_vector  # noqa: E402
from metrics import MeanAveragePrecision  # noqa: E402
from models import load_waveform  # noqa: E402


class DetectionDataset(detection.DetectionDataset):
    """Return Python label lists so the shared collator builds tensors directly."""

    def __getitem__(self, index):
        row = self.split[index]
        waveform, sample_rate = load_waveform(row["audio"])
        return waveform, sample_rate, get_detection_label_vector(row)


@torch.no_grad()
def evaluate(model, loader, device, precision) -> dict:
    """Evaluate only the loss and benchmark primary metric (mAP)."""
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
    }


def main() -> None:
    detection.DetectionDataset = DetectionDataset
    detection.evaluate = evaluate
    detection.main()


if __name__ == "__main__":
    main()
