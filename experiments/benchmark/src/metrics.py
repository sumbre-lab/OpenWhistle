"""Test metrics and clip-bootstrap sampling shared by probing and fine-tuning.

Detection preserves the released rank-based AP implementation adapted from
https://github.com/amdegroot/tnt: descending Torch sort, float32 precision at
positive ranks, and zero AP for labels without positives.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score


class AveragePrecision:
    """Accumulate multilabel scores and compute AP independently per label."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = torch.empty(0, dtype=torch.float32)
        self.targets = torch.empty(0, dtype=torch.int64)

    def update(self, output, target):
        scores = torch.as_tensor(output).detach().to(dtype=torch.float32, device="cpu")
        targets = torch.as_tensor(target).detach().to(device="cpu")
        if scores.ndim == 1:
            scores = scores[:, None]
        if targets.ndim == 1:
            targets = targets[:, None]
        if scores.ndim != 2 or scores.shape != targets.shape:
            raise ValueError(
                "Scores and targets must have matching (clips, labels) shapes."
            )
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("Detection targets must be binary.")
        targets = targets.to(dtype=torch.int64)
        if not torch.isfinite(scores).all():
            raise ValueError("Detection scores must be finite.")
        if self.scores.numel() == 0:
            self.scores = scores.clone()
            self.targets = targets.clone()
        else:
            self.scores = torch.cat((self.scores, scores), dim=0)
            self.targets = torch.cat((self.targets, targets), dim=0)

    def get_metric(self):
        if self.scores.numel() == 0:
            raise ValueError("No detection predictions have been added.")
        ap = torch.zeros(self.scores.size(1))
        ranks = torch.arange(1, self.scores.size(0) + 1).float()
        for label in range(self.scores.size(1)):
            _, order = torch.sort(self.scores[:, label], 0, True)
            truth = self.targets[order, label]
            precision = truth.float().cumsum(0).div(ranks)
            ap[label] = precision[truth.bool()].sum() / max(truth.sum(), 1)
        return ap


class MeanAveragePrecision:
    """Macro-average AP over all declared detection labels."""

    def __init__(self):
        self.ap = AveragePrecision()

    def reset(self):
        self.ap.reset()

    def update(self, output, target):
        self.ap.update(output, target)

    def get_metric(self):
        return {"map": self.ap.get_metric().mean().item()}

    def get_primary_metric(self):
        return self.get_metric()["map"]


def score_predictions(dataset_name: str, y_true, y_pred_or_score) -> float:
    if dataset_name == "detection":
        metric = MeanAveragePrecision()
        metric.update(y_pred_or_score, y_true)
        return float(metric.get_primary_metric())
    return float(
        f1_score(
            y_true,
            y_pred_or_score,
            average="macro",
            zero_division=0,
        )
    )


def bootstrap_metric_scores(
    dataset_name: str,
    y_true,
    y_pred_or_score,
    num_bootstrap: int,
    seed: int,
) -> list[float]:
    if num_bootstrap <= 0:
        return []

    rng = np.random.default_rng(seed)
    n_examples = len(y_true)
    bootstrap_scores = []
    for _ in range(num_bootstrap):
        indices = rng.integers(0, n_examples, size=n_examples)
        bootstrap_scores.append(
            score_predictions(
                dataset_name,
                y_true[indices],
                y_pred_or_score[indices],
            )
        )
    return bootstrap_scores
