"""Image-level and optional pixel-level metrics (scikit-learn + NumPy)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ImageLevelMetrics:
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    fpr: np.ndarray
    tpr: np.ndarray


def compute_image_level_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> ImageLevelMetrics:
    """Compute standard binary metrics (label: 0=good, 1=defect)."""
    y_true = y_true.astype(np.int32).ravel()
    y_pred = y_pred.astype(np.int32).ravel()
    y_score = y_score.astype(np.float64).ravel()
    roc = float(roc_auc_score(y_true, y_score))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return ImageLevelMetrics(
        roc_auc=roc,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm.tolist(),
        fpr=fpr,
        tpr=tpr,
    )


def subsampled_pixel_auroc(
    y_true_mask_flat: np.ndarray,
    y_score_flat: np.ndarray,
    max_samples: int = 200_000,
    seed: int = 0,
) -> float | None:
    """ROC-AUC over flattened pixels; subsample for speed/memory."""
    y_true_mask_flat = y_true_mask_flat.astype(np.uint8).ravel()
    y_score_flat = y_score_flat.astype(np.float64).ravel()
    if y_true_mask_flat.size == 0:
        return None
    # Need both classes for AUROC
    if y_true_mask_flat.min() == y_true_mask_flat.max():
        return None
    n = y_true_mask_flat.size
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        y_true_mask_flat = y_true_mask_flat[idx]
        y_score_flat = y_score_flat[idx]
    return float(roc_auc_score(y_true_mask_flat, y_score_flat))


def metrics_to_serializable(m: ImageLevelMetrics) -> dict[str, Any]:
    """JSON-friendly dict (drops large arrays; keep ROC AUC numeric)."""
    return {
        "roc_auc": m.roc_auc,
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "confusion_matrix": m.confusion_matrix,
    }
