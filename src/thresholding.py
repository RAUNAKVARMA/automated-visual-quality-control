"""Thresholding utilities for pass/defect decisions from anomaly scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


LabelStr = Literal["Pass", "Defect"]


@dataclass(frozen=True)
class ThresholdResult:
    """Image-level inspection decision."""

    label: LabelStr
    score: float
    threshold: float
    is_anomaly: bool


def normalized_score_decision(score: float, threshold: float) -> ThresholdResult:
    """Apply a decision on Anomalib-style normalized scores (typically in ``[0, 1]``).

    Anomalib's post-processor marks anomalies when the normalized score is **greater**
    than the adaptive threshold (see ``PostProcessor._apply_threshold``).
    """
    is_anomaly = float(score) > float(threshold)
    label: LabelStr = "Defect" if is_anomaly else "Pass"
    return ThresholdResult(label=label, score=float(score), threshold=float(threshold), is_anomaly=is_anomaly)


def raw_score_decision(score: float, threshold: float) -> ThresholdResult:
    """Decision on raw (unnormalized) anomaly scores using the same ``>`` rule."""
    is_anomaly = float(score) > float(threshold)
    label: LabelStr = "Defect" if is_anomaly else "Pass"
    return ThresholdResult(label=label, score=float(score), threshold=float(threshold), is_anomaly=is_anomaly)


def tensor_to_float_list(t: torch.Tensor | None) -> list[float]:
    """Detach scores to Python floats."""
    if t is None:
        return []
    flat = t.detach().float().cpu().view(-1)
    return [float(x) for x in flat]


def anomaly_map_to_binary_mask(
    anomaly_map: np.ndarray,
    percentile: float = 99.5,
    absolute_min: float | None = None,
) -> np.ndarray:
    """Build a boolean defect mask from a 2D anomaly map.

    Uses a high percentile of the map as a soft cutoff (robust when maps are
    smooth). If ``absolute_min`` is set, pixels must also exceed that value.
    """
    if anomaly_map.ndim != 2:
        msg = f"Expected HxW anomaly map, got shape {anomaly_map.shape}"
        raise ValueError(msg)
    thresh = float(np.percentile(anomaly_map, percentile))
    if absolute_min is not None:
        thresh = max(thresh, float(absolute_min))
    # Use >= so pixels equal to the percentile cutoff (e.g. flat plateaus) count as anomalous.
    return anomaly_map >= thresh
