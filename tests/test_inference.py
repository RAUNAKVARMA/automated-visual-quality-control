"""Unit tests for thresholding and visualization helpers (no GPU)."""

from __future__ import annotations

import numpy as np

from src.thresholding import anomaly_map_to_binary_mask, normalized_score_decision
from src.visualization import overlay_heatmap_on_image, resize_map_to_image


def test_normalized_score_decision() -> None:
    r = normalized_score_decision(0.8, 0.5)
    assert r.label == "Defect"
    assert r.is_anomaly is True
    r2 = normalized_score_decision(0.2, 0.5)
    assert r2.label == "Pass"
    assert r2.is_anomaly is False


def test_anomaly_map_mask_and_overlay() -> None:
    amap = np.zeros((32, 32), dtype=np.float32)
    amap[10:20, 10:20] = 10.0  # ensure percentile threshold stays below peak
    mask = anomaly_map_to_binary_mask(amap)
    assert mask.dtype == bool
    assert bool(mask[15, 15])
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    heat = np.zeros((32, 32, 3), dtype=np.uint8)
    heat[..., 0] = 255  # blue channel in BGR
    out = overlay_heatmap_on_image(rgb, heat)
    assert out.shape == (32, 32, 3)


def test_resize_map_to_image() -> None:
    m = np.ones((16, 16), dtype=np.float32)
    r = resize_map_to_image(m, 32, 64)
    assert r.shape == (32, 64)
