"""Heatmaps, overlays, and figure export."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert float [0,1] or uint8 image to uint8 RGB."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def anomaly_map_to_color_heatmap(anomaly_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a 2D anomaly map to a BGR heatmap (uint8)."""
    if anomaly_map.ndim != 2:
        msg = f"Expected 2D map, got {anomaly_map.shape}"
        raise ValueError(msg)
    norm = anomaly_map.astype(np.float32)
    norm -= float(norm.min())
    maxv = float(norm.max()) + 1e-8
    norm = (norm / maxv * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(norm, colormap)
    return heat_bgr


def overlay_heatmap_on_image(
    image_rgb_uint8: np.ndarray,
    heatmap_bgr_uint8: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend a CV2 colormap (BGR) on top of an RGB image."""
    if image_rgb_uint8.shape[:2] != heatmap_bgr_uint8.shape[:2]:
        heatmap_bgr_uint8 = cv2.resize(
            heatmap_bgr_uint8,
            (image_rgb_uint8.shape[1], image_rgb_uint8.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    base = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR).astype(np.float32)
    heat = heatmap_bgr_uint8.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * heat
    blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended_u8, cv2.COLOR_BGR2RGB)


def resize_map_to_image(anomaly_map: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize anomaly map to match an image height/width."""
    return cv2.resize(anomaly_map, (width, height), interpolation=cv2.INTER_LINEAR)


def save_confusion_matrix_png(
    cm: np.ndarray,
    labels: Tuple[str, str],
    out_path: Path,
    title: str = "Confusion matrix",
) -> None:
    """Save a confusion matrix figure as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig: Figure = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve_png(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, out_path: Path) -> None:
    """Save ROC curve PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig: Figure = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_image_png(rgb_uint8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_mask_png(mask_bool: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(path), m)
