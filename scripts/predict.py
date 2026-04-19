"""Run inference on a single image and save heatmap, overlay, mask, and JSON metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.utils as su  # noqa: E402

su.ensure_src_on_path()

from src.config import ProjectPaths, load_yaml  # noqa: E402
from src.inference import predict_images  # noqa: E402
from src.thresholding import normalized_score_decision  # noqa: E402
from src.visualization import (  # noqa: E402
    anomaly_map_to_color_heatmap,
    overlay_heatmap_on_image,
    resize_map_to_image,
    save_image_png,
    save_mask_png,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on one image.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Normalized score threshold for Pass/Defect.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = ProjectPaths.from_config(cfg)
    out_dir = args.out_dir or (paths.predictions_dir / "single")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = predict_images(cfg, args.ckpt, args.image, batch_size=1)
    if not preds:
        raise SystemExit("No predictions returned.")
    p = preds[0]
    stem = Path(p.image_path).stem

    raw_rgb = np.array(Image.open(p.image_path).convert("RGB"))
    h, w = raw_rgb.shape[:2]
    amap = resize_map_to_image(p.anomaly_map, h, w)
    heat_bgr = anomaly_map_to_color_heatmap(amap)
    overlay = overlay_heatmap_on_image(raw_rgb, heat_bgr)
    mask = p.pred_mask if p.pred_mask is not None else np.zeros((h, w), dtype=bool)
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    decision = normalized_score_decision(p.pred_score, args.threshold)
    meta = {
        "image_path": p.image_path,
        "pred_score": p.pred_score,
        "threshold": args.threshold,
        "label": decision.label,
        "anomalib_label": "Defect" if p.pred_label_anomalib else "Pass",
        "model": p.model_name,
        "backbone": p.backbone,
        "inference_time_s": p.inference_time_s,
    }
    su.dump_json(meta, out_dir / f"{stem}_result.json")
    save_image_png(raw_rgb, out_dir / f"{stem}_input.png")
    save_image_png(overlay, out_dir / f"{stem}_overlay.png")
    save_mask_png(mask, out_dir / f"{stem}_mask.png")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
