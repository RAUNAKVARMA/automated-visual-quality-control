"""Batch inference over a folder; writes CSV + per-image artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.utils as su  # noqa: E402

su.ensure_src_on_path()

from src.config import ProjectPaths, load_yaml  # noqa: E402
from src.inference import predict_images  # noqa: E402
from src.thresholding import normalized_score_decision  # noqa: E402
from src.visualization import anomaly_map_to_color_heatmap, overlay_heatmap_on_image, resize_map_to_image, save_mask_png  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict on a folder of images.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-images", action="store_true", help="Save overlay PNGs for each file.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = ProjectPaths.from_config(cfg)
    out_dir = args.out_dir or (paths.predictions_dir / "batch")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = predict_images(cfg, args.ckpt, args.input_dir, batch_size=args.batch_size)
    rows = []
    for p in preds:
        raw_rgb = Image.open(p.image_path).convert("RGB")
        arr = np.asarray(raw_rgb)
        h, w = arr.shape[:2]
        amap = resize_map_to_image(p.anomaly_map, h, w)
        heat_bgr = anomaly_map_to_color_heatmap(amap)
        overlay = overlay_heatmap_on_image(arr, heat_bgr)
        decision = normalized_score_decision(p.pred_score, args.threshold)
        stem = Path(p.image_path).stem
        if args.save_images:
            from src.visualization import save_image_png

            save_image_png(overlay, out_dir / f"{stem}_overlay.png")
        mask = p.pred_mask if p.pred_mask is not None else None
        if mask is not None:
            save_mask_png(mask, out_dir / f"{stem}_mask.png")
        rows.append(
            {
                "path": p.image_path,
                "pred_score": p.pred_score,
                "label_custom": decision.label,
                "label_anomalib": "Defect" if p.pred_label_anomalib else "Pass",
                "threshold": args.threshold,
                "inference_time_s": p.inference_time_s,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "batch_report.csv"
    df.to_csv(csv_path, index=False)
    su.dump_json({"count": len(rows), "csv": str(csv_path)}, out_dir / "batch_summary.json")
    print(f"Wrote {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
