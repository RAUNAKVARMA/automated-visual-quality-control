"""Evaluate a trained checkpoint on the official MVTec AD test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.utils as su  # noqa: E402

su.ensure_src_on_path()

from anomalib.engine import Engine  # noqa: E402
from src.config import ProjectPaths, load_yaml  # noqa: E402
from src.data_module import assert_category_structure, build_mvtec_datamodule  # noqa: E402
from src.inference import (  # noqa: E402
    collect_test_predictions_for_metrics,
    load_model_from_checkpoint,
    move_image_batch_to_device,
    predict_images,
)
from src.metrics import compute_image_level_metrics, metrics_to_serializable, subsampled_pixel_auroc  # noqa: E402
from src.visualization import (  # noqa: E402
    anomaly_map_to_color_heatmap,
    overlay_heatmap_on_image,
    resize_map_to_image,
    save_image_png,
    save_confusion_matrix_png,
    save_roc_curve_png,
)


def _maybe_pixel_auroc(model, loader, device: torch.device, max_batches: int = 50) -> float | None:
    """Compute pixel AUROC on a subset of batches when masks exist."""
    model.eval()
    y_list: list[np.ndarray] = []
    s_list: list[np.ndarray] = []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if b_idx >= max_batches:
                break
            if batch.gt_mask is None:
                return None
            batch = move_image_batch_to_device(batch, device)
            if model.pre_processor and model.pre_processor.transform:
                batch.image, batch.gt_mask = model.pre_processor.transform(batch.image, batch.gt_mask)
            out = model.test_step(batch, 0)
            if out.anomaly_map is None or out.gt_mask is None:
                return None
            amap = out.anomaly_map.detach().float().cpu().numpy()
            mask = out.gt_mask.detach().float().cpu().numpy()
            for i in range(amap.shape[0]):
                y_list.append((mask[i, 0] > 0.5).astype(np.uint8).ravel())
                s_list.append(amap[i, 0].ravel())
    if not y_list:
        return None
    y = np.concatenate(y_list)
    s = np.concatenate(s_list)
    return subsampled_pixel_auroc(y, s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on MVTec AD test set.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint path (defaults to newest under model dir).")
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.category:
        cfg["data"]["category"] = args.category
    paths = ProjectPaths.from_config(cfg)
    assert_category_structure(paths.mvtec_root, cfg["data"]["category"])

    trainer_cfg = cfg.get("trainer", {})
    model_root = Path(trainer_cfg.get("default_root_dir", "outputs/models"))
    model_root = model_root / cfg["data"]["category"]
    ckpt = args.ckpt or su.find_latest_checkpoint(model_root)
    if ckpt is None:
        msg = "No checkpoint found. Pass --ckpt explicitly."
        raise SystemExit(msg)

    dm = build_mvtec_datamodule(
        root=paths.mvtec_root,
        category=cfg["data"]["category"],
        train_batch_size=int(cfg["data"]["eval_batch_size"]),
        eval_batch_size=int(cfg["data"]["eval_batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        seed=cfg.get("project", {}).get("seed"),
    )
    model = load_model_from_checkpoint(cfg, ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    engine = Engine(
        accelerator="auto",
        devices=1,
        default_root_dir=str(paths.reports_dir / "engine_eval"),
        logger=False,
        enable_checkpointing=False,
    )
    anomalib_metrics = engine.test(model=model, datamodule=dm, ckpt_path=str(ckpt), verbose=False)

    dm.setup("test")
    loader = dm.test_dataloader()
    y_true, y_score, y_pred = collect_test_predictions_for_metrics(model, loader, device)
    img_metrics = compute_image_level_metrics(y_true, y_score, y_pred)
    pixel_auroc = _maybe_pixel_auroc(model, loader, device)

    tag = f"{cfg['model']['name']}_{cfg['data']['category']}"
    fig_dir = paths.figures_dir / tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    cm = np.array(img_metrics.confusion_matrix)
    save_confusion_matrix_png(cm, ("good (0)", "defect (1)"), fig_dir / "confusion_matrix.png")
    save_roc_curve_png(img_metrics.fpr, img_metrics.tpr, img_metrics.roc_auc, fig_dir / "roc_curve.png")

    # Sample qualitative heatmaps from the test split
    test_root = paths.mvtec_root / cfg["data"]["category"] / "test"
    sample_paths = sorted(test_root.rglob("*.png"))[:8]
    for sp in sample_paths:
        try:
            pr = predict_images(cfg, ckpt, sp, batch_size=1)[0]
        except (OSError, ValueError, RuntimeError):
            continue
        raw = np.array(Image.open(sp).convert("RGB"))
        h, w = raw.shape[:2]
        amap = resize_map_to_image(pr.anomaly_map, h, w)
        heat = anomaly_map_to_color_heatmap(amap)
        overlay = overlay_heatmap_on_image(raw, heat)
        save_image_png(overlay, fig_dir / f"sample_{sp.parent.name}_{sp.stem}_overlay.png")

    flat_metrics = metrics_to_serializable(img_metrics)
    flat_metrics["pixel_auroc_sampled"] = pixel_auroc
    def _scalarize(d: object) -> object:
        if isinstance(d, dict):
            return {k: _scalarize(v) for k, v in d.items()}
        if hasattr(d, "item"):
            return float(d.item())  # type: ignore[no-any-return]
        return d

    flat_metrics["anomalib_test"] = _scalarize(anomalib_metrics[0]) if anomalib_metrics else {}

    su.dump_json(flat_metrics, paths.reports_dir / f"metrics_{tag}.json")
    flat_csv = {k: v for k, v in flat_metrics.items() if k != "anomalib_test"}
    pd.DataFrame([flat_csv]).to_csv(paths.reports_dir / f"metrics_{tag}.csv", index=False)
    print(f"Saved metrics to {paths.reports_dir / f'metrics_{tag}.json'}")
    print(f"Saved figures under {fig_dir}")


if __name__ == "__main__":
    main()
