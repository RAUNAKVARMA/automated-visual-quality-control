"""Training checkpoint loading and batched inference via Anomalib Engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import torch
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from anomalib.pre_processing import PreProcessor
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize

from src.thresholding import anomaly_map_to_binary_mask

ModelName = Literal["Padim", "Patchcore"]


@dataclass
class SinglePrediction:
    """One image prediction record."""

    image_path: str
    pred_score: float
    pred_label_anomalib: bool
    anomaly_map: np.ndarray
    pred_mask: np.ndarray | None
    inference_time_s: float
    backbone: str
    model_name: ModelName


def _image_size_tuple(cfg: dict[str, Any]) -> tuple[int, int]:
    size = cfg["preprocess"]["image_size"]
    if isinstance(size, int):
        return (size, size)
    h, w = int(size[0]), int(size[1])
    return (h, w)


def build_preprocessor(image_size: tuple[int, int]) -> PreProcessor:
    """Match ``AnomalibModule.configure_pre_processor`` defaults."""
    return PreProcessor.configure_pre_processor(image_size)


def build_model(cfg: dict[str, Any]) -> Padim | Patchcore:
    """Construct PaDiM or PatchCore from a config dict."""
    mcfg = cfg["model"]
    name = mcfg["name"]
    image_size = _image_size_tuple(cfg)
    pre_processor = build_preprocessor(image_size)
    if name == "Padim":
        return Padim(
            backbone=mcfg["backbone"],
            layers=list(mcfg["layers"]),
            pre_trained=bool(mcfg.get("pre_trained", True)),
            n_features=mcfg.get("n_features"),
            pre_processor=pre_processor,
        )
    if name == "Patchcore":
        return Patchcore(
            backbone=mcfg["backbone"],
            layers=tuple(mcfg["layers"]),
            pre_trained=bool(mcfg.get("pre_trained", True)),
            coreset_sampling_ratio=float(mcfg.get("coreset_sampling_ratio", 0.1)),
            num_neighbors=int(mcfg.get("num_neighbors", 9)),
            pre_processor=pre_processor,
        )
    msg = f"Unsupported model.name: {name!r}"
    raise ValueError(msg)


def load_model_from_checkpoint(cfg: dict[str, Any], ckpt_path: str | Path) -> Padim | Patchcore:
    """Load a Lightning checkpoint for PaDiM or PatchCore."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        msg = f"Checkpoint not found: {ckpt_path}"
        raise FileNotFoundError(msg)
    name: ModelName = cfg["model"]["name"]
    if name == "Padim":
        return Padim.load_from_checkpoint(str(ckpt_path))
    if name == "Patchcore":
        return Patchcore.load_from_checkpoint(str(ckpt_path))
    msg = f"Unsupported model: {name}"
    raise ValueError(msg)


def _flatten_predict_outputs(outputs: Any) -> list[Any]:
    """Lightning may return nested lists per dataloader/epoch."""
    flat: list[Any] = []
    if outputs is None:
        return flat
    if isinstance(outputs, list):
        for item in outputs:
            flat.extend(_flatten_predict_outputs(item))
        return flat
    return [outputs]


def move_image_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move tensor fields on an Anomalib batch to ``device`` (in place)."""
    for field in fields(batch):
        name = field.name
        value = getattr(batch, name)
        if isinstance(value, torch.Tensor):
            setattr(batch, name, value.to(device))
    return batch


def predict_images(
    cfg: dict[str, Any],
    ckpt_path: str | Path,
    path: str | Path,
    batch_size: int = 8,
    accelerator: str = "auto",
    devices: int = 1,
) -> list[SinglePrediction]:
    """Run inference on a single image path or a directory of images."""
    path = Path(path)
    if not path.exists():
        msg = f"Path not found: {path}"
        raise FileNotFoundError(msg)
    image_size = _image_size_tuple(cfg)
    h, w = image_size
    resize_tf = Compose([Resize((h, w), antialias=True)])
    dataset = PredictDataset(path=path, transform=resize_tf, image_size=(h, w))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_model_from_checkpoint(cfg, ckpt_path)
    engine = Engine(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(Path(cfg["trainer"].get("default_root_dir", "outputs/tmp_engine"))),
        enable_checkpointing=False,
        logger=False,
    )
    t0 = time.perf_counter()
    raw = engine.predict(
        model=model,
        dataloaders=loader,
        ckpt_path=str(ckpt_path),
        return_predictions=True,
    )
    elapsed = time.perf_counter() - t0
    batches = _flatten_predict_outputs(raw)
    n_items = sum(b.image.shape[0] for b in batches) if batches else 0
    per_item = elapsed / max(n_items, 1)
    results: list[SinglePrediction] = []
    for b in batches:
        bs = b.image.shape[0]
        scores = b.pred_score.detach().float().cpu().numpy() if b.pred_score is not None else np.zeros(bs)
        labels = b.pred_label.detach().cpu().numpy() if b.pred_label is not None else np.zeros(bs, dtype=bool)
        maps = b.anomaly_map.detach().float().cpu().numpy() if b.anomaly_map is not None else None
        masks = b.pred_mask.detach().cpu().numpy() if getattr(b, "pred_mask", None) is not None else None
        paths = list(b.image_path)
        for i in range(bs):
            amap = maps[i, 0] if maps is not None else np.zeros((h, w), dtype=np.float32)
            pmask = masks[i, 0] if masks is not None else None
            if pmask is None:
                pmask = anomaly_map_to_binary_mask(amap)
            else:
                pmask = pmask.astype(bool)
            results.append(
                SinglePrediction(
                    image_path=paths[i],
                    pred_score=float(scores[i].item() if scores.ndim > 0 else scores),
                    pred_label_anomalib=bool(labels[i]),
                    anomaly_map=amap,
                    pred_mask=pmask,
                    inference_time_s=float(per_item),
                    backbone=cfg["model"]["backbone"],
                    model_name=cfg["model"]["name"],
                )
            )
    return results


def predict_paths(
    cfg: dict[str, Any],
    ckpt_path: str | Path,
    image_paths: Sequence[str | Path],
    batch_size: int = 8,
    accelerator: str = "auto",
    devices: int = 1,
) -> list[SinglePrediction]:
    """Run inference on an explicit list of image paths (may span folders)."""
    out: list[SinglePrediction] = []
    for p in image_paths:
        out.extend(
            predict_images(
                cfg,
                ckpt_path,
                Path(p),
                batch_size=min(batch_size, 8),
                accelerator=accelerator,
                devices=devices,
            )
        )
    return out


def collect_test_predictions_for_metrics(
    model: Padim | Patchcore,
    test_loader: Iterable,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror ``test`` behaviour: preprocess, forward, post-process, collect arrays.

    Returns ``(y_true, y_score_raw, y_pred)`` with labels in ``{0,1}``.
    """
    model.eval()
    ys: list[int] = []
    raw_scores: list[float] = []
    preds: list[int] = []
    with torch.inference_mode():
        for batch in test_loader:
            batch = move_image_batch_to_device(batch, device)
            if model.pre_processor and model.pre_processor.transform:
                batch.image, batch.gt_mask = model.pre_processor.transform(batch.image, batch.gt_mask)
            out = model.test_step(batch, 0)
            raw = out.pred_score.detach().float().cpu().numpy().ravel()
            if model.post_processor is not None:
                model.post_processor.post_process_batch(out)
            gt = out.gt_label.detach().cpu().numpy().astype(np.int32).ravel()
            pl = out.pred_label.detach().cpu().numpy().ravel()
            for j in range(gt.shape[0]):
                ys.append(int(gt[j]))
                raw_scores.append(float(raw[j]))
                preds.append(int(bool(pl[j])))
    return (
        np.asarray(ys, dtype=np.int32),
        np.asarray(raw_scores, dtype=np.float64),
        np.asarray(preds, dtype=np.int32),
    )
