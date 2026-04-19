"""MVTec AD datamodule factory (Anomalib 2.x)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anomalib.data import MVTecAD


# Official MVTec AD 2D categories (15)
MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def build_mvtec_datamodule(
    root: str | Path,
    category: str,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    seed: int | None = None,
) -> MVTecAD:
    """Create an :class:`MVTecAD` Lightning datamodule.

    ``root`` should follow the official layout::

        root/
          bottle/
            train/good/*.png
            test/good|defect/*.png
            ground_truth/...
    """
    root_path = Path(root)
    if category not in MVTEC_CATEGORIES:
        msg = f"Unknown category {category!r}. Expected one of {MVTEC_CATEGORIES}."
        raise ValueError(msg)
    if not root_path.is_dir():
        msg = (
            f"MVTec root not found: {root_path}. "
            "Download MVTec AD and extract it (see scripts/download_dataset.md)."
        )
        raise FileNotFoundError(msg)
    return MVTecAD(
        root=str(root_path),
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
    )


def assert_category_structure(root: Path, category: str) -> None:
    """Raise if expected train folder is missing."""
    train_good = root / category / "train" / "good"
    if not train_good.is_dir():
        msg = f"Expected training images at {train_good}"
        raise FileNotFoundError(msg)


def config_datamodule_section(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``data`` subsection from a full config dict."""
    return cfg["data"]
