"""YAML-driven configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a nested dictionary."""
    path = Path(path)
    if not path.is_file():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Top-level YAML must be a mapping, got {type(data)}"
        raise ValueError(msg)
    return data


def resolve_path(base: Path, maybe_relative: str | Path) -> Path:
    """Resolve a path that may be relative to ``base``."""
    p = Path(maybe_relative)
    return p if p.is_absolute() else (base / p).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem paths for a training run."""

    root: Path
    mvtec_root: Path
    outputs_dir: Path
    predictions_dir: Path
    reports_dir: Path
    figures_dir: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ProjectPaths:
        root = Path(cfg.get("project", {}).get("root", ".")).resolve()
        data_root = resolve_path(root, cfg["data"]["mvtec_root"])
        outputs = resolve_path(root, "outputs")
        return cls(
            root=root,
            mvtec_root=data_root,
            outputs_dir=outputs,
            predictions_dir=outputs / "predictions",
            reports_dir=outputs / "reports",
            figures_dir=outputs / "figures",
        )
