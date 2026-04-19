"""Path setup and small filesystem helpers for CLI scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Repository root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parents[1]


def ensure_src_on_path() -> Path:
    """Ensure repository root is on ``sys.path`` (for ``import src...``)."""
    root = repo_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def find_latest_checkpoint(root: Path, pattern: str = "*.ckpt") -> Path | None:
    """Return the most recently modified checkpoint under ``root`` (recursive)."""
    if not root.exists():
        return None
    ckpts = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
