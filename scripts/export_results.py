"""Merge per-image prediction CSVs from a directory into one consolidated report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.utils as su  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge batch_report.csv files.")
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.predictions_root
    files = sorted(root.rglob("batch_report.csv"))
    if not files:
        raise SystemExit(f"No batch_report.csv found under {root}")
    dfs = [pd.read_csv(p) for p in files]
    merged = pd.concat(dfs, ignore_index=True)
    out = args.output or (root / "merged_predictions.csv")
    merged.to_csv(out, index=False)
    su.dump_json({"sources": [str(p) for p in files], "rows": len(merged)}, out.with_suffix(".json"))
    print(f"Merged {len(files)} files -> {out} ({len(merged)} rows).")


if __name__ == "__main__":
    main()
