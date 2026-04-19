"""Train PaDiM on a single MVTec AD category."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.utils as su  # noqa: E402

su.ensure_src_on_path()

from anomalib.engine import Engine  # noqa: E402
from src.config import ProjectPaths, load_yaml  # noqa: E402
from src.data_module import assert_category_structure, build_mvtec_datamodule  # noqa: E402
from src.inference import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PaDiM on MVTec AD.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "padim_config.yaml")
    parser.add_argument("--category", type=str, default=None, help="Override YAML category.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.category:
        cfg["data"]["category"] = args.category
    paths = ProjectPaths.from_config(cfg)
    assert_category_structure(paths.mvtec_root, cfg["data"]["category"])

    dm = build_mvtec_datamodule(
        root=paths.mvtec_root,
        category=cfg["data"]["category"],
        train_batch_size=int(cfg["data"]["train_batch_size"]),
        eval_batch_size=int(cfg["data"]["eval_batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        seed=cfg.get("project", {}).get("seed"),
    )
    model = build_model(cfg)
    trainer_cfg = cfg.get("trainer", {})
    out_dir = Path(trainer_cfg.get("default_root_dir", "outputs/models/padim"))
    out_dir = out_dir / cfg["data"]["category"]
    engine = Engine(
        accelerator=str(trainer_cfg.get("accelerator", "auto")),
        devices=int(trainer_cfg.get("devices", 1)),
        default_root_dir=str(out_dir),
    )
    engine.fit(model=model, datamodule=dm)
    ckpt = engine.checkpoint_callback.best_model_path if engine.checkpoint_callback else None
    meta = {
        "checkpoint": ckpt,
        "category": cfg["data"]["category"],
        "model": "Padim",
        "output_dir": str(out_dir),
    }
    report_path = paths.reports_dir / f"train_padim_{cfg['data']['category']}.json"
    su.dump_json(meta, report_path)
    print("Training complete.")
    print(f"Metadata written to {report_path}")
    if ckpt:
        print(f"Best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
