"""Streamlit demo for PaDiM / PatchCore anomaly inspection."""

from __future__ import annotations

import copy
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.utils as su  # noqa: E402

su.ensure_src_on_path()

from src.config import ProjectPaths, load_yaml  # noqa: E402
from src.data_module import MVTEC_CATEGORIES  # noqa: E402
from src.inference import predict_images  # noqa: E402
from src.thresholding import normalized_score_decision  # noqa: E402
from src.visualization import anomaly_map_to_color_heatmap, overlay_heatmap_on_image, resize_map_to_image  # noqa: E402


def _default_ckpt(model_key: str, category: str) -> Path | None:
    base = ROOT / "outputs" / "models" / model_key / category
    return su.find_latest_checkpoint(base)


@st.cache_resource
def _load_cfg(config_path: str) -> dict:
    return load_yaml(Path(config_path))


def main() -> None:
    st.set_page_config(page_title="Visual QC — Anomaly Detection", layout="wide")
    st.title("Automated visual quality control")
    st.caption("PaDiM / PatchCore via Anomalib · MVTec AD categories")

    with st.sidebar:
        model_choice = st.selectbox("Model", ["Padim", "Patchcore"], index=0)
        category = st.selectbox("MVTec category", MVTEC_CATEGORIES, index=0)
        config_path = ROOT / "configs" / (
            "padim_config.yaml" if model_choice == "Padim" else "patchcore_config.yaml"
        )
        cfg = copy.deepcopy(_load_cfg(str(config_path)))
        cfg["data"]["category"] = category
        paths = ProjectPaths.from_config(cfg)

        ckpt_default = _default_ckpt(model_choice.lower(), category)
        ckpt_in = st.text_input("Checkpoint path", value=str(ckpt_default) if ckpt_default else "")
        threshold = st.slider("Normalized score threshold", 0.0, 1.0, 0.5, 0.01)
        st.markdown("---")
        st.markdown(
            f"**Backbone:** `{cfg['model']['backbone']}`  \n"
            f"**Image size:** `{cfg['preprocess']['image_size']}`  \n"
            f"**Dataset root:** `{paths.mvtec_root}`"
        )

    tab_single, tab_batch = st.tabs(["Single image", "Batch folder"])

    ckpt_path = Path(ckpt_in) if ckpt_in else None
    ckpt_ok = ckpt_path is not None and ckpt_path.is_file()
    if not ckpt_ok:
        st.warning("Provide a valid checkpoint file (.ckpt) in the sidebar to run inference.")

    with tab_single:
        up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp"])
        if not ckpt_ok:
            st.info("Checkpoint required for inference.")
        elif up is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
                tmp.write(up.getbuffer())
                tmp_path = Path(tmp.name)
            t0 = time.perf_counter()
            try:
                preds = predict_images(cfg, ckpt_path, tmp_path, batch_size=1)
            finally:
                tmp_path.unlink(missing_ok=True)
            dt = time.perf_counter() - t0
            if not preds:
                st.error("Inference returned no results.")
                return
            p = preds[0]
            decision = normalized_score_decision(p.pred_score, threshold)
            raw = np.array(Image.open(up).convert("RGB"))
            h, w = raw.shape[:2]
            amap = resize_map_to_image(p.anomaly_map, h, w)
            heat_bgr = anomaly_map_to_color_heatmap(amap)
            overlay = overlay_heatmap_on_image(raw, heat_bgr)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Input")
                st.image(raw, use_container_width=True)
            with c2:
                st.subheader("Heatmap (resized)")
                st.image(cv2_rgb(heat_bgr), use_container_width=True)
            with c3:
                st.subheader("Overlay")
                st.image(overlay, use_container_width=True)

            st.metric("Anomaly score (normalized)", f"{p.pred_score:.4f}")
            st.metric("Label (slider)", decision.label)
            st.metric("Label (checkpoint)", "Defect" if p.pred_label_anomalib else "Pass")
            st.metric("Inference time (approx.)", f"{dt:.3f}s")

            row = {
                "path": up.name,
                "pred_score": p.pred_score,
                "label_slider": decision.label,
                "label_ckpt": "Defect" if p.pred_label_anomalib else "Pass",
                "threshold": threshold,
                "infer_s": dt,
            }
            st.download_button(
                "Download result row (CSV)",
                data=pd.DataFrame([row]).to_csv(index=False).encode("utf-8"),
                file_name="prediction_row.csv",
                mime="text/csv",
            )

    with tab_batch:
        folder = st.text_input("Folder with images", value="")
        run_batch = st.button("Run batch inference")
        if not ckpt_ok:
            st.info("Checkpoint required for batch inference.")
        elif run_batch and folder:
            fp = Path(folder)
            if not fp.is_dir():
                st.error("Folder does not exist.")
            else:
                with st.spinner("Running batch inference..."):
                    t0 = time.perf_counter()
                    preds = predict_images(cfg, ckpt_path, fp, batch_size=8)
                    dt = time.perf_counter() - t0
                rows = []
                for p in preds:
                    d = normalized_score_decision(p.pred_score, threshold)
                    rows.append(
                        {
                            "path": p.image_path,
                            "pred_score": p.pred_score,
                            "label": d.label,
                            "label_ckpt": "Defect" if p.pred_label_anomalib else "Pass",
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                st.caption(f"Processed {len(rows)} images in {dt:.2f}s wall time.")
                st.download_button(
                    "Download batch CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )


def cv2_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
