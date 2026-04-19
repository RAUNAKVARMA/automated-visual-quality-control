# Automated visual quality control

Research-oriented **industrial anomaly detection** using the **MVTec AD** benchmark, **Anomalib** models (**PaDiM** primary, **PatchCore** baseline), and a **Streamlit** frontend for interactive inspection.

## Features

- **Dataset:** MVTec AD (15 categories), training on defect-free images only, evaluation on the official test split via `MVTecAD`.
- **Models:** PaDiM and PatchCore with **torchvision** CNN backbones.
- **Outputs:** pass/defect decision, anomaly score, heatmap, overlay, defect mask, JSON/CSV metrics, confusion matrix, ROC curve, optional pixel-level AUROC on masked regions, batch CSV reports.
- **Stack:** Python 3.11 (recommended), PyTorch, Anomalib 2.3.x, OpenCV, Streamlit, scikit-learn.

## Repository layout

```text
automated-visual-quality-control/
├── app/
│   └── streamlit_app.py
├── configs/
│   ├── padim_config.yaml
│   └── patchcore_config.yaml
├── data/
│   └── mvtec/              # place extracted MVTec AD here
├── notebooks/
│   └── eda.ipynb
├── outputs/
│   ├── models/
│   ├── predictions/
│   ├── reports/
│   └── figures/
├── scripts/
│   ├── download_dataset.md
│   ├── train_padim.py
│   ├── train_patchcore.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── batch_predict.py
│   ├── export_results.py
│   └── utils.py
├── src/
│   ├── data_module.py
│   ├── inference.py
│   ├── visualization.py
│   ├── thresholding.py
│   ├── metrics.py
│   └── config.py
├── tests/
│   └── test_inference.py
├── requirements.txt
└── README.md
```

## Setup

1. Install **Python 3.11** (3.10+ generally works).
2. Create a virtual environment and install dependencies:

```bash
cd automated-visual-quality-control
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Download **MVTec AD** and extract under `data/mvtec` so each category folder (e.g. `bottle/train/good`) exists. See `scripts/download_dataset.md`.

4. Adjust paths in `configs/padim_config.yaml` / `configs/patchcore_config.yaml` if your dataset is not at `data/mvtec`.

## Training

Train PaDiM (example category `bottle`):

```bash
python scripts/train_padim.py --config configs/padim_config.yaml --category bottle
```

Train PatchCore:

```bash
python scripts/train_patchcore.py --config configs/patchcore_config.yaml --category bottle
```

Checkpoints and Lightning logs are written under `outputs/models/<model>/<category>/` (see `trainer.default_root_dir` in each YAML).

## Evaluation

```bash
python scripts/evaluate.py --config configs/padim_config.yaml --category bottle
```

Writes:

- `outputs/reports/metrics_<Model>_<category>.json` and `.csv`
- `outputs/figures/<Model>_<category>/confusion_matrix.png`
- `outputs/figures/<Model>_<category>/roc_curve.png`
- Sample overlays under the same figures folder

You may pass `--ckpt path\to\model.ckpt`; otherwise the newest checkpoint under the model output directory is used.

## Single-image and batch inference

```bash
python scripts/predict.py --config configs/padim_config.yaml --ckpt outputs/models/padim/bottle/.../file.ckpt --image path\to\image.png
```

```bash
python scripts/batch_predict.py --config configs/padim_config.yaml --ckpt path\to\file.ckpt --input-dir path\to\folder --save-images
```

Merge multiple `batch_report.csv` files:

```bash
python scripts/export_results.py --predictions-root outputs/predictions
```

## Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Use the sidebar to pick **PaDiM** or **PatchCore**, the **category** (for default checkpoint discovery), and a **checkpoint** path. Upload an image or run a **batch folder** path. The **threshold** slider applies to **normalized** anomaly scores (same convention as Anomalib’s post-processed scores).

## Tests

```bash
python -m pytest tests/test_inference.py
```

## Implementation notes

- Anomalib **2.3.x** uses `MVTecAD`, `Padim`, `Patchcore`, and `Engine` (`fit`, `test`, `predict`).
- Paths are driven by YAML plus `ProjectPaths` in `src/config.py`.
- `src/inference.py` wraps `Engine.predict` with `PredictDataset` and resizing consistent with training image size.

## License

This project code is provided as a template for research and demos. **MVTec AD** has its own license; do not redistribute the dataset from this repository.
