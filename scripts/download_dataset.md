# Downloading MVTec AD

MVTec AD is maintained by MVTec Software GmbH and is widely used for benchmarking anomaly detection.

1. Visit the official download page: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. Complete the dataset request form and download the archive.
3. Extract the archive so that this repository points at the folder that **directly contains the 15 category directories** (for example `bottle`, `cable`, …).

Expected layout:

```text
data/mvtec/
  bottle/
    train/good/*.png
    test/...
    ground_truth/...
  cable/
  ...
```

Set `data.mvtec_root` in `configs/padim_config.yaml` (or pass a matching layout) to `data/mvtec` if you extract the dataset there.

**Note:** Redistribution of MVTec AD through this repository is not permitted; only these instructions are included.
