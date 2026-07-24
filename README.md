# pointcloud_seg — Cell-Tower Point-Cloud Semantic Segmentation

A deep-learning pipeline for **per-point semantic segmentation of cell-tower LiDAR point clouds**.
It takes raw `.las`/`.laz` scans of telecom towers and labels every point as one of five classes,
using an EdgeConv (DGCNN-style) encoder–decoder with grid pooling, an attention bottleneck, and a
combined weighted-CE / Lovász-softmax loss. The workflow runs as discrete, re-runnable stages
(downsample → feature extraction → train → infer → post-process) plus a one-command end-to-end driver.

## Classes

| id | name | notes |
|----|------|-------|
| 0 | Noise | ground, vegetation, clutter |
| 1 | TowerBody | tower structure (raw label 5 is folded in via `merge_labels`) |
| 2 | PanelAntenna | flat panel antennas |
| 3 | MicrowaveDish | microwave dishes |
| 4 | RRU | remote radio units |

## Pipeline overview

```
raw .las/.laz
    │  Stage 1  Poisson-disk downsampling (PDAL, ~3 cm)
    ▼
downsampled .las
    │  Stage 2  21-D per-point features (normals, multi-scale PCA geometry, height, RGB)
    │           + slicing into overlapping super-chunks → .npz cache
    ▼
feature cache (.npz)
    │  Stage 3  Training (single train/val tower split, MLflow-tracked)
    ▼
checkpoints (.pt)
    │  Stage 4  Inference w/ test-time augmentation → labelled .las
    ▼
labelled .las
    │  Stage 5  DBSCAN speckle cleanup / cluster post-processing
    ▼
final labelled .las
```

## Repository layout

```
.
├── 01_downsample.py            # Stage 1 — batch PDAL Poisson-disk downsampling
├── 02_extract_features.py      # Stage 2/3 — feature extraction → .npz cache
├── 03_train.py                 # Stage 3 — training (single train/val split)
├── 04_infer.py                 # Stage 4 — inference + TTA → labelled .las
├── 05_dbscan_postprocess.py    # Stage 5 — DBSCAN speckle cleanup on labelled cloud
├── 05_dbscan_cluster.py        # standalone DBSCAN clustering utility (for CloudCompare)
├── run_pipeline.py             # end-to-end driver (downsample → features → train)
├── notes.txt                   # experiment log (run_01 … run_48)
└── pointcloud_seg/
    ├── config.py               # dataclass Config (+ YAML load/save)
    ├── infer.py                # inference orchestration
    ├── requirements.txt        # Python dependencies
    ├── sample_configs/
    │   └── config.yaml         # fully-commented reference configuration
    ├── preprocessing/          # downsample, features, chunking, augmentation, merge, projection, postprocess
    ├── model/                  # segmenter, edgeconv, attention, pooling, norm
    └── train/                  # trainer, dataset, loss, ema
```

## Model

An EdgeConv-based encoder–decoder ("rare-class-aware backbone"):

- **Input:** 21 per-point features (17 base geometric/colour + 4 object-scale PCA features that
  separate panel antennas from RRUs), with optional per-feature input standardization.
- **Encoder:** stacked EdgeConv blocks with a dynamic KNN graph, SE channel attention, and grid
  (voxel) pooling at each stage. Default widths `[32, 64, 128, 256]`, voxel sizes
  `[0.025, 0.05, 0.10, 0.20]` m.
- **Bottleneck:** multi-head self-attention.
- **Decoder:** EdgeConv blocks with nearest / interpolation unpooling.
- **Loss:** `(weighted-CE | focal) + Lovász-softmax`, with `log`/`median`/`inverse` class
  weighting to counter class imbalance.
- **Training aids:** cosine LR schedule with warmup, AMP, EMA, per-epoch augmentation
  (Z-rotation, scale, jitter, colour) + class-weighted point dropout, background-chunk skipping,
  and multiple checkpoint selectors (val-mIoU, val-loss, train-loss, composite).
- **Inference:** test-time augmentation over 0°/90°/180°/270° rotations, softmax-averaged, then
  projected back to full resolution.

## Installation

```bash
# 1. system prerequisite — PDAL (not a pip package)
conda install -c conda-forge pdal        # or: apt install pdal

# 2. python deps
pip install -r pointcloud_seg/requirements.txt

# 3. (GPU) install the CUDA build of torch instead of the CPU wheel, e.g.
#    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Core dependencies: `numpy`, `scipy`, `laspy` + `laszip` (LAS/LAZ I/O), `torch`, `mlflow`,
`PyYAML`, `tqdm`, and `scikit-learn` (DBSCAN utilities).

## Usage

All scripts are configured via a YAML file; start from
[pointcloud_seg/sample_configs/config.yaml](pointcloud_seg/sample_configs/config.yaml), which
documents every option.

### End-to-end (downsample → features → train)

```bash
python run_pipeline.py \
    --config    pointcloud_seg/sample_configs/config.yaml \
    --raw-dir   /data/raw \
    --val-stems tower_03 tower_11 \
    --save-dir  outputs/run_01
```

`--val-stems` is required (towers held out for validation; their metrics are pooled). Skip stages
you've already run with `--skip-downsample`, `--skip-features`, or train from an existing cache with
`--from-cache --cache-dir ...`.

### Stage by stage

```bash
# 1. downsample
python 01_downsample.py --config config.yaml --raw-dir /data/raw

# 2. features → .npz cache
python 02_extract_features.py --config config.yaml

# 3. train (manual train/val folders)
python 03_train.py --config config.yaml \
    --train-dir data/split/train --val-dir data/split/val \
    --save-dir outputs/run_01            # add --from-cache to train from .npz

# 4. infer on a new tower (with TTA)
python 04_infer.py --config config.yaml \
    --checkpoint outputs/run_01/best_model.pt \
    --raw data/raw/tower_new.las \
    --output results/tower_new_labelled.las

# 5. DBSCAN speckle cleanup on a labelled cloud
python 05_dbscan_postprocess.py --config config.yaml \
    --input  results/tower_new_labelled.las \
    --output results/tower_new_labelled_dbscan.las
```

### Experiment tracking

Training logs to MLflow. Point it at your tracking server before running:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_TRACKING_USERNAME=user     # if auth is enabled
export MLFLOW_TRACKING_PASSWORD=pass
```

By default `log_dir` uses a local SQLite MLflow store (see `config.yaml`).

## Data & outputs

Run artifacts (downsampled clouds, feature caches, checkpoints, MLflow DB) live under
`pointcloud_seg/data/` and are **not tracked in git** (see [.gitignore](.gitignore)). Provide your
own raw scans in `raw_dir` and point `--save-dir` wherever you want results written.

## Notes

[notes.txt](notes.txt) is the running experiment log (loss-function sweeps, augmentation/dropout
ablations, model-size changes, etc.) across runs `run_01`–`run_48`.
