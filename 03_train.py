"""
Stage 3 — Training (single train/val tower split).

Split manually by placing towers in two folders and passing them as --train-dir /
--val-dir. Train = every tower in --train-dir; val = every tower in --val-dir (any
number of val towers; their metrics are pooled).

Two input modes:
  --from-las   (default)  Folders contain downsampled .las/.laz files. TowerDataset
                          computes the 17-D features once at 0° and applies full
                          augmentation (Z-rot + scale + jitter + colour) + class-weighted
                          point dropout analytically, resampled per-epoch.

  --from-cache            Folders are cache roots whose immediate sub-directories are
                          per-tower chunk_*.npz dirs (scripts/02_extract_features.py).
                          The cached features ARE the 0° features and the SAME per-epoch
                          augmentation + dropout are applied, so the training distribution
                          is identical to --from-las without recomputing KNN features.

Example (.las folders):
    python scripts/03_train.py \
        --config pointcloud_seg/sample_configs/config.yaml \
        --train-dir data/split/train \
        --val-dir   data/split/val \
        --save-dir  outputs/run_01

From cache (cache-root folders):
    python scripts/03_train.py \
        --config my_config.yaml --from-cache \
        --train-dir data/cache/train \
        --val-dir   data/cache/val \
        --save-dir  outputs/run_cached
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointcloud_seg.config import Config
from pointcloud_seg.train.dataset import list_towers_in_dir
from pointcloud_seg.train.trainer import train_single_fold


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Train CellTowerSegmenter (single fold)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--save-dir", required=True, help="Where to write checkpoints + logs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # folder-based split
    parser.add_argument("--train-dir", required=True,
                        help="Folder of training towers (.las/.laz, or cache tower dirs with --from-cache)")
    parser.add_argument("--val-dir", required=True,
                        help="Folder of validation towers (any number; metrics pooled)")

    # cache mode
    parser.add_argument("--from-cache", action="store_true",
                        help="Folders are cache roots of per-tower chunk_*.npz dirs")

    # warm-start from an existing checkpoint (weights only; fresh optimizer/schedule, epoch 1)
    parser.add_argument("--init-weights", default=None,
                        help="Path to a .pt checkpoint; load its model weights (strict) as "
                             "initialization, then train fresh. Config must build the same "
                             "architecture as the checkpoint.")

    # overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    # apply overrides
    if args.lr is not None:
        cfg.lr = args.lr
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.amp is not None:
        cfg.amp = args.amp

    # resolve towers from the two folders (per intake mode)
    train_paths = list_towers_in_dir(args.train_dir, from_cache=args.from_cache)
    val_paths = list_towers_in_dir(args.val_dir, from_cache=args.from_cache)

    os.makedirs(args.save_dir, exist_ok=True)
    cfg.checkpoint_dir = args.save_dir
    cfg.to_yaml(os.path.join(args.save_dir, "config_used.yaml"))

    print(f"Config saved → {args.save_dir}/config_used.yaml")
    print(f"MLflow URI    : {cfg.log_dir}")
    print(f"Experiment    : {cfg.mlflow_experiment}")
    print(f"Train towers  : {len(train_paths)} from {args.train_dir}")
    print(f"Val towers    : {[Path(p).stem for p in val_paths]} from {args.val_dir}")

    train_single_fold(
        config=cfg,
        train_paths=train_paths,
        val_paths=val_paths,
        save_dir=args.save_dir,
        device=args.device,
        from_cache=args.from_cache,
        cache_dir=cfg.cache_dir,
        init_weights=args.init_weights,
    )


if __name__ == "__main__":
    main()
