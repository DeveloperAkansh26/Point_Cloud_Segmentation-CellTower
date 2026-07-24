"""
Full end-to-end pipeline: downsample → feature extraction → train (single split).

This is the one-command way to go from raw .las files to a trained model.
It calls each stage in order, passing outputs from one stage to the next.
Training is a single train/val tower split; you hold out one or more val towers by
stem (--val-stems), and their validation metrics are pooled. Every other downsampled
tower is used for training.

Usage:
    python scripts/run_pipeline.py \
        --config    pointcloud_seg/sample_configs/config.yaml \
        --raw-dir   /data/raw \
        --val-stems tower_03 tower_11 \
        --save-dir  /outputs/run_01

(--val-stems is required — there is no automatic default.)

Skip stages you've already run:
    python scripts/run_pipeline.py \
        --config   my_config.yaml \
        --raw-dir  /data/raw \
        --save-dir /outputs/run_01 \
        --skip-downsample    # already have downsampled files
        --skip-features      # already have .npz cache

Train from an existing cache (normally paired with --skip-downsample). The cache root
defaults to <save-dir>/cache; point elsewhere with --cache-dir. Val/train tower stems are
still read from the downsampled .las in <save-dir>/downsampled, so those must exist
(or be found via --skip-downsample):
    python scripts/run_pipeline.py \
        --config   my_config.yaml \
        --save-dir /outputs/run_01 \
        --skip-downsample --from-cache --cache-dir /data/feature_cache \
        --val-stems tower_03_3.0cm

Environment variables required for MLflow:
    export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
    export MLFLOW_TRACKING_USERNAME=user   # if auth needed
    export MLFLOW_TRACKING_PASSWORD=pass
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointcloud_seg.config import Config
from pointcloud_seg.preprocessing.downsample import batch_downsample
from pointcloud_seg.train.trainer import train_single_fold


def find_las_files(directory: str) -> List[str]:
    d = Path(directory)
    return sorted([str(p) for p in d.glob("*.las")] + [str(p) for p in d.glob("*.laz")])


def stage_banner(name: str, step: int, total: int):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  Step {step}/{total}: {name}")
    print(f"{line}")


def main():
    parser = argparse.ArgumentParser(
        description="Full preprocessing + training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--raw-dir", default=None, help="Override config.raw_dir")
    parser.add_argument("--save-dir", required=True, help="Root output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--val-stems", nargs="+", required=True,
                        help="Tower stem(s) to hold out as validation (any number; metrics pooled)")

    parser.add_argument("--skip-downsample", action="store_true",
                        help="Skip Stage 1 (already have downsampled files)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip Stage 2 feature extraction (train from .las directly)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached .npz chunks for training (normally paired with --skip-downsample)")
    parser.add_argument("--cache-dir", default=None,
                        help="External feature cache root (default: <save-dir>/cache)")

    # config overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    # ------------------------------------------------------------------ setup
    cfg = Config.from_yaml(args.config)
    if args.raw_dir:
        cfg.raw_dir = args.raw_dir
    if args.lr is not None:
        cfg.lr = args.lr
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.workers is not None:
        cfg.pdal_workers = args.workers

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds_dir = save_dir / "downsampled"
    cache_dir = Path(args.cache_dir) if args.cache_dir else save_dir / "cache"
    ckpt_dir = save_dir / "checkpoints"

    cfg.downsampled_dir = str(ds_dir)
    cfg.cache_dir = str(cache_dir)
    cfg.checkpoint_dir = str(ckpt_dir)
    cfg.log_dir = cfg.log_dir  # honour from config (MLflow URI)

    cfg.to_yaml(str(save_dir / "config_used.yaml"))

    total_steps = 3 if not args.skip_features else 2
    t_start = time.time()

    # --------------------------------------------------------------- stage 1
    stage_banner("Poisson-disk downsampling (PDAL)", 1, total_steps)
    if args.skip_downsample:
        raw_paths = find_las_files(cfg.raw_dir)
        print(f"Skipping — looking for existing downsampled files in {ds_dir}")
        ds_paths = find_las_files(str(ds_dir))
        if not ds_paths:
            print("ERROR: --skip-downsample set but no .las files found in downsampled dir")
            sys.exit(1)
    else:
        raw_paths = find_las_files(cfg.raw_dir)
        if not raw_paths:
            print(f"ERROR: No .las/.laz files in {cfg.raw_dir}")
            sys.exit(1)
        print(f"Found {len(raw_paths)} raw file(s)")
        ds_paths, failed = batch_downsample(raw_paths, cfg)
        if failed:
            print(f"ERROR: {len(failed)} file(s) failed downsampling — aborting")
            sys.exit(1)

    print(f"Downsampled files: {len(ds_paths)}")

    # --------------------------------------------------------------- stage 2
    if not args.skip_features and not args.from_cache:
        stage_banner("Feature extraction → .npz cache", 2, total_steps)
        import json
        from pointcloud_seg.train.dataset import _load_las
        from pointcloud_seg.preprocessing.features import compute_global_features
        from pointcloud_seg.preprocessing.chunking import slice_into_superchunks
        import numpy as np

        from tqdm import tqdm

        cache_dir.mkdir(parents=True, exist_ok=True)
        all_meta = []

        pbar = tqdm(ds_paths, desc="Feature extraction", unit="tower")
        for las_path in pbar:
            tower_id = Path(las_path).stem
            pbar.set_postfix_str(tower_id)
            tower_dir = cache_dir / tower_id
            tower_dir.mkdir(exist_ok=True)
            t0 = time.time()

            xyz, rgb, labels = _load_las(las_path, cfg)
            precomputed = compute_global_features(xyz, rgb, cfg)
            chunks = slice_into_superchunks(xyz, rgb, precomputed, cfg,
                                            labels=labels, tower_id=tower_id)
            saved = []
            for i, chunk in enumerate(chunks):
                out = tower_dir / f"chunk_{i:03d}.npz"
                lbl = chunk.labels if chunk.labels is not None else np.full(len(chunk.features), -1, dtype=np.int64)
                np.savez_compressed(str(out),
                                    features=chunk.features, labels=lbl,
                                    xyz=xyz[chunk.point_indices],
                                    point_indices=chunk.point_indices,
                                    chunk_counts=chunk.chunk_counts)
                saved.append(str(out))
            meta = {
                "tower_id": tower_id, "n_points": int(len(xyz)),
                "n_chunks": len(chunks), "elapsed_sec": round(time.time() - t0, 2),
                "chunk_files": saved,
            }
            all_meta.append(meta)
            tqdm.write(f"  [{tower_id}] {len(xyz):,} pts → {len(chunks)} chunks in {time.time() - t0:.1f}s")

        with open(cache_dir / "manifest.json", "w") as f:
            json.dump(all_meta, f, indent=2)
        print(f"Cache written → {cache_dir}")
        use_cache = True
    else:
        use_cache = args.from_cache

    # --------------------------------------------------------------- stage 3
    stage_banner("Training (single train/val split)", total_steps, total_steps)
    print(f"Device: {args.device}")

    if len(ds_paths) < 2:
        print("ERROR: need at least 2 towers (>=1 train + 1 val)")
        sys.exit(1)

    # explicit, manual val selection — no arbitrary default
    val_stems = set(args.val_stems)
    ds_stems = {Path(p).stem for p in ds_paths}
    missing = val_stems - ds_stems
    if missing:
        print(f"ERROR: val stem(s) not found in downsampled files: {sorted(missing)}")
        sys.exit(1)

    train_stems = [Path(p).stem for p in ds_paths if Path(p).stem not in val_stems]
    if not train_stems:
        print("ERROR: --val-stems selected every tower; no training towers left")
        sys.exit(1)

    if use_cache:
        # train/val resolve to per-tower cache directories (cache_dir/<stem>)
        train_paths = [str(cache_dir / s) for s in train_stems]
        val_paths = [str(cache_dir / s) for s in sorted(val_stems)]
    else:
        train_paths = [p for p in ds_paths if Path(p).stem in train_stems]
        val_paths = [p for p in ds_paths if Path(p).stem in val_stems]

    print(f"Train: {len(train_paths)} towers  |  Val: {sorted(val_stems)}")

    train_single_fold(
        config=cfg,
        train_paths=train_paths,
        val_paths=val_paths,
        save_dir=str(ckpt_dir),
        device=args.device,
        from_cache=use_cache,
        cache_dir=str(cache_dir),
    )

    # -----------------------------------------------------------------  done
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total_time/60:.1f} min")
    print(f"  Checkpoints → {ckpt_dir}")
    print(f"  Config used → {save_dir}/config_used.yaml")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
