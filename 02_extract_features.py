"""
Stage 2+3 — Global feature extraction on downsampled clouds.

Reads each downsampled .las from config.downsampled_dir, computes the full
17-feature descriptor (normals, PCA geometry, height, RGB), slices the cloud
into overlapping super-chunks, and saves everything as a .npz cache.

Saved .npz keys per chunk:
  features       (M, 17) float32  — model-ready input
  labels         (M,)    int64    — classification labels (−1 if unlabelled)
  xyz            (M, 3)  float64  — chunk point coordinates (original, not local)
  point_indices  (M,)    int64    — indices into the downsampled cloud
  chunk_counts   (M,)    int32    — how many chunks each point appears in
  tower_id               str      — stem of the source file

Output directory structure:
  <cache_dir>/<tower_stem>/chunk_000.npz
                           chunk_001.npz
                           ...
                           meta.json         (tower-level stats)

These .npz files are read directly by scripts/03_train.py (--from-cache mode).

Usage:
    python scripts/02_extract_features.py --config pointcloud_seg/sample_configs/config.yaml
    python scripts/02_extract_features.py --config my_config.yaml --files ds_a.las ds_b.las
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointcloud_seg.config import Config
from pointcloud_seg.train.dataset import _load_las
from pointcloud_seg.preprocessing.features import compute_global_features
from pointcloud_seg.preprocessing.chunking import slice_into_superchunks
from pointcloud_seg.preprocessing.augmentation import augment_features, class_weighted_dropout


def process_one_tower(las_path: str, cfg: Config, cache_root: Path) -> dict:
    tower_id = Path(las_path).stem
    tower_dir = cache_root / tower_id
    tower_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    xyz, rgb, labels = _load_las(las_path, cfg)
    precomputed = compute_global_features(xyz, rgb, cfg)
    chunks = slice_into_superchunks(
        xyz, rgb, precomputed, cfg,
        labels=labels,
        tower_id=tower_id,
    )

    saved_chunks = []
    for i, chunk in enumerate(chunks):
        out = tower_dir / f"chunk_{i:03d}.npz"
        lbl = chunk.labels if chunk.labels is not None else np.full(len(chunk.features), -1, dtype=np.int64)
        np.savez_compressed(
            str(out),
            features=chunk.features,
            labels=lbl,
            xyz=xyz[chunk.point_indices],
            point_indices=chunk.point_indices,
            chunk_counts=chunk.chunk_counts,
        )
        saved_chunks.append(str(out))

    # Offline augmentation: generate one copy per rotation angle.
    # Each augmented chunk gets the fixed rotation + random scale/jitter/colour jitter
    # + class-weighted dropout baked in. Named aug_{angle:03d}_{chunk_idx:03d}.npz so
    # the validation glob (chunk_*.npz) never picks them up.
    # Augmented chunks only need features + labels — point_indices/xyz are
    # only used by InferenceTowerData (validation), which never loads aug_*.npz.
    # Background-only chunks (all labels in bg_classes) are skipped: augmenting
    # them produces more dominant-class data with no rare-class signal.
    bg_classes = list(cfg.bg_classes)
    bg_flags = []
    for chunk in chunks:
        lbl = chunk.labels
        valid = lbl[lbl >= 0] if lbl is not None else np.array([], dtype=np.int64)
        bg_flags.append(len(valid) > 0 and bool(np.isin(valid, bg_classes).all()))

    saved_aug_chunks = []
    for rot in cfg.offline_aug_rotations:
        for i, chunk in enumerate(chunks):
            if bg_flags[i]:
                continue
            lbl = chunk.labels if chunk.labels is not None else np.full(len(chunk.features), -1, dtype=np.int64)
            aug_feats = augment_features(chunk.features, cfg, rotation_deg=rot)
            valid_mask = lbl >= 0
            if cfg.aug_class_dropout and valid_mask.any():
                aug_feats, aug_lbl = class_weighted_dropout(
                    aug_feats[valid_mask], lbl[valid_mask], cfg
                )
            else:
                aug_lbl = lbl[valid_mask] if valid_mask.any() else lbl
                aug_feats = aug_feats[valid_mask] if valid_mask.any() else aug_feats
            out = tower_dir / f"aug_{rot:03d}_{i:03d}.npz"
            np.savez_compressed(str(out), features=aug_feats, labels=aug_lbl)
            saved_aug_chunks.append(str(out))

    elapsed = time.time() - t0
    meta = {
        "tower_id": tower_id,
        "source_las": las_path,
        "n_points": int(len(xyz)),
        "n_chunks": len(chunks),
        "n_aug_chunks": len(saved_aug_chunks),
        "chunk_sizes": [int(len(c.features)) for c in chunks],
        "has_labels": labels is not None,
        "elapsed_sec": round(elapsed, 2),
        "chunk_files": saved_chunks,
        "aug_chunk_files": saved_aug_chunks,
    }
    with open(tower_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    tqdm.write(
        f"  [{tower_id}] {len(xyz):,} pts → {len(chunks)} chunk(s) + "
        f"{len(saved_aug_chunks)} aug chunk(s) in {elapsed:.1f}s"
    )
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2+3: compute features and save .npz chunk cache"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input-dir", default=None, help="Override config.downsampled_dir")
    parser.add_argument("--cache-dir", default=None, help="Override config.cache_dir")
    parser.add_argument("--files", nargs="+", default=None, help="Explicit downsampled .las paths")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip towers whose cache directory already exists")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.input_dir:
        cfg.downsampled_dir = args.input_dir
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    if cfg.cache_dir is None:
        cfg.cache_dir = str(Path(cfg.downsampled_dir).parent / "cache")
        print(f"cache_dir not set in config — using {cfg.cache_dir}")

    cache_root = Path(cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.files:
        las_paths = args.files
    else:
        ds_dir = Path(cfg.downsampled_dir)
        las_paths = sorted([str(p) for p in ds_dir.glob("*.las")] +
                           [str(p) for p in ds_dir.glob("*.laz")])

    if not las_paths:
        print(f"No downsampled .las files found in {cfg.downsampled_dir}")
        sys.exit(1)

    if args.skip_existing:
        before = len(las_paths)
        las_paths = [p for p in las_paths
                     if not (cache_root / Path(p).stem).exists()]
        print(f"Skipping {before - len(las_paths)} already-cached tower(s)")

    print(f"Feature extraction for {len(las_paths)} tower(s)")
    print(f"  knn_small={cfg.knn_small}  knn_large={cfg.knn_large}")
    print(f"  chunk_target={cfg.chunk_target_points:,}  overlap={cfg.overlap_fraction}")
    print(f"  cache → {cache_root}\n")

    all_meta = []
    pbar = tqdm(las_paths, desc="Extracting features", unit="tower")
    for path in pbar:
        pbar.set_postfix_str(Path(path).stem)
        try:
            meta = process_one_tower(path, cfg, cache_root)
            all_meta.append(meta)
        except Exception as e:
            tqdm.write(f"  [ERROR] {path}: {e}")

    manifest_path = cache_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    total_chunks = sum(m["n_chunks"] for m in all_meta)
    total_aug_chunks = sum(m.get("n_aug_chunks", 0) for m in all_meta)
    total_pts = sum(m["n_points"] for m in all_meta)
    print(f"\n=== Stage 2+3 complete ===")
    print(f"  Towers processed : {len(all_meta)}")
    print(f"  Total points     : {total_pts:,}")
    print(f"  Original chunks  : {total_chunks}")
    print(f"  Augmented chunks : {total_aug_chunks}  ({len(cfg.offline_aug_rotations)} rotation(s) × {total_chunks})")
    print(f"  Total train chunks: {total_chunks + total_aug_chunks}")
    print(f"  Cache location   : {cache_root}")
    print(f"  Manifest         : {manifest_path}")
    print("\nTrain from this cache with:")
    print(f"  scripts/03_train.py --from-cache --train-dir {cache_root} --val-dir {cache_root}")
    print("  (point --train-dir / --val-dir at the cache roots holding the per-tower chunk dirs)")


if __name__ == "__main__":
    main()
