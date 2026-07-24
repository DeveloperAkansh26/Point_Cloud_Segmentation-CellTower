import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import laspy
from tqdm import tqdm

from pointcloud_seg.config import Config
from pointcloud_seg.model.segmenter import CellTowerSegmenter
from pointcloud_seg.preprocessing.downsample import downsample_tower
from pointcloud_seg.preprocessing.features import compute_global_features
from pointcloud_seg.preprocessing.chunking import slice_into_superchunks
from pointcloud_seg.preprocessing.merge import merge_probs
from pointcloud_seg.preprocessing.postprocess import (
    dbscan_cleanup,
    reclassify_ground,
    resolve_antenna_rru,
)
from pointcloud_seg.preprocessing.projection import write_labelled_las
from pointcloud_seg.train.dataset import _apply_label_merge


# class index -> human name, for metric printing (mirrors the config.num_classes comment)
CLASS_NAMES = {0: "Noise", 1: "TowerBody", 2: "PanelAntenna", 3: "MicrowaveDish", 4: "RRU"}


def _load_las(path: str):
    las = laspy.read(path)
    xyz = np.stack([np.array(las.x), np.array(las.y), np.array(las.z)], axis=1).astype(np.float64)
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        rgb = np.stack([
            np.array(las.red, dtype=np.float32),
            np.array(las.green, dtype=np.float32),
            np.array(las.blue, dtype=np.float32),
        ], axis=1) / 65535.0
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.float32)
    return xyz, rgb, las


def _labels_from_las(las, config: Config) -> Optional[np.ndarray]:
    """
    Extract the ground-truth classification field from a loaded las and apply
    config.merge_labels (the SAME '5'->1 remap chokepoint training uses via
    dataset._load_las). Returns (N,) int64 labels, or None if the field is absent.
    """
    labels = None
    if hasattr(las, "raw_classification"):
        labels = np.array(las.raw_classification, dtype=np.int64)
    elif hasattr(las, "classification"):
        labels = np.array(las.classification, dtype=np.int64)
    return _apply_label_merge(labels, config)


def compute_iou(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """
    Per-class IoU, mIoU, and overall accuracy.

    Kept identical to pointcloud_seg.train.trainer.compute_iou so inference reports
    the exact same numbers as the training/validation logs. Duplicated locally on
    purpose: importing the trainer would pull in mlflow and the whole training stack
    just to run inference.
    """
    per_class = {}
    ious = []
    for c in range(num_classes):
        tp = int(((labels == c) & (preds == c)).sum())
        fp = int(((labels != c) & (preds == c)).sum())
        fn = int(((labels == c) & (preds != c)).sum())
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        per_class[c] = iou
        if denom > 0:
            ious.append(iou)
    per_class["mIoU"] = float(np.nanmean(ious)) if ious else float("nan")
    per_class["accuracy"] = float((preds == labels).mean())
    return per_class


def print_metrics(metrics: dict, num_classes: int, title: str) -> None:
    """Pretty-print a compute_iou() result: per-class IoU, then mIoU + accuracy."""
    print(f"\n=== {title} ===")
    for c in range(num_classes):
        name = CLASS_NAMES.get(c, f"class{c}")
        print(f"  class {c} {name:<13} IoU = {metrics.get(c, float('nan')):.4f}")
    print(f"  {'mIoU':<21} = {metrics['mIoU']:.4f}")
    print(f"  {'accuracy':<21} = {metrics['accuracy']:.4f}")


def predict_probs(
    model: CellTowerSegmenter,
    xyz: np.ndarray,
    rgb: np.ndarray,
    config: Config,
    device: str,
    tta_rotations: Optional[List[int]] = None,
    amp: bool = True,
) -> np.ndarray:
    """
    Single source of truth for the chunk → forward → overlap-merge → TTA-average
    path. Shared by production inference (infer_tower) and the training-time
    deployment-metric evaluation.

    Global features (the expensive KNN+PCA) are computed ONCE on the unrotated
    cloud. Under a Z-rotation only the surface normals change (nx,ny rotate, nz
    fixed); height_norm, all PCA eigenvalue-ratio features, and RGB are rotation-
    invariant — so each TTA view only rotates xyz + normals analytically rather
    than recomputing features. This is mathematically equivalent to recomputing
    (a rigid rotation preserves KNN neighbour sets) at ~1/Nrot the feature cost.

    Returns (N, K) float32 probabilities averaged over the TTA rotations.
    """
    N = len(xyz)
    angles = tta_rotations if tta_rotations else [0]

    # rotation-invariant features, computed once
    precomputed = compute_global_features(xyz, rgb, config)
    normals_base = precomputed["normals"]

    prob_accum = np.zeros((N, config.num_classes), dtype=np.float32)

    for angle in tqdm(angles, desc="TTA rotations", unit="rot", leave=False):
        if angle % 360 == 0:
            xyz_r = xyz
            pre_r = precomputed
        else:
            theta = np.deg2rad(float(angle))
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            xyz_r = xyz.copy()
            xyz_r[:, 0] = xyz[:, 0] * cos_t - xyz[:, 1] * sin_t
            xyz_r[:, 1] = xyz[:, 0] * sin_t + xyz[:, 1] * cos_t

            normals_r = normals_base.copy()
            normals_r[:, 0] = normals_base[:, 0] * cos_t - normals_base[:, 1] * sin_t
            normals_r[:, 1] = normals_base[:, 0] * sin_t + normals_base[:, 1] * cos_t
            # normals_r[:, 2] (nz) unchanged by a Z-rotation
            pre_r = {**precomputed, "normals": normals_r}

        chunks = slice_into_superchunks(xyz_r, rgb, pre_r, config)

        per_chunk_probs: List[np.ndarray] = []
        per_chunk_indices: List[np.ndarray] = []
        for chunk in chunks:
            feats = torch.from_numpy(chunk.features).to(device)
            xyz_t = feats[:, :3]
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(feats, xyz_t)        # (M, K)
            per_chunk_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            per_chunk_indices.append(chunk.point_indices)

        # overlap-merge this rotation → (N, K) probs
        prob_accum += merge_probs(per_chunk_probs, per_chunk_indices, N, config.num_classes)

    return prob_accum / len(angles)


def infer_tower(
    checkpoint_path: str,
    config: Config,
    output_path: str,
    raw_las_path: Optional[str] = None,
    downsampled_las_path: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """
    Full inference pipeline:
    1. Load + downsample tower
    2. For each TTA rotation: compute features → chunk → forward → per-rotation probs (N_ds, K)
    3. Average probs across TTA rotations
    4. Argmax → labels on downsampled cloud
    5. If the input carries ground-truth classification labels, compute + print the
       same metrics as training (per-class IoU, mIoU, accuracy) on the raw argmax
       (matching the training-time final-TTA eval), plus a post-processed variant.
    6. Write labelled .las (downsampled resolution)

    Returns a dict: {output_path, metrics, metrics_postprocess, preds, gt}. `metrics`
    (and `preds`/`gt`) are None when the input has no ground-truth labels; `preds` is
    the raw argmax so batch callers can pool predictions across towers into one mIoU.
    """
    # load checkpoint and verify feature width
    ckpt = torch.load(checkpoint_path, map_location=device)
    assert ckpt.get("num_features", config.num_features) == config.num_features, (
        f"Checkpoint num_features {ckpt.get('num_features')} != config {config.num_features}. "
        "Feature builder mismatch — re-train or use matching config."
    )

    model = CellTowerSegmenter(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # downsample if needed
    if downsampled_las_path is None or not os.path.exists(downsampled_las_path):
        if raw_las_path is None:
            raise ValueError(
                "No downsampled file available and no raw_las_path given; "
                "provide a raw .las to downsample or an existing downsampled .las."
            )
        print(f"Downsampling {raw_las_path} ...")
        downsampled_las_path = downsample_tower(raw_las_path, config)

    ds_xyz, ds_rgb, ds_las = _load_las(downsampled_las_path)
    N_ds = len(ds_xyz)
    print(f"Downsampled cloud: {N_ds} points")

    # ground truth for metrics (aligned to the downsampled cloud we predict on).
    # Treated as "no labels" when the classification field is absent or entirely 0
    # (an unlabelled cloud defaults to all-Noise), so inference on unlabelled towers
    # simply skips metrics instead of reporting nonsense against a zero field.
    gt_labels = _labels_from_las(ds_las, config)
    has_gt = gt_labels is not None and bool(np.any(gt_labels != 0))

    # TTA: features computed once, rotations averaged inside predict_probs.
    # Disabled by default (config.use_tta=False) → single 0° forward pass.
    if config.use_tta and config.tta_rotations:
        tta_angles = config.tta_rotations
        print(f"TTA enabled, rotations: {tta_angles}")
    else:
        tta_angles = [0]
        print("TTA disabled (single 0° pass)")

    prob_avg = predict_probs(
        model, ds_xyz, ds_rgb, config, device,
        tta_rotations=tta_angles, amp=config.amp,
    )
    ds_labels = np.argmax(prob_avg, axis=1).astype(np.int32)
    raw_labels = ds_labels.copy()   # pre-postprocess argmax (matches training final-TTA eval)

    # optional post-processing, each gated by its own config toggle. Order: systematic
    # geometric-rule fixes first, then generic speckle cleanup last.
    if config.use_ground_reclassify:
        precomputed = compute_global_features(ds_xyz, ds_rgb, config)
        ground_z = float(np.percentile(ds_xyz[:, 2], config.ground_percentile))
        ds_labels = reclassify_ground(
            ds_xyz, precomputed["normals"], ds_labels, ground_z,
            config.ground_height_thresh, config.ground_nz_thresh,
            config.ground_class, config.noise_class,
        )
    if config.use_confusion_resolve:
        ds_labels = resolve_antenna_rru(
            ds_xyz, ds_labels, config.confusion_eps, config.confusion_min_samples,
            config.confusion_dominant_frac, config.confusion_margin,
            config.antenna_class, config.rru_class,
        )
    # DBSCAN speckle cleanup: relabel isolated small clusters to their
    # spatial neighbours' majority class (off unless config.use_dbscan_postprocess)
    if config.use_dbscan_postprocess:
        ds_labels = dbscan_cleanup(
            ds_xyz,
            ds_labels,
            num_classes=config.num_classes,
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples,
            min_cluster_size=config.dbscan_min_cluster_size,
            project_knn=config.project_knn,
            classes=config.dbscan_classes,
        )

    # metrics (only when the input carries ground truth). Score the raw argmax to match
    # the training-time final-TTA eval; if post-processing changed any label, also report
    # the post-processed variant so the delta is visible.
    metrics = metrics_pp = None
    if has_gt:
        metrics = compute_iou(raw_labels, gt_labels, config.num_classes)
        print_metrics(metrics, config.num_classes,
                      "Inference metrics (raw argmax — matches training)")
        if np.any(ds_labels != raw_labels):
            metrics_pp = compute_iou(ds_labels.astype(raw_labels.dtype), gt_labels, config.num_classes)
            print_metrics(metrics_pp, config.num_classes,
                          "Inference metrics (after post-processing)")
    else:
        print("No ground-truth labels in input — skipping metrics.")

    # write output (downsampled resolution)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    write_labelled_las(ds_xyz, ds_rgb, ds_labels, output_path, source_header=ds_las.header)
    print(f"Saved labelled cloud ({N_ds} points): {output_path}")

    result = {
        "preds_raw": raw_labels,
        "preds_final": ds_labels,
        "gt": gt_labels if has_gt else None,
        "metrics_raw": metrics,
        "metrics_final": metrics_pp if metrics_pp is not None else metrics,
    } if has_gt else None
    return output_path, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference on a single tower")
    parser.add_argument("--raw", required=True, help="Path to raw .las/.laz")
    parser.add_argument("--downsampled", default=None, help="Already-downsampled .las (optional)")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pt")
    parser.add_argument("--config", required=True, help="config.yaml path")
    parser.add_argument("--output", required=True, help="Output .las path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--tta", action="store_true",
        help="Enable test-time augmentation (overrides use_tta=false in config)"
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable TTA (overrides use_tta in config)"
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.tta:
        cfg.use_tta = True
    if args.no_tta:
        cfg.use_tta = False

    infer_tower(
        raw_las_path=args.raw,
        checkpoint_path=args.checkpoint,
        config=cfg,
        output_path=args.output,
        downsampled_las_path=args.downsampled,
        device=args.device,
    )
