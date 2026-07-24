"""
Stage 4 — Inference on one or more raw towers.

Runs the full inference pipeline:
  1. Poisson-disk downsample (skipped if --downsampled is given)
  2. For each TTA rotation: compute features → chunk → model forward → probabilities
  3. Average TTA probabilities, argmax → labels on downsampled cloud
  4. Write labelled .las at downsampled resolution to output path

Single tower:
    python scripts/04_infer.py \
        --config  pointcloud_seg/sample_configs/config.yaml \
        --checkpoint outputs/run_01/best_model.pt \
        --raw     data/raw/tower_new.las \
        --output  results/tower_new_labelled.las

With pre-existing downsampled file (no --raw needed):
    python scripts/04_infer.py \
        --config     my_config.yaml \
        --checkpoint best_model.pt \
        --downsampled ds.las \
        --output     labelled.las \
        --no-tta

Batch (one output per input, names match input):
    python scripts/04_infer.py \
        --config     my_config.yaml \
        --checkpoint best_model.pt \
        --batch-dir  data/raw \
        --output-dir results/

Batch from pre-downsampled files (no raw needed):
    python scripts/04_infer.py \
        --config         my_config.yaml \
        --checkpoint     best_model.pt \
        --downsampled-dir data/downsampled \
        --output-dir     results/
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pointcloud_seg.config import Config
from pointcloud_seg.infer import infer_tower, compute_iou, print_metrics


def find_las_files(directory: str):
    d = Path(directory)
    return sorted([str(p) for p in d.glob("*.las")] + [str(p) for p in d.glob("*.laz")])


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Inference on cell tower point clouds")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable TTA (only 0° rotation, faster)")

    # single tower
    single = parser.add_argument_group("single tower")
    single.add_argument("--raw", default=None, help="Raw .las path")
    single.add_argument("--downsampled", default=None, help="Pre-downsampled .las (optional)")
    single.add_argument("--output", default=None, help="Output labelled .las path")

    # batch
    batch = parser.add_argument_group("batch mode")
    batch.add_argument("--batch-dir", default=None, help="Directory of raw .las files")
    batch.add_argument("--downsampled-dir", default=None,
                       help="Directory of pre-downsampled .las files (raw not needed)")
    batch.add_argument("--output-dir", default=None, help="Directory to write labelled outputs")

    args = parser.parse_args()

    if not args.raw and not args.downsampled and not args.batch_dir and not args.downsampled_dir:
        parser.error(
            "Provide --raw (or --downsampled) for single tower, "
            "or --batch-dir (or --downsampled-dir) for batch mode"
        )

    cfg = Config.from_yaml(args.config)
    if args.no_tta:
        cfg.tta_rotations = [0]
        print("TTA disabled (single 0° rotation)")
    else:
        print(f"TTA rotations: {cfg.tta_rotations}")

    if args.raw or args.downsampled:
        # single tower
        src = args.raw or args.downsampled
        output = args.output or str(Path(src).with_name(Path(src).stem + "_labelled.las"))
        print(f"\nInference: {src} → {output}")
        infer_tower(
            raw_las_path=args.raw,
            checkpoint_path=args.checkpoint,
            config=cfg,
            output_path=output,
            downsampled_las_path=args.downsampled,
            device=args.device,
        )
        print(f"Done → {output}")

    else:
        # batch mode — source is a downsampled dir (raw not needed) or a raw dir
        if args.downsampled_dir:
            src_paths = find_las_files(args.downsampled_dir)
            src_dir, is_downsampled = args.downsampled_dir, True
        else:
            src_paths = find_las_files(args.batch_dir)
            src_dir, is_downsampled = args.batch_dir, False

        if not src_paths:
            print(f"No .las files found in {src_dir}")
            sys.exit(1)

        out_dir = Path(args.output_dir or src_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        succeeded, failed = [], []
        pooled_preds, pooled_gt, scored_towers = [], [], 0
        for i, src in enumerate(src_paths):
            out = str(out_dir / (Path(src).stem + "_labelled.las"))
            print(f"\n[{i+1}/{len(src_paths)}] {Path(src).name} → {Path(out).name}")
            try:
                _out_path, res = infer_tower(
                    raw_las_path=None if is_downsampled else src,
                    downsampled_las_path=src if is_downsampled else None,
                    checkpoint_path=args.checkpoint,
                    config=cfg,
                    output_path=out,
                    device=args.device,
                )
                succeeded.append(out)
                # collect raw argmax + GT for a pooled batch mIoU (mirrors training's
                # end-of-run TTA eval, which pools predictions across all val towers).
                if res is not None and res["gt"] is not None:
                    pooled_preds.append(res["preds_raw"])
                    pooled_gt.append(res["gt"])
                    scored_towers += 1
            except Exception as e:
                print(f"  [ERROR] {e}")
                failed.append(src)

        print(f"\n=== Batch inference complete ===")
        print(f"  Succeeded : {len(succeeded)}")
        print(f"  Failed    : {len(failed)}")
        if failed:
            for p in failed:
                print(f"    FAILED: {p}")

        # pooled metrics across every labelled tower (raw argmax → one mIoU)
        if pooled_preds:
            pooled = compute_iou(
                np.concatenate(pooled_preds), np.concatenate(pooled_gt), cfg.num_classes
            )
            print_metrics(
                pooled, cfg.num_classes,
                f"Batch pooled metrics — {scored_towers} labelled tower(s), raw argmax",
            )
        else:
            print("  (no labelled towers — pooled metrics skipped)")

        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
