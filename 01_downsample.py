"""
Stage 1 — Batch Poisson-disk downsampling via PDAL.

Discovers all .las/.laz files in config.raw_dir and runs parallel PDAL
Poisson-disk sampling. Outputs go to config.downsampled_dir.

Usage:
    python scripts/01_downsample.py --config pointcloud_seg/sample_configs/config.yaml
    python scripts/01_downsample.py --config my_config.yaml --raw-dir /data/raw
    python scripts/01_downsample.py --config my_config.yaml --files a.las b.las
"""

import argparse
import sys
from pathlib import Path

# allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointcloud_seg.config import Config
from pointcloud_seg.preprocessing.downsample import batch_downsample


def find_las_files(directory: str):
    d = Path(directory)
    files = sorted(list(d.glob("*.las")) + list(d.glob("*.laz")))
    return [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Batch downsample raw LAS files via PDAL")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--raw-dir", default=None, help="Override config.raw_dir")
    parser.add_argument("--output-dir", default=None, help="Override config.downsampled_dir")
    parser.add_argument("--files", nargs="+", default=None, help="Explicit list of .las/.laz paths")
    parser.add_argument("--workers", type=int, default=None, help="Override config.pdal_workers")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    if args.raw_dir:
        cfg.raw_dir = args.raw_dir
    if args.output_dir:
        cfg.downsampled_dir = args.output_dir
    if args.workers:
        cfg.pdal_workers = args.workers

    if args.files:
        raw_paths = args.files
    else:
        raw_paths = find_las_files(cfg.raw_dir)

    if not raw_paths:
        print(f"No .las/.laz files found in {cfg.raw_dir}")
        sys.exit(1)

    print(f"Found {len(raw_paths)} file(s) to downsample")
    print(f"  downsample_cm  : {cfg.downsample_cm}")
    print(f"  output dir     : {cfg.downsampled_dir}")
    print(f"  parallel workers: {cfg.pdal_workers}")
    print()

    succeeded, failed = batch_downsample(raw_paths, cfg)

    print(f"\n=== Stage 1 complete ===")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        sys.exit(1)

    print("\nDownsampled files:")
    for p in succeeded:
        print(f"  {p}")


if __name__ == "__main__":
    main()
