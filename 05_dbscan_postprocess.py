"""
Stage 5 — Standalone DBSCAN speckle cleanup on an already-labelled cloud.

Runs ONLY the DBSCAN post-processing step (no model, no TTA). The input is a
labelled .las — typically the output of stage 4 inference (04_infer.py), whose
`classification` field already holds the model/TTA-predicted labels. Isolated
small clusters (and DBSCAN-noise points) are relabelled to the majority class of
their nearest retained neighbours, and a new labelled .las is written with the
source georeferencing preserved.

This is decoupled from inference so you can re-run cleanup and sweep the DBSCAN
parameters cheaply without re-running the model.

Single file (params from config):
    python 05_dbscan_postprocess.py \
        --config pointcloud_seg/sample_configs/config.yaml \
        --input  results/tower_new_labelled.las \
        --output results/tower_new_labelled_dbscan.las

Single file with explicit params (no config needed):
    python 05_dbscan_postprocess.py \
        --input  labelled.las \
        --output cleaned.las \
        --eps 0.10 --min-samples 10 --min-cluster-size 50

Batch (one output per input, "_dbscan" suffix):
    python 05_dbscan_postprocess.py \
        --config    my_config.yaml \
        --batch-dir results/ \
        --output-dir results_dbscan/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from pointcloud_seg.config import Config
from pointcloud_seg.infer import _load_las
from pointcloud_seg.preprocessing.features import compute_global_features
from pointcloud_seg.preprocessing.postprocess import (
    dbscan_cleanup,
    reclassify_ground,
    resolve_antenna_rru,
)
from pointcloud_seg.preprocessing.projection import write_labelled_las


def find_las_files(directory: str):
    d = Path(directory)
    return sorted([str(p) for p in d.glob("*.las")] + [str(p) for p in d.glob("*.laz")])


def process_one(input_path: str, output_path: str, cfg: Config, params: dict):
    """Load a labelled .las, run the enabled post-processing steps, write the result.

    Applied in the same order as inference: ground reclassify → antenna/RRU resolve →
    DBSCAN speckle cleanup. The two rule fixes are gated by their config toggles; DBSCAN
    runs when its toggle is on or when any DBSCAN parameter was overridden on the CLI.
    """
    xyz, rgb, las = _load_las(input_path)
    labels = np.asarray(las.classification, dtype=np.int32)
    print(f"Loaded {len(xyz)} points; label histogram: "
          f"{np.bincount(labels, minlength=params['num_classes']).tolist()}")

    if cfg.use_ground_reclassify:
        normals = compute_global_features(xyz, rgb, cfg)["normals"]
        ground_z = float(np.percentile(xyz[:, 2], cfg.ground_percentile))
        labels = reclassify_ground(
            xyz, normals, labels, ground_z,
            cfg.ground_height_thresh, cfg.ground_nz_thresh,
            cfg.ground_class, cfg.noise_class,
        )

    if cfg.use_confusion_resolve:
        labels = resolve_antenna_rru(
            xyz, labels, cfg.confusion_eps, cfg.confusion_min_samples,
            cfg.confusion_dominant_frac, cfg.confusion_margin,
            cfg.antenna_class, cfg.rru_class,
        )

    if params["dbscan_enabled"]:
        labels = dbscan_cleanup(
            xyz,
            labels,
            num_classes=params["num_classes"],
            eps=params["eps"],
            min_samples=params["min_samples"],
            min_cluster_size=params["min_cluster_size"],
            project_knn=params["project_knn"],
            classes=params["classes"],
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_labelled_las(xyz, rgb, labels, output_path, source_header=las.header)
    print(f"Saved cleaned cloud → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: standalone DBSCAN speckle cleanup on a labelled .las"
    )
    parser.add_argument("--config", default=None,
                        help="config.yaml to read DBSCAN defaults from (optional)")

    # DBSCAN parameter overrides (win over config; required if no --config)
    parser.add_argument("--eps", type=float, default=None, help="DBSCAN eps (metres)")
    parser.add_argument("--min-samples", type=int, default=None, help="DBSCAN core min_samples")
    parser.add_argument("--min-cluster-size", type=int, default=None,
                        help="clusters below this are treated as speckle")
    parser.add_argument("--project-knn", type=int, default=None,
                        help="neighbours used to relabel speckle")
    parser.add_argument("--num-classes", type=int, default=None, help="number of classes")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="restrict cleanup to these class ids (default: all)")

    # single file
    single = parser.add_argument_group("single file")
    single.add_argument("--input", default=None, help="Labelled .las to clean")
    single.add_argument("--output", default=None, help="Output .las path")

    # batch
    batch = parser.add_argument_group("batch mode")
    batch.add_argument("--batch-dir", default=None, help="Directory of labelled .las files")
    batch.add_argument("--output-dir", default=None, help="Directory to write cleaned outputs")

    args = parser.parse_args()

    if not args.input and not args.batch_dir:
        parser.error("Provide --input for a single file, or --batch-dir for batch mode")

    cfg = Config.from_yaml(args.config) if args.config else Config()
    # DBSCAN runs if enabled in config OR any DBSCAN parameter was overridden on the CLI.
    dbscan_cli = any(v is not None for v in
                     (args.eps, args.min_samples, args.min_cluster_size, args.classes))
    # CLI overrides win over config values
    params = {
        "dbscan_enabled": cfg.use_dbscan_postprocess or dbscan_cli,
        "eps": args.eps if args.eps is not None else cfg.dbscan_eps,
        "min_samples": args.min_samples if args.min_samples is not None else cfg.dbscan_min_samples,
        "min_cluster_size": (args.min_cluster_size if args.min_cluster_size is not None
                             else cfg.dbscan_min_cluster_size),
        "project_knn": args.project_knn if args.project_knn is not None else cfg.project_knn,
        "num_classes": args.num_classes if args.num_classes is not None else cfg.num_classes,
        "classes": args.classes if args.classes is not None else cfg.dbscan_classes,
    }
    print(f"Steps enabled — ground_reclassify: {cfg.use_ground_reclassify}, "
          f"confusion_resolve: {cfg.use_confusion_resolve}, dbscan: {params['dbscan_enabled']}")
    if params["dbscan_enabled"]:
        print(f"DBSCAN params: eps={params['eps']} min_samples={params['min_samples']} "
              f"min_cluster_size={params['min_cluster_size']} project_knn={params['project_knn']} "
              f"classes={params['classes'] or 'all'}")

    if args.input:
        output = args.output or str(
            Path(args.input).with_name(Path(args.input).stem + "_dbscan.las")
        )
        print(f"\nDBSCAN cleanup: {args.input} → {output}")
        process_one(args.input, output, cfg, params)
        print(f"Done → {output}")
        return

    # batch mode
    src_paths = find_las_files(args.batch_dir)
    if not src_paths:
        print(f"No .las files found in {args.batch_dir}")
        sys.exit(1)

    out_dir = Path(args.output_dir or args.batch_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    succeeded, failed = [], []
    for i, src in enumerate(src_paths):
        out = str(out_dir / (Path(src).stem + "_dbscan.las"))
        print(f"\n[{i+1}/{len(src_paths)}] {Path(src).name} → {Path(out).name}")
        try:
            process_one(src, out, cfg, params)
            succeeded.append(out)
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed.append(src)

    print(f"\n=== DBSCAN cleanup complete ===")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        for p in failed:
            print(f"    FAILED: {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
