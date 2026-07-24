"""
Standalone — DBSCAN clustering of a LAS point cloud.

Reads a LAS/LAZ file and runs DBSCAN on the XYZ coordinates (optionally plus RGB
via --use-rgb and/or the one-hot model prediction via --use-prediction), then
writes a new LAS that carries TWO scalar fields, both colourable in CloudCompare:

  * `classification` (uint8)  — the model prediction, preserved as-is from the
    input. This is the field the segmentation pipeline reads/writes
    (see infer.py / dataset.py), so the output stays pipeline compatible.
  * a cluster extra dimension (uint32, name set by --cluster-field, default
    "cluster") — the DBSCAN cluster id. Noise (DBSCAN label -1) is written as 0,
    real clusters as 1..k.

Georeferencing (offsets, scales, CRS VLRs) and RGB are preserved from the source
so the output lands at the correct projected coordinates.

Usage:
    python 05_dbscan_cluster.py --input cloud.las --output clustered.las
    python 05_dbscan_cluster.py -i cloud.las -o out.las --eps 0.5 --min-samples 10
    python 05_dbscan_cluster.py -i cloud.las -o out.las --cluster-field label --workers -1
    python 05_dbscan_cluster.py -i cloud.las -o out.las --use-prediction --prediction-weight 5
"""

import argparse
import sys

import numpy as np
import laspy
from sklearn.cluster import DBSCAN


def load_las(path: str):
    """
    Return (xyz float64, rgb float32 in [0,1] or None, prediction uint8, source laspy.LasData).

    `prediction` is the source `classification` field (the model prediction in the
    seg pipeline); it is all-zeros if the input has no classification field.
    """
    las = laspy.read(path)
    xyz = np.stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1
    ).astype(np.float64)

    if all(hasattr(las, c) for c in ("red", "green", "blue")):
        rgb = np.stack(
            [
                np.asarray(las.red, dtype=np.float32),
                np.asarray(las.green, dtype=np.float32),
                np.asarray(las.blue, dtype=np.float32),
            ],
            axis=1,
        ) / 65535.0
    else:
        rgb = None

    if hasattr(las, "classification"):
        prediction = np.asarray(las.classification, dtype=np.uint8)
    else:
        prediction = np.zeros(len(xyz), dtype=np.uint8)

    return xyz, rgb, prediction, las


def write_clustered_las(xyz, rgb, prediction, clusters, cluster_field, output_path, source_header):
    """
    Write a LAS carrying both fields, preserving the source georeferencing
    (offsets/scales/CRS VLRs) so coordinates stay correct and can't overflow int32:

      * `classification` (uint8)   <- `prediction` (the preserved model prediction)
      * `cluster_field`  (uint32)  <- `clusters`   (the DBSCAN cluster ids)
    """
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = source_header.offsets
    header.scales = source_header.scales

    # pyproj-free CRS preservation: copy the WKT / GeoTIFF-key CRS VLRs directly.
    CRS_IDS = (2111, 2112, 34735, 34736, 34737)
    copied_wkt = False
    for vlr in source_header.vlrs:
        if getattr(vlr, "record_id", None) in CRS_IDS:
            header.vlrs.append(vlr)
            copied_wkt |= vlr.record_id in (2111, 2112)
    if copied_wkt:
        header.global_encoding.wkt = 1  # required for point formats >= 6

    las = laspy.LasData(header=header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    if rgb is not None:
        rgb_16 = (np.clip(rgb, 0.0, 1.0) * 65535).astype(np.uint16)
        las.red = rgb_16[:, 0]
        las.green = rgb_16[:, 1]
        las.blue = rgb_16[:, 2]

    # preserved model prediction stays in the standard classification field
    las.classification = prediction.astype(np.uint8)

    # DBSCAN cluster ids go in a separate uint32 extra dimension (no 255-cluster cap)
    las.add_extra_dim(laspy.ExtraBytesParams(name=cluster_field, type=np.uint32))
    las[cluster_field] = clusters.astype(np.uint32)

    las.write(output_path)


def main():
    p = argparse.ArgumentParser(
        description="DBSCAN-cluster a LAS point cloud and store cluster ids as classification labels."
    )
    p.add_argument("-i", "--input", required=True, help="Input .las/.laz path")
    p.add_argument("-o", "--output", required=True, help="Output .las path")
    p.add_argument(
        "--eps",
        type=float,
        default=2,
        help="DBSCAN neighbourhood radius, in the cloud's coordinate units (default: 0.5)",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=1000,
        help="Min points in a neighbourhood to form a core point (default: 10)",
    )
    p.add_argument(
        "--cluster-field",
        default="cluster",
        help="Name of the extra scalar dimension that holds DBSCAN cluster ids "
        "(uint32). The model prediction is always kept in `classification`. (default: cluster)",
    )
    p.add_argument(
        "--use-rgb",
        action="store_true",
        help="Include normalized RGB as extra clustering features alongside XYZ.",
    )
    p.add_argument(
        "--use-prediction",
        action="store_true",
        help="Include the model prediction (source `classification`) as a clustering "
        "feature, one-hot encoded and scaled by --prediction-weight.",
    )
    p.add_argument(
        "--prediction-weight",
        type=float,
        default=5,
        help="Weight on the one-hot prediction feature (in coordinate units). Two points "
        "of different predicted classes differ by sqrt(2)*weight in feature space, so "
        "weight <~ eps is a soft bias and weight >> eps forces per-class separation. "
        "(default: 1.0)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Parallel jobs for the neighbour search; -1 uses all cores (default: -1)",
    )
    args = p.parse_args()

    print(f"Reading {args.input} ...")
    xyz, rgb, prediction, src = load_las(args.input)
    n = len(xyz)
    print(f"  {n:,} points")
    if n == 0:
        sys.exit("Input cloud is empty; nothing to cluster.")

    features = xyz
    feature_desc = "XYZ"
    if args.use_rgb:
        if rgb is None:
            print("  --use-rgb requested but cloud has no RGB; falling back to XYZ only.")
        else:
            features = np.hstack([features, rgb])
            feature_desc += " + RGB"

    if args.use_prediction:
        classes = np.unique(prediction)
        if classes.size < 2:
            print(
                f"  --use-prediction requested but the prediction is constant "
                f"(class {int(classes[0])}); it adds no separation, ignoring."
            )
        else:
            # one-hot so class ids stay categorical (no false ordinal distances),
            # scaled by the weight to control its pull relative to XYZ/RGB and eps.
            onehot = (prediction[:, None] == classes[None, :]).astype(np.float32)
            features = np.hstack([features, onehot * args.prediction_weight])
            feature_desc += f" + prediction (weight={args.prediction_weight}, {classes.size} classes)"

    if feature_desc != "XYZ":
        print(f"  clustering on {feature_desc}")

    print(f"Running DBSCAN (eps={args.eps}, min_samples={args.min_samples}) ...")
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=args.workers)
    raw = db.fit_predict(features)

    # DBSCAN uses -1 for noise. Shift so noise -> 0 and clusters -> 1..k, keeping the
    # cluster ids non-negative for the uint32 extra dimension.
    n_clusters = len(set(raw)) - (1 if -1 in raw else 0)
    n_noise = int(np.sum(raw == -1))
    clusters = np.where(raw == -1, 0, raw + 1)

    print(f"  {n_clusters} clusters, {n_noise:,} noise points (-> {args.cluster_field} 0)")

    print(f"Writing {args.output} ...")
    write_clustered_las(xyz, rgb, prediction, clusters, args.cluster_field, args.output, src.header)
    print(
        f"Done. In CloudCompare, colour by the '{args.cluster_field}' scalar field for the "
        "DBSCAN clusters, or 'Classification' for the preserved model prediction."
    )


if __name__ == "__main__":
    main()
