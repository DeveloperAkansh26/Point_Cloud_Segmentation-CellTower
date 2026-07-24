import numpy as np
from scipy.spatial import cKDTree
from typing import Dict

from pointcloud_seg.config import Config

# 15k (was 50k) so the (M, knn_xlarge, 3) neighbor gather for the object-scale PCA stays within
# memory — knn_xlarge=1300 is ~15x the k=96 neighborhood.
_BATCH_SIZE = 15_000
_EPS = 1e-6


def compute_global_features(xyz: np.ndarray, rgb: np.ndarray, config: Config) -> Dict[str, np.ndarray]:
    """
    Build ONE KD-tree and compute all per-point features in batches.
    Returns dict with keys: normals, linearity_s, planarity_s, scattering_s,
    planarity_l, scattering_l, planarity_diff, curvature, height_norm,
    scattering_o, planarity_o, linearity_o, vert_extent_o.

    The *_o (object-scale) features use a much larger neighborhood (knn_xlarge) so a point
    sees its whole object: a boxy RRU shows high 3-D scatter / small vertical extent, while a
    flat panel antenna stays planar/linear with a large vertical extent — the signal the small
    knn_small/knn_large scales cannot separate.
    """
    N = len(xyz)
    tree = cKDTree(xyz)

    normals = np.empty((N, 3), dtype=np.float32)
    linearity_s = np.empty(N, dtype=np.float32)
    planarity_s = np.empty(N, dtype=np.float32)
    scattering_s = np.empty(N, dtype=np.float32)
    planarity_l = np.empty(N, dtype=np.float32)
    scattering_l = np.empty(N, dtype=np.float32)
    curvature = np.empty(N, dtype=np.float32)
    scattering_o = np.empty(N, dtype=np.float32)
    planarity_o = np.empty(N, dtype=np.float32)
    linearity_o = np.empty(N, dtype=np.float32)
    vert_extent_o = np.empty(N, dtype=np.float32)

    k_s = config.knn_small
    k_l = config.knn_large
    # object scale is capped at N-1 so small clouds don't over-query the KD-tree
    k_o = min(config.knn_xlarge, max(N - 1, 1))

    for start in range(0, N, _BATCH_SIZE):
        end = min(start + _BATCH_SIZE, N)
        batch_xyz = xyz[start:end]
        M = end - start

        # query knn_small + 1 to exclude self, then slice
        _, idx_s = tree.query(batch_xyz, k=k_s + 1, workers=-1)
        idx_s = idx_s[:, 1:]  # (M, k_s) exclude self

        _, idx_l = tree.query(batch_xyz, k=k_l + 1, workers=-1)
        idx_l = idx_l[:, 1:]  # (M, k_l) exclude self

        _, idx_o = tree.query(batch_xyz, k=k_o + 1, workers=-1)
        idx_o = idx_o[:, 1:]  # (M, k_o) exclude self

        # small-scale PCA
        nbrs_s = xyz[idx_s]  # (M, k_s, 3)
        diff_s = nbrs_s - batch_xyz[:, None, :]  # (M, k_s, 3)
        cov_s = np.matmul(diff_s.transpose(0, 2, 1), diff_s) / k_s  # (M, 3, 3)
        evals_s, evecs_s = np.linalg.eigh(cov_s)
        # eigh returns ascending order; reverse to get l1 >= l2 >= l3
        l1_s = evals_s[:, 2]
        l2_s = evals_s[:, 1]
        l3_s = evals_s[:, 0]
        # normal = eigenvector of smallest eigenvalue
        nrm = evecs_s[:, :, 0]  # (M, 3)
        # flip normals with negative Z to point into upper hemisphere
        flip_mask = nrm[:, 2] < 0
        nrm[flip_mask] = -nrm[flip_mask]

        normals[start:end] = nrm.astype(np.float32)

        l1_s_safe = np.maximum(l1_s, _EPS)
        linearity_s[start:end] = ((l1_s - l2_s) / l1_s_safe).astype(np.float32)
        planarity_s[start:end] = ((l2_s - l3_s) / l1_s_safe).astype(np.float32)
        scattering_s[start:end] = (l3_s / l1_s_safe).astype(np.float32)

        # real curvature (NOT duplicate of scattering_l)
        curvature[start:end] = (l3_s / (l1_s + l2_s + l3_s + _EPS)).astype(np.float32)

        # large-scale PCA
        nbrs_l = xyz[idx_l]  # (M, k_l, 3)
        diff_l = nbrs_l - batch_xyz[:, None, :]
        cov_l = np.matmul(diff_l.transpose(0, 2, 1), diff_l) / k_l
        evals_l, _ = np.linalg.eigh(cov_l)
        l1_l = evals_l[:, 2]
        l2_l = evals_l[:, 1]
        l3_l = evals_l[:, 0]

        l1_l_safe = np.maximum(l1_l, _EPS)
        planarity_l[start:end] = ((l2_l - l3_l) / l1_l_safe).astype(np.float32)
        scattering_l[start:end] = (l3_l / l1_l_safe).astype(np.float32)

        # object-scale PCA — separates boxy RRU (volumetric) from flat antenna (planar/linear)
        nbrs_o = xyz[idx_o]  # (M, k_o, 3)
        diff_o = nbrs_o - batch_xyz[:, None, :]
        cov_o = np.matmul(diff_o.transpose(0, 2, 1), diff_o) / k_o
        evals_o, _ = np.linalg.eigh(cov_o)
        l1_o = evals_o[:, 2]
        l2_o = evals_o[:, 1]
        l3_o = evals_o[:, 0]

        l1_o_safe = np.maximum(l1_o, _EPS)
        scattering_o[start:end] = (l3_o / l1_o_safe).astype(np.float32)
        planarity_o[start:end] = ((l2_o - l3_o) / l1_o_safe).astype(np.float32)
        linearity_o[start:end] = ((l1_o - l2_o) / l1_o_safe).astype(np.float32)
        # vertical span (metres) of the object-scale neighborhood, including self: antennas are
        # tall, RRUs compact. Absolute length (NOT a ratio) — that is the discriminative quantity.
        z_nbr = nbrs_o[:, :, 2]  # (M, k_o)
        z_self = batch_xyz[:, 2]
        z_hi = np.maximum(z_nbr.max(axis=1), z_self)
        z_lo = np.minimum(z_nbr.min(axis=1), z_self)
        vert_extent_o[start:end] = (z_hi - z_lo).astype(np.float32)

    planarity_diff = (planarity_s - planarity_l).astype(np.float32)

    z = xyz[:, 2]
    z_min = z.min()
    z_max = z.max()
    height_norm = ((z - z_min) / (z_max - z_min + _EPS)).astype(np.float32)

    return {
        "normals": normals,
        "linearity_s": linearity_s,
        "planarity_s": planarity_s,
        "scattering_s": scattering_s,
        "planarity_l": planarity_l,
        "scattering_l": scattering_l,
        "planarity_diff": planarity_diff,
        "curvature": curvature,
        "height_norm": height_norm,
        "scattering_o": scattering_o,
        "planarity_o": planarity_o,
        "linearity_o": linearity_o,
        "vert_extent_o": vert_extent_o,
    }


def build_feature_matrix(
    xyz_local: np.ndarray,  # (M, 3) already normalized per-chunk
    rgb: np.ndarray,        # (M, 3) values in [0, 1]
    precomputed: Dict[str, np.ndarray],
    config: Config,
) -> np.ndarray:
    """
    Assemble the 21-D feature vector per point.
    precomputed must already be sliced to the chunk's point indices.
    """
    height_norm = precomputed["height_norm"].reshape(-1, 1)
    feats = np.column_stack([
        xyz_local,                          # 0:3
        height_norm,                        # 3
        precomputed["normals"],             # 4:7
        precomputed["linearity_s"],         # 7
        precomputed["planarity_s"],         # 8
        precomputed["scattering_s"],        # 9
        precomputed["planarity_l"],         # 10
        precomputed["scattering_l"],        # 11
        precomputed["planarity_diff"],      # 12
        precomputed["curvature"],           # 13
        precomputed["scattering_o"],        # 14  object-scale (antenna vs RRU)
        precomputed["planarity_o"],         # 15
        precomputed["linearity_o"],         # 16
        precomputed["vert_extent_o"],       # 17
        rgb,                               # 18:21
    ])
    assert feats.shape[1] == config.num_features, (
        f"feature width {feats.shape[1]} != {config.num_features}"
    )
    return feats.astype(np.float32)
