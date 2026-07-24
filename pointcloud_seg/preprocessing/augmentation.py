import numpy as np
from typing import Dict, Tuple

from pointcloud_seg.config import Config
from pointcloud_seg.preprocessing.features import compute_global_features


def augment_tower(
    xyz: np.ndarray,      # (N, 3)
    rgb: np.ndarray,      # (N, 3) [0, 1]
    normals: np.ndarray,  # (N, 3) — will be updated analytically for rotation/scale
    config: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply augmentation to the WHOLE tower.
    Returns augmented (xyz, rgb, normals).
    """
    xyz = xyz.copy()
    rgb = rgb.copy()
    normals = normals.copy()

    # 1. Z-rotation
    if config.aug_rotate_z:
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_new = xyz[:, 0] * cos_t - xyz[:, 1] * sin_t
        y_new = xyz[:, 0] * sin_t + xyz[:, 1] * cos_t
        xyz[:, 0] = x_new
        xyz[:, 1] = y_new

        # rotate normals XY by same theta; nz unchanged
        nx_new = normals[:, 0] * cos_t - normals[:, 1] * sin_t
        ny_new = normals[:, 0] * sin_t + normals[:, 1] * cos_t
        normals[:, 0] = nx_new
        normals[:, 1] = ny_new

    # re-normalize normals to unit length after rotation
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    # 2. Uniform scale
    s_min, s_max = config.aug_scale_range
    s = np.random.uniform(s_min, s_max)
    xyz *= s
    # normals are direction vectors — scale does not affect direction

    # 3. Jitter
    xyz += np.random.normal(0, config.aug_jitter_sigma_m, size=xyz.shape)

    # 4. Colour jitter
    if config.aug_colour_jitter:
        brightness = np.random.uniform(0.8, 1.2)
        contrast = np.random.uniform(0.8, 1.2)
        rgb = rgb * brightness * contrast
        hue_noise = np.random.normal(0, 0.02, size=rgb.shape)
        rgb = rgb + hue_noise
        rgb = np.clip(rgb, 0.0, 1.0)

    return xyz, rgb, normals


def augment_and_recompute(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Augment the full tower and recompute all global features on augmented xyz.
    Returns augmented (xyz, rgb, labels, precomputed).
    """
    # compute normals before augmenting so we can pass them into augment_tower
    precomputed_orig = compute_global_features(xyz, rgb, config)
    normals_orig = precomputed_orig["normals"]

    xyz_aug, rgb_aug, _normals_aug = augment_tower(xyz, rgb, normals_orig, config)

    # recompute ALL global features on augmented xyz for correctness
    precomputed_aug = compute_global_features(xyz_aug, rgb_aug, config)

    return xyz_aug, rgb_aug, labels, precomputed_aug


def augment_features(feats: np.ndarray, config: Config, rotation_deg=None) -> np.ndarray:
    """
    Augmentation applied analytically to the 17-feature vector.

    Equivalent to recomputing features on a Z-rotated/scaled cloud because Z-rotation
    preserves KNN neighbour sets, and all PCA eigenvalue-ratio features are
    rotation- and scale-invariant.

    rotation_deg: if None, draw a random angle in [0, 2π); if a number, use that
                  fixed angle (degrees) — used for offline augmentation at known angles.

    Feature layout:
      [0:3]   local XYZ      [3]     height_norm    [4:7]   normals
      [7:10]  small PCA      [10:12] large PCA       [12:14] diff + curvature
      [14:17] RGB
    """
    feats = feats.copy()

    # 1. Z-rotation: only local XY and normal XY rotate (nz unchanged)
    if config.aug_rotate_z:
        theta = np.deg2rad(rotation_deg) if rotation_deg is not None \
                else np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_new = feats[:, 0] * cos_t - feats[:, 1] * sin_t
        y_new = feats[:, 0] * sin_t + feats[:, 1] * cos_t
        feats[:, 0] = x_new
        feats[:, 1] = y_new
        nx_new = feats[:, 4] * cos_t - feats[:, 5] * sin_t
        ny_new = feats[:, 4] * sin_t + feats[:, 5] * cos_t
        feats[:, 4] = nx_new
        feats[:, 5] = ny_new

    # 2. Uniform scale on local XYZ (normals are unit vectors, unaffected;
    #    height_norm and eigenvalue-ratio features are scale-invariant)
    s_min, s_max = config.aug_scale_range
    feats[:, :3] *= np.random.uniform(s_min, s_max)

    # 3. XYZ jitter
    if config.aug_jitter_sigma_m > 0:
        feats[:, :3] += np.random.normal(0, config.aug_jitter_sigma_m, (len(feats), 3))

    # 4. Colour jitter on RGB [14:17]
    if config.aug_colour_jitter:
        brightness = np.random.uniform(0.8, 1.2)
        contrast = np.random.uniform(0.8, 1.2)
        feats[:, 14:17] = np.clip(
            feats[:, 14:17] * brightness * contrast
            + np.random.normal(0, 0.02, (len(feats), 3)),
            0.0, 1.0,
        )

    return feats.astype(np.float32)


def class_weighted_dropout(
    features: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-frequency point dropout: rare classes survive with higher probability,
    abundant classes are randomly thinned.
    Extra per-class multipliers from config.dropout_class_multipliers further suppress
    over-represented classes (multiplier < 1 = more dropout).
    """
    num_classes = config.num_classes
    class_counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1)
    freq = class_counts / class_counts.sum()
    surv_prob = 1.0 / np.log(1.02 + freq)
    surv_prob = surv_prob / surv_prob.max()
    for cls, mult in getattr(config, "dropout_class_multipliers", {}).items():
        surv_prob[int(cls)] *= float(mult)
    per_point_prob = surv_prob[labels]
    mask = np.random.random(len(labels)) < per_point_prob
    if mask.sum() == 0:
        return features, labels
    return features[mask], labels[mask]
