import numpy as np
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional

from pointcloud_seg.config import Config
from pointcloud_seg.preprocessing.features import build_feature_matrix


@dataclass
class SuperChunk:
    features: np.ndarray            # (M, 17) float32
    point_indices: np.ndarray       # (M,) int — indices into downsampled cloud
    labels: Optional[np.ndarray] = None      # (M,) int, training only
    tower_id: Optional[str] = None
    col_id: int = 0
    chunk_counts: Optional[np.ndarray] = None  # how many chunks each point appears in


def slice_into_superchunks(
    xyz: np.ndarray,           # (N, 3) full downsampled cloud
    rgb: np.ndarray,           # (N, 3) colours [0, 1]
    precomputed: Dict[str, np.ndarray],  # from compute_global_features
    config: Config,
    labels: Optional[np.ndarray] = None,
    tower_id: Optional[str] = None,
) -> List[SuperChunk]:
    N = len(xyz)

    if N <= config.chunk_target_points:
        # single-column degenerate case
        xyz_local = xyz.copy()
        mean_xy = xyz_local[:, :2].mean(axis=0)
        xyz_local[:, 0] -= mean_xy[0]
        xyz_local[:, 1] -= mean_xy[1]
        # de-mean Z too: chunk-relative vertical coordinate, not absolute elevation, so towers at
        # different ground elevations share the same Z range. height_norm (feat 3) still gives [0,1].
        xyz_local[:, 2] -= xyz_local[:, 2].mean()

        feats = build_feature_matrix(
            xyz_local,
            rgb,
            precomputed,
            config,
        )
        indices = np.arange(N, dtype=np.int64)
        chunk = SuperChunk(
            features=feats,
            point_indices=indices,
            labels=labels,
            tower_id=tower_id,
            col_id=0,
            chunk_counts=np.ones(N, dtype=np.int32),
        )
        return [chunk]

    n_cols = max(1, ceil(N / config.chunk_target_points))
    sorted_idx = np.argsort(xyz[:, 0])

    raw_window = int(N / n_cols / max(1 - config.overlap_fraction, 1e-6))
    window_size = max(config.chunk_target_points, min(raw_window, config.chunk_max_points))
    stride = max(1, int(window_size * (1 - config.overlap_fraction)))

    chunks: List[SuperChunk] = []
    membership = np.zeros(N, dtype=np.int32)
    last_w_start = -1

    # Keep emitting windows (advancing by `stride`) until the cloud is fully
    # covered. The loop is NOT bounded by n_cols: with overlap, windows advance
    # by `stride` and window_size is clamped to chunk_max_points, so n_cols
    # windows can fall short of N and silently drop the highest-X points.
    # Termination is guaranteed by the two guards below (w_start past N, or the
    # end-of-cloud shift-back collapsing onto a start we already used).
    i = 0
    while True:
        w_start = i * stride
        if w_start >= N:
            break
        w_end = w_start + window_size
        if w_end > N:
            # shift back so the window ends at N
            w_end = N
            w_start = max(0, N - window_size)
        # prevent duplicate final window being identical to previous:
        # after the shift-back above, w_start can collapse onto the previous
        # window's start, producing an identical chunk. Compare window *start
        # positions* (not point ids) and stop if it repeats.
        if w_start == last_w_start:
            break
        last_w_start = w_start

        indices = sorted_idx[w_start:w_end]

        membership[indices] += 1

        xyz_chunk = xyz[indices]
        mean_xy = xyz_chunk[:, :2].mean(axis=0)
        xyz_local = xyz_chunk.copy()
        xyz_local[:, 0] -= mean_xy[0]
        xyz_local[:, 1] -= mean_xy[1]
        # de-mean Z (see single-column branch): chunk-relative vertical coordinate, not absolute.
        xyz_local[:, 2] -= xyz_chunk[:, 2].mean()

        pre_slice = {k: v[indices] for k, v in precomputed.items()}
        feats = build_feature_matrix(xyz_local, rgb[indices], pre_slice, config)

        chunk_labels = labels[indices] if labels is not None else None

        chunks.append(SuperChunk(
            features=feats,
            point_indices=indices,
            labels=chunk_labels,
            tower_id=tower_id,
            col_id=i,
        ))

        i += 1

    # fill chunk_counts for each chunk
    for chunk in chunks:
        chunk.chunk_counts = membership[chunk.point_indices]

    return chunks
