import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def dbscan_cleanup(
    xyz: np.ndarray,          # (N, 3) point coordinates
    labels: np.ndarray,       # (N,) int predicted labels (argmax of probs)
    num_classes: int,
    eps: float,
    min_samples: int,
    min_cluster_size: int,
    project_knn: int = 3,
    classes=None,             # iterable of class ids to clean; None = all
) -> np.ndarray:
    """
    Object-level speckle removal via per-class Euclidean clustering.

    Each predicted class is grouped into connected clusters with DBSCAN. Points that
    fall in a cluster smaller than `min_cluster_size` (or that DBSCAN flags as noise)
    are treated as spurious — e.g. a few "RRU" points sprayed onto the tower body — and
    relabelled to the majority label of their `project_knn` nearest *retained* points.

    Clustering is done independently within each predicted class so that two different
    objects that physically touch are never merged into one label. Reassignment is
    resolved against retained points only and in a single pass, so the result is
    independent of class iteration order.

    Returns (N,) int labels.
    """
    labels = labels.astype(np.int32)
    N = len(labels)
    if N == 0:
        return labels.copy()

    target_classes = range(num_classes) if classes is None else list(classes)
    spurious = np.zeros(N, dtype=bool)

    for c in target_classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        # A whole class smaller than one valid cluster is entirely speckle.
        if len(idx) < min_cluster_size:
            spurious[idx] = True
            continue

        cluster_ids = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz[idx]).labels_
        # cluster size per point (label -1 == DBSCAN noise)
        uniq, inv, counts = np.unique(cluster_ids, return_inverse=True, return_counts=True)
        member_count = counts[inv]
        drop = (cluster_ids == -1) | (member_count < min_cluster_size)
        spurious[idx[drop]] = True

    retained = ~spurious
    n_retained = int(retained.sum())
    n_spurious = int(spurious.sum())
    # Nothing to fix, or nothing to reassign against — return unchanged.
    if n_spurious == 0 or n_retained == 0:
        return labels.copy()

    new_labels = labels.copy()
    retained_idx = np.where(retained)[0]
    tree = cKDTree(xyz[retained_idx])
    k = min(project_knn, n_retained)
    _, nn = tree.query(xyz[spurious], k=k)
    nn = np.atleast_2d(nn.T).T if k == 1 else nn      # normalise to (n_spurious, k)
    neighbour_labels = labels[retained_idx[nn]]        # (n_spurious, k)

    # majority label across the k retained neighbours
    votes = np.zeros((n_spurious, num_classes), dtype=np.int32)
    rows = np.arange(n_spurious)
    for j in range(k):
        np.add.at(votes, (rows, neighbour_labels[:, j]), 1)
    new_labels[spurious] = votes.argmax(axis=1).astype(np.int32)

    print(
        f"DBSCAN cleanup: relabelled {n_spurious} / {N} points "
        f"({100.0 * n_spurious / N:.2f}%) as speckle"
    )
    return new_labels


def reclassify_ground(
    xyz: np.ndarray,          # (N, 3)
    normals: np.ndarray,      # (N, 3) per-point surface normals (upper-hemisphere)
    labels: np.ndarray,       # (N,) int predicted labels
    ground_z: float,          # estimated ground-plane height
    height_thresh: float,     # metres above ground_z to still count as "near ground"
    nz_thresh: float,         # |normal.z| above this == flat/horizontal surface
    ground_class: int = 1,    # class over-predicted on the ground (TowerBody)
    noise_class: int = 0,
) -> np.ndarray:
    """
    Relabel ground points wrongly predicted as `ground_class` to `noise_class`.

    A point is flipped only when it is (a) predicted as `ground_class`, (b) near the ground
    plane (`z - ground_z < height_thresh`) AND (c) sitting on a flat/horizontal surface
    (`|normal.z| > nz_thresh`). The normal test is the discriminator: flat ground has an
    upward normal (nz≈1) while the vertical tower base has a horizontal normal (nz≈0), so the
    real base at the same low height is preserved.
    """
    labels = labels.astype(np.int32).copy()
    near_ground = (xyz[:, 2] - ground_z) < height_thresh
    flat = np.abs(normals[:, 2]) > nz_thresh
    mask = (labels == ground_class) & near_ground & flat
    n = int(mask.sum())
    labels[mask] = noise_class
    print(f"Ground reclassify: {n} {ground_class}→{noise_class} points (near-ground + flat normal)")
    return labels


def resolve_antenna_rru(
    xyz: np.ndarray,          # (N, 3)
    labels: np.ndarray,       # (N,) int predicted labels
    eps: float,               # joint-cluster link distance (m)
    min_samples: int,         # DBSCAN core-point count
    dominant_frac: float,     # antenna fraction required to reclaim a cluster's bottom
    margin: float,            # max vertical gap (m) bridged when flood-filling down the bottom
    antenna_class: int = 2,   # PanelAntenna
    rru_class: int = 4,       # RRU
) -> np.ndarray:
    """
    Fix the PanelAntenna-bottom-predicted-as-RRU confusion (contiguous bottom strip only).

    PanelAntenna+RRU points are clustered jointly so an antenna and its mislabelled region fall
    in one cluster. In every cluster where PanelAntenna is the majority (`>= dominant_frac`), we
    start from the antenna's own vertical span (which catches a *middle* mislabel) and flood-fill
    the RRU points *both upward and downward*, bridging vertical gaps up to `margin`, flipping the
    contiguous continuation to PanelAntenna. This handles the defect at the top, middle, or bottom
    of the antenna. The flood stops at any vertical gap larger than `margin` — so a genuinely
    separate RRU above or below the antenna (or RRU-dominated clusters) is left as RRU.
    """
    labels = labels.astype(np.int32).copy()
    joint = np.where((labels == antenna_class) | (labels == rru_class))[0]
    if len(joint) == 0:
        return labels

    cluster_ids = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz[joint]).labels_
    n_flipped = 0
    for c in np.unique(cluster_ids):
        if c == -1:
            continue
        members = joint[cluster_ids == c]
        mlabels = labels[members]
        if float((mlabels == antenna_class).mean()) < dominant_frac:
            continue
        ant = members[mlabels == antenna_class]
        rru = members[mlabels == rru_class]
        if len(ant) == 0 or len(rru) == 0:
            continue

        z_lo = float(xyz[ant, 2].min())
        z_hi = float(xyz[ant, 2].max())
        rz = xyz[rru, 2]
        # RRU already inside the antenna's vertical span (middle mislabel)
        reclaim = (rz <= z_hi) & (rz >= z_lo)
        remaining = ~reclaim

        # flood-fill downward through contiguous RRU below the span, bridging gaps <= margin
        floor = z_lo
        while True:
            band = remaining & (rz >= floor - margin) & (rz < floor)
            if not band.any():
                break
            floor = float(rz[band].min())
            reclaim |= band
            remaining &= ~band

        # flood-fill upward through contiguous RRU above the span, bridging gaps <= margin
        ceil = z_hi
        while True:
            band = remaining & (rz <= ceil + margin) & (rz > ceil)
            if not band.any():
                break
            ceil = float(rz[band].max())
            reclaim |= band
            remaining &= ~band

        labels[rru[reclaim]] = antenna_class
        n_flipped += int(reclaim.sum())

    print(f"Antenna/RRU resolve: {n_flipped} {rru_class}→{antenna_class} points (contiguous antenna extension)")
    return labels
