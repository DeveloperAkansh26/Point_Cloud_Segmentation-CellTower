import numpy as np
import laspy
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from pointcloud_seg.config import Config
from pointcloud_seg.preprocessing.features import compute_global_features
from pointcloud_seg.preprocessing.chunking import SuperChunk, slice_into_superchunks
from pointcloud_seg.preprocessing.augmentation import augment_features, class_weighted_dropout


def list_towers_in_dir(directory: str, from_cache: bool) -> List[str]:
    """
    Enumerate the towers contained in a folder.

    from_cache=False: returns sorted .las/.laz file paths in the folder.
    from_cache=True:  returns sorted immediate sub-directories that contain chunk_*.npz
                      (one tower per sub-directory, the layout written by
                      scripts/02_extract_features.py).
    """
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")

    if from_cache:
        towers = [str(sub) for sub in sorted(d.iterdir())
                  if sub.is_dir() and any(sub.glob("chunk_*.npz"))]
    else:
        towers = sorted([str(p) for p in d.glob("*.las")] + [str(p) for p in d.glob("*.laz")])

    if not towers:
        kind = "cache tower sub-directories" if from_cache else ".las/.laz files"
        raise FileNotFoundError(f"No {kind} found in {directory}")
    return towers


def _compute_bg_chunk_flags(
    chunks: List[Tuple[np.ndarray, np.ndarray]], config: Config
) -> List[bool]:
    """
    Per-chunk flag: True iff the chunk's labels are entirely in config.bg_classes
    (e.g. {0: Noise, 1: TowerBody}). Computed once on the raw stored labels so it
    stays stable across epochs. Used by BackgroundSkipSampler to drop pure-background
    chunks with a per-epoch probability.
    """
    bg = list(config.bg_classes)
    return [
        bool(lbl is not None and len(lbl) > 0 and np.isin(lbl, bg).all())
        for _, lbl in chunks
    ]


class BackgroundSkipSampler(torch.utils.data.Sampler):
    """
    Each epoch, drop all-background chunks with probability `skip_prob`; shuffle the
    rest. __iter__ is re-invoked by the DataLoader once per epoch, so the skip set is
    re-rolled fresh each epoch. Reseeded per epoch (seed + epoch) for reproducibility;
    call set_epoch(epoch) before each epoch.
    """

    def __init__(self, bg_flags: List[bool], skip_prob: float, seed: int = 0):
        self.bg_flags = bg_flags
        self.skip_prob = skip_prob
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        idx = [
            i for i, is_bg in enumerate(self.bg_flags)
            if not (is_bg and g.random() < self.skip_prob)
        ]
        g.shuffle(idx)
        return iter(idx)

    def __len__(self) -> int:
        # Upper bound (all chunks kept); the real per-epoch count is <= this. Only
        # affects the DataLoader length estimate / tqdm total, not correctness.
        return len(self.bg_flags)


def _apply_label_merge(labels: Optional[np.ndarray], config: Optional[Config]) -> Optional[np.ndarray]:
    """
    Remap raw classification labels per config.merge_labels (raw -> target), e.g. {5: 1}.
    Reads from the original array and writes a copy, so it is idempotent and merges never
    chain into one another. No-op when labels is None or no merge map is configured.
    """
    if labels is None or config is None or not getattr(config, "merge_labels", None):
        return labels
    out = labels.copy()
    for src, dst in config.merge_labels.items():
        out[labels == int(src)] = int(dst)
    return out


def _load_las(path: str, config: Optional[Config] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load xyz (N,3) float64, rgb (N,3) float32 [0,1], labels (N,) int64 or None.

    When `config` is passed, config.merge_labels is applied to the labels here — the single
    chokepoint for label remapping, so it is baked into the .npz cache (Stage 2) and applied
    identically on the --from-las training/val paths.
    """
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

    labels = None
    if hasattr(las, "raw_classification"):
        labels = np.array(las.raw_classification, dtype=np.int64)
    elif hasattr(las, "classification"):
        labels = np.array(las.classification, dtype=np.int64)

    labels = _apply_label_merge(labels, config)
    return xyz, rgb, labels



class TowerDataset(Dataset):
    """
    Training dataset: one item per super-chunk.

    Features are computed ONCE at 0° (no augmentation) and stored raw. Augmentation
    (Z-rotation + scale + jitter + colour jitter) and inverse-frequency class dropout
    are applied analytically in __getitem__, so each epoch resamples a fresh stochastic
    view of every chunk. Class weights should be computed from the raw stored labels
    (self.chunks) so they stay stable across epochs.
    """

    def __init__(self, tower_paths: List[str], config: Config, augment: bool = True):
        super().__init__()
        self.config = config
        self.augment = augment
        # raw, 0° (feats, labels) per chunk — augmentation/dropout happen in __getitem__
        self.chunks: List[Tuple[np.ndarray, np.ndarray]] = []

        for path in tqdm(tower_paths, desc="Building train features", unit="tower"):
            tower_id = Path(path).stem
            xyz, rgb, labels = _load_las(path, config)

            precomputed = compute_global_features(xyz, rgb, config)

            superchunks: List[SuperChunk] = slice_into_superchunks(
                xyz, rgb, precomputed, config,
                labels=labels,
                tower_id=tower_id,
            )

            for chunk in superchunks:
                self.chunks.append((chunk.features, chunk.labels))

        self.bg_chunk_flags = _compute_bg_chunk_flags(self.chunks, config)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, labels = self.chunks[idx]
        if self.augment:
            features = augment_features(features, self.config)
            if labels is not None and self.config.aug_class_dropout:
                features, labels = class_weighted_dropout(features, labels, self.config)
        return torch.from_numpy(features), torch.from_numpy(labels).long()


class CachedChunkDataset(Dataset):
    """
    Training dataset that loads pre-extracted .npz chunks (from
    scripts/02_extract_features.py) instead of recomputing features from .las.

    Behaviourally identical to TowerDataset: the cached features are the 0° features,
    and the SAME per-epoch augment_features + class_weighted_dropout are applied in
    __getitem__. Points with label < 0 (unlabelled padding) are dropped once at load.
    """

    def __init__(self, npz_paths: List[str], config: Config, augment: bool = True):
        super().__init__()
        self.config = config
        self.augment = augment
        self.chunks: List[Tuple[np.ndarray, np.ndarray]] = []

        for p in tqdm(npz_paths, desc="Loading cached chunks", unit="chunk"):
            d = np.load(p)
            feats = d["features"]      # (M, 17) float32
            labels = d["labels"]       # (M,) int64

            valid = labels >= 0
            if valid.sum() == 0:
                continue
            self.chunks.append((feats[valid], labels[valid].astype(np.int64)))

        self.bg_chunk_flags = _compute_bg_chunk_flags(self.chunks, config)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, labels = self.chunks[idx]
        if self.augment:
            features = augment_features(features, self.config)
            if self.config.aug_class_dropout:
                features, labels = class_weighted_dropout(features, labels, self.config)
        return torch.from_numpy(features), torch.from_numpy(labels).long()


def collate_padded(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length chunks to the same M for batching.
    Returns (features (B, M_max, 17), labels (B, M_max), mask (B, M_max) bool).
    Padded positions have label -1 (ignored by loss).
    """
    max_m = max(f.shape[0] for f, _ in batch)
    B = len(batch)
    C = batch[0][0].shape[1]

    feat_out = torch.zeros(B, max_m, C, dtype=torch.float32)
    lbl_out = torch.full((B, max_m), fill_value=-1, dtype=torch.long)
    mask_out = torch.zeros(B, max_m, dtype=torch.bool)

    for i, (f, l) in enumerate(batch):
        M = f.shape[0]
        feat_out[i, :M] = f
        lbl_out[i, :M] = l
        mask_out[i, :M] = True

    return feat_out, lbl_out, mask_out


class InferenceTowerData:
    """
    Loads one labelled tower for inference-style validation.
    No augmentation. Stores chunks with point_indices and chunk_counts
    so validation can mirror the real inference merge step.

    Two sources, identical interface (.chunks, .N, .xyz, .rgb, .gt_labels):
      - __init__(las_path):        compute 0° features from a downsampled .las file.
      - from_cache(tower_dir):     rebuild from pre-extracted chunk_*.npz (no recompute).
    """

    def __init__(self, las_path: str, config: Config):
        tower_id = Path(las_path).stem
        xyz, rgb, labels = _load_las(las_path, config)

        precomputed = compute_global_features(xyz, rgb, config)
        chunks: List[SuperChunk] = slice_into_superchunks(
            xyz, rgb, precomputed, config,
            labels=labels,
            tower_id=tower_id,
        )

        self.tower_id = tower_id
        self.chunks = chunks
        self.N = len(xyz)
        # Raw cloud kept so the end-of-run TTA evaluation can call predict_probs
        # (the real deployment path) instead of the precomputed 0° chunks.
        self.xyz = xyz
        self.rgb = rgb
        # Ground truth per original downsampled point
        self.gt_labels: Optional[np.ndarray] = labels

    @classmethod
    def from_cache(cls, tower_dir: str, config: Config) -> "InferenceTowerData":
        """
        Rebuild a validation tower from its cached chunk_*.npz files instead of the
        raw .las. Each npz stores features, labels, xyz (raw xyz[point_indices]),
        point_indices and chunk_counts (scripts/02_extract_features.py), which is
        everything validation + final-TTA need. The full-cloud rgb required by
        predict_probs is recovered from features[:, 14:17] (raw RGB in [0, 1]).
        """
        npz_paths = sorted(str(p) for p in Path(tower_dir).glob("chunk_*.npz"))
        if not npz_paths:
            raise FileNotFoundError(f"No chunk_*.npz found in {tower_dir}")

        self = cls.__new__(cls)
        tower_id = Path(tower_dir).name
        chunks: List[SuperChunk] = []
        max_idx = -1
        loaded = []
        for p in npz_paths:
            d = np.load(p)
            feats = d["features"]
            labels = d["labels"].astype(np.int64)
            point_indices = d["point_indices"].astype(np.int64)
            chunk_counts = d["chunk_counts"] if "chunk_counts" in d else None
            xyz_chunk = d["xyz"]
            chunks.append(SuperChunk(
                features=feats,
                point_indices=point_indices,
                labels=labels,
                tower_id=tower_id,
                chunk_counts=chunk_counts,
            ))
            loaded.append((point_indices, xyz_chunk, feats, labels))
            max_idx = max(max_idx, int(point_indices.max()))

        N = max_idx + 1
        xyz_full = np.zeros((N, 3), dtype=np.float64)
        rgb_full = np.zeros((N, 3), dtype=np.float32)
        gt_full = np.full(N, -1, dtype=np.int64)
        # Scatter per-point arrays back to the full downsampled cloud; overlapping
        # chunks carry consistent values per point, so last-write is fine.
        for point_indices, xyz_chunk, feats, labels in loaded:
            xyz_full[point_indices] = xyz_chunk
            rgb_full[point_indices] = feats[:, 14:17]
            gt_full[point_indices] = labels

        self.tower_id = tower_id
        self.chunks = chunks
        self.N = N
        self.xyz = xyz_full
        self.rgb = rgb_full
        self.gt_labels = gt_full if (gt_full >= 0).any() else None
        # Parity with the las path: a fully unlabelled tower has labels=None per chunk
        # (02_extract writes an all -1 array), so validate() skips its loss instead of
        # feeding -1 targets to cross-entropy.
        if self.gt_labels is None:
            for chunk in self.chunks:
                chunk.labels = None
        return self
