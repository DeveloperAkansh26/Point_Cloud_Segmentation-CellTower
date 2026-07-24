from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yaml
import os


@dataclass
class Config:
    # ---- paths ----
    raw_dir: str = "/path/to/raw"
    downsampled_dir: str = "/path/to/downsampled"
    cache_dir: Optional[str] = None
    output_dir: str = "/path/to/outputs"

    # ---- stage 1: downsampling ----
    downsample_cm: float = 3.0
    pdal_dataformat: int = 7
    pdal_workers: int = 35

    # ---- stage 2/3: features ----
    num_classes: int = 5
    knn_small: int = 32
    knn_large: int = 96
    # object-scale neighborhood for the antenna/RRU-discriminating features (scattering_o,
    # planarity_o, linearity_o, vert_extent_o). Must span a whole RRU box (~0.4 m) so its
    # volumetric scatter separates from a flat antenna panel. At 3 cm downsample k=1300 ≈ 0.6 m
    # radius — measured to give the strongest 2↔4 separation (Δscat_o≈−0.12 vs ≈0 at k=256).
    # Fixed-k (not radius) so the neighborhood spreads ALONG a thin panel and keeps the antenna's
    # vertical-extent signal; also keeps the vectorized eigh batch in features.py. Tune 768–2048.
    knn_xlarge: int = 1300
    num_features: int = 21  # ASSERTED — do not change without updating feature builder
    ground_percentile: float = 1.0
    # raw classification labels to merge before training/eval (raw -> target),
    # applied once in _load_las and baked into the .npz cache. e.g. {5: 1}
    merge_labels: dict = field(default_factory=dict)

    # ---- stage 4: super-chunks ----
    chunk_target_points: int = 150_000
    chunk_max_points: int = 200_000
    overlap_fraction: float = 0.5
    point_ceiling: int = 1_500_000

    # class-weighted point dropout (set True to enable; False by default)
    aug_class_dropout: bool = False
    # extra survival-probability multiplier per class (< 1 = more dropout)
    dropout_class_multipliers: dict = field(default_factory=lambda: {1: 0.25, 2: 0.25})
    # fixed Z-rotation angles (degrees) used for offline chunk augmentation in 02_extract_features.py
    offline_aug_rotations: List[int] = field(default_factory=lambda: [90, 180, 270])

    # ---- stage 5: augmentation (TRAIN only) ----
    aug_rotate_z: bool = True
    aug_scale_range: Tuple[float, float] = (0.9, 1.1)
    aug_jitter_sigma_m: float = 0.01
    aug_colour_jitter: bool = True

    # ---- stage 6/7: inference merge + projection ----
    merge_method: str = "softmax_average"
    project_knn: int = 3

    # DBSCAN cluster-based speckle cleanup (post-processing, off by default).
    # Groups each predicted class into Euclidean clusters; clusters smaller than
    # dbscan_min_cluster_size (and DBSCAN noise points) are treated as speckle and
    # relabelled to the majority class of their `project_knn` nearest retained neighbours.
    use_dbscan_postprocess: bool = False
    dbscan_eps: float = 0.10             # metres; ~3x the 3cm downsample spacing
    dbscan_min_samples: int = 10         # DBSCAN core-point neighbour count
    dbscan_min_cluster_size: int = 50    # clusters below this are treated as speckle
    dbscan_classes: Optional[List[int]] = None  # None = all classes; else restrict cleanup

    # Ground → Noise reclassification (post-processing, off by default).
    # Relabels near-ground points with an upward-facing normal from ground_class to noise_class;
    # the normal test protects the vertical tower base (horizontal normal) at the same height.
    use_ground_reclassify: bool = False
    ground_height_thresh: float = 0.5    # metres above the estimated ground plane
    ground_nz_thresh: float = 0.85       # |normal.z| above this = flat/horizontal (ground, not wall)
    ground_class: int = 1                # class over-predicted on the ground (TowerBody)
    noise_class: int = 0                 # (ground level estimated from ground_percentile)

    # Antenna/RRU confusion resolution (post-processing, off by default).
    # Joint-clusters PanelAntenna+RRU; in antenna-dominated clusters flips RRU points that are the
    # contiguous top/middle/bottom continuation of the antenna back to PanelAntenna. Standalone RRUs stay.
    use_confusion_resolve: bool = False
    confusion_eps: float = 0.30          # joint-cluster link distance (m)
    confusion_min_samples: int = 10
    confusion_dominant_frac: float = 0.5  # antenna fraction needed to reclaim its mislabelled part
    confusion_margin: float = 0.10       # max vertical gap (m) bridged when flood-filling up/down
    antenna_class: int = 2
    rru_class: int = 4

    # ---- model ----
    knn_k: int = 32
    encoder_widths: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_depths: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    pool_ratio: int = 4          # grid-pool x4 per stage
    num_attention_heads: int = 4
    head_dropout: float = 0.4
    block_dropout: float = 0.0   # feature dropout inside EVERY EdgeConv block (enc+dec); 0.0 = off
    se_reduction: int = 4        # SE block squeeze ratio

    # normalization (independent toggles; defaults = current LayerNorm-only model)
    #   both on  -> LayerNorm then BatchNorm   one on -> just that norm   both off -> no norm
    use_layernorm: bool = True
    use_batchnorm: bool = False

    # per-feature input standardization (BatchNorm1d over the 17 raw features) applied
    # BEFORE the embedding Linear. Equalizes the heterogeneous input scales and recenters
    # inputs at eval — the legacy input-BatchNorm analogue. Default ON (the prior default,
    # which fed raw unnormalized features, overfit/collapsed on val). Set False to recover
    # the exact pre-change forward pass.
    standardize_input: bool = True

    # voxel sizes (metres) per encoder stage for grid pooling
    pool_voxel_sizes: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.40])

    # ---- rare-class-aware backbone (all default to the CURRENT model) ----
    # Fix 1 — pooling aggregation inside each GridPool:
    #   "max"       (default, current): channel-wise voxel max (no params)
    #   "meanmax"   : concat[segment-mean, max] → Linear(2C, C)
    #   "attention" : learned segment-softmax weighted sum, concat with max → Linear(2C, C)
    pool_type: str = "max"
    # Fix 2 — decoder unpool interpolation:
    #   "nearest" (default, current): copy each fine point its own voxel's coarse feature
    #   "interp"  : inverse-distance blend over the fine KNN neighbours' voxels
    unpool_type: str = "nearest"
    unpool_interp_k: int = 8        # neighbours used by "interp" unpool (must be <= knn_k)
    # Fix 3B — fuse the un-pooled level-0 embedding into the head (default OFF = current model)
    head_fuse_fine: bool = False

    # ---- optimizer / schedule ----
    optimizer: str = "adamw"
    lr: float = 0.001
    weight_decay: float = 0.05
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    max_epochs: int = 300
    early_stop_patience: int = 30
    early_stop_metric: str = "val_miou"

    # ---- EMA (exponential moving average of weights) ----
    use_ema: bool = True        # validate + checkpoint on EMA weights (smooths val volatility)
    ema_decay: float = 0.999    # lower (~0.99) if very few optimizer steps per epoch

    # ---- checkpoint selection ----
    # Each selector saves its own best_*.pt every run; early stopping still uses only
    # early_stop_metric. Names: "val-miou" | "val-loss" | "train-loss" | "train-miou" | "composite".
    save_checkpoints: List[str] = field(
        default_factory=lambda: ["val-miou", "val-loss", "train-loss", "composite"]
    )
    # Weights for the blended "composite" checkpoint score:
    #   w_miou*val_mIoU + w_vloss*exp(-val_loss) + w_tloss*exp(-train_loss)
    #   - w_gap*max(0, train_mIoU - val_mIoU)
    composite_weights: dict = field(
        default_factory=lambda: {"miou": 1.0, "val_loss": 0.5, "train_loss": 0.25, "gap": 0.5}
    )
    # Which saved checkpoint the end-of-run full-TTA eval loads (deployment-honest).
    tta_eval_checkpoint: str = "val-miou"

    # ---- batching ----
    batch_size: int = 4
    grad_accum_steps: int = 1
    num_workers: int = 8
    amp: bool = True
    grad_checkpoint: bool = False
    # drop all-background chunks with prob skip_bg_prob each epoch (train only)
    skip_bg_chunks: bool = False
    skip_bg_prob: float = 0.5
    bg_classes: List[int] = field(default_factory=lambda: [0, 1])  # Noise, TowerBody

    # ---- loss ----
    loss_type: str = "ce"                # "ce" (weighted cross-entropy) | "focal"
    focal_gamma: float = 2.0             # gamma for focal loss; 0 == weighted CE
    loss_ce_weight: float = 1.0          # weight on the classification (CE/focal) term
    loss_lovasz_weight: float = 1.0      # weight on the Lovász-softmax term
    class_weight_method: str = "log"     # "log" | "median" | "inverse"
    class_weight_mu: float = 0.15        # mu for log method: 1/log(1+mu+freq)
    label_smoothing: float = 0.0         # CE label smoothing epsilon; 0.0 = off. Try 0.05–0.1.

    # ---- inference / TTA ----
    use_tta: bool = False   # enable test-time augmentation at inference (off by default)
    tta_rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])

    # ---- logging ----
    mlflow_experiment: str = "celltower_seg"
    log_dir: str = "/path/to/mlflow"
    checkpoint_dir: str = "/path/to/checkpoints"
    seed: int = 42

    def __post_init__(self):
        valid_pool = {"max", "meanmax", "attention"}
        if self.pool_type not in valid_pool:
            raise ValueError(f"pool_type must be one of {sorted(valid_pool)}, got {self.pool_type!r}")
        valid_unpool = {"nearest", "interp"}
        if self.unpool_type not in valid_unpool:
            raise ValueError(f"unpool_type must be one of {sorted(valid_unpool)}, got {self.unpool_type!r}")
        if self.unpool_type == "interp" and self.unpool_interp_k > self.knn_k:
            raise ValueError(
                f"unpool_interp_k ({self.unpool_interp_k}) must be <= knn_k ({self.knn_k}) — "
                "interp reuses the encoder KNN graph, which has knn_k neighbours."
            )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f)
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        c.__post_init__()   # re-validate after YAML overrides
        return c

    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
