import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from pointcloud_seg.config import Config
from pointcloud_seg.model.segmenter import CellTowerSegmenter
from pointcloud_seg.train.ema import ModelEMA
from pointcloud_seg.preprocessing.merge import merge_predictions, boundary_interior_iou
from pointcloud_seg.train.dataset import (
    TowerDataset,
    CachedChunkDataset,
    InferenceTowerData,
    BackgroundSkipSampler,
    collate_padded,
)
from pointcloud_seg.train.loss import (
    SegmentationLoss,
    compute_class_weights,
    lovász_softmax,
    classification_term,
)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism: cudnn deterministic + benchmark off. warn_only=True so GridPool's
    # CUDA scatter atomics (scatter_add_/scatter_reduce_) warn instead of crashing —
    # they remain a residual nondeterminism source until the pooling rewrite.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int):
    # Per-worker RNG seeding so num_workers>0 augmentation is reproducible.
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _warmup_cosine_schedule(warmup_epochs: int, total_epochs: int, eta_min_ratio: float = 1e-3):
    """Returns a per-epoch lr_lambda for linear warmup then cosine decay."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(epoch / max(warmup_epochs, 1), eta_min_ratio)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine
    return lr_lambda


def compute_iou(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict:
    """Per-class IoU, mIoU, and overall accuracy."""
    per_class = {}
    ious = []
    for c in range(num_classes):
        tp = int(((labels == c) & (preds == c)).sum())
        fp = int(((labels != c) & (preds == c)).sum())
        fn = int(((labels == c) & (preds != c)).sum())
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        per_class[c] = iou
        if denom > 0:
            ious.append(iou)
    per_class["mIoU"] = float(np.nanmean(ious)) if ious else float("nan")
    per_class["accuracy"] = float((preds == labels).mean())
    return per_class


# Checkpoint selector name -> filename. Shared by Trainer.run (saving) and
# _final_tta_eval (loading the deployable checkpoint).
_SELECTOR_FILES = {
    "val-miou": "best_val-miou_model.pt",
    "val-loss": "best_val-loss_model.pt",
    "train-loss": "best_train-loss_model.pt",
    "train-miou": "best_train-miou_model.pt",
    "composite": "best_composite_model.pt",
}

# Default composite weights (also the Config default). Kept here so offline
# select_best_epoch() can recompute without a Config instance.
_DEFAULT_COMPOSITE_WEIGHTS = {"miou": 1.0, "val_loss": 0.5, "train_loss": 0.25, "gap": 0.5}


def composite_score_from_entry(entry: Dict, weights: Optional[Dict] = None) -> float:
    """
    Blended checkpoint score for one training_log entry. Rewards high val mIoU and
    low train/val loss, penalises the overfit gap (train_mIoU - val_mIoU). Losses are
    mapped through exp(-loss) into (0, 1] so they compose with mIoU on a bounded,
    epoch-stable scale (greedy "save on improvement" therefore picks the true argmax):

        composite =  w_miou*val_mIoU + w_vloss*exp(-val_loss)
                  +  w_tloss*exp(-train_loss) - w_gap*max(0, train_mIoU - val_mIoU)

    Returns NaN if any required metric is missing/NaN (epoch skipped by callers).
    """
    w = weights or _DEFAULT_COMPOSITE_WEIGHTS
    miou = entry.get("val_mIoU", float("nan"))
    vloss = entry.get("val_loss", float("nan"))
    tloss = entry.get("train_loss", float("nan"))
    tr_miou = entry.get("train_mIoU", float("nan"))
    vals = [miou, vloss, tloss, tr_miou]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return float("nan")
    gap = max(0.0, tr_miou - miou)
    return (
        w.get("miou", 1.0) * miou
        + w.get("val_loss", 0.5) * math.exp(-vloss)
        + w.get("train_loss", 0.25) * math.exp(-tloss)
        - w.get("gap", 0.5) * gap
    )


def select_best_epoch(
    training_log,
    weights: Optional[Dict] = None,
    metric: str = "composite",
) -> Optional[Dict]:
    """
    Offline re-selection from a finished run's log — the in-code version of eyeballing
    the learning curve. `training_log` is a path to training_log.json or the loaded
    list of per-epoch dicts.

    metric: "composite" (uses `weights`) | "val_mIoU" | "train_mIoU" | "val_loss" | "train_loss".
    Returns {metric, best_epoch, best_score, ranking:[(epoch, score), ...]} or None.

    NOTE: this only *reports* the best epoch; it cannot recover that epoch's weights
    unless its checkpoint was saved during training (that's what Trainer.run persists).
    """
    if isinstance(training_log, (str, os.PathLike)):
        with open(training_log) as f:
            log = json.load(f)
    else:
        log = training_log

    minimize = metric in ("val_loss", "train_loss")

    def _score(e: Dict) -> float:
        if metric == "composite":
            return composite_score_from_entry(e, weights)
        return e.get(metric, float("nan"))

    scored = [(e["epoch"], _score(e)) for e in log]
    scored = [(ep, s) for ep, s in scored
              if not (s is None or (isinstance(s, float) and math.isnan(s)))]
    if not scored:
        return None
    ranking = sorted(scored, key=lambda x: x[1], reverse=not minimize)
    best_epoch, best_score = ranking[0]
    return {"metric": metric, "best_epoch": best_epoch,
            "best_score": best_score, "ranking": ranking}


class Trainer:
    def __init__(
        self,
        config: Config,
        model: CellTowerSegmenter,
        train_tower_paths: List[str],
        val_tower_paths: List[str],
        device: str = "cuda",
        fold_id: Optional[int] = None,
        train_dataset=None,
        val_from_cache: bool = False,
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.fold_id = fold_id

        set_seed(config.seed)

        # --- training dataset + class weights ---
        # An already-built dataset may be passed in (e.g. CachedChunkDataset) to avoid
        # recomputing features from .las; otherwise build a TowerDataset here.
        self.train_dataset = train_dataset if train_dataset is not None else TowerDataset(train_tower_paths, config)

        all_labels = np.concatenate(
            [lbl for _, lbl in self.train_dataset.chunks if lbl is not None], axis=0
        )
        label_counts = np.bincount(all_labels.astype(np.int64), minlength=config.num_classes)
        class_weights = compute_class_weights(
            label_counts, config.num_classes,
            method=config.class_weight_method,
            mu=config.class_weight_mu,
        ).to(device)

        self.criterion = SegmentationLoss(
            num_classes=config.num_classes,
            class_weights=class_weights,
            ce_weight=config.loss_ce_weight,
            lovasz_weight=config.loss_lovasz_weight,
            loss_type=config.loss_type,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )

        # --- val data (stored for inference-mirrored validation) ---
        # One InferenceTowerData per val tower; metrics are pooled across them.
        # val_from_cache=True rebuilds each val tower from its cached chunk_*.npz dir
        # (val_tower_paths are tower cache dirs) instead of the raw .las.
        if val_from_cache:
            self.val_datas = [InferenceTowerData.from_cache(p, config) for p in val_tower_paths]
        else:
            self.val_datas = [InferenceTowerData(p, config) for p in val_tower_paths]

        # --- optimizer ---
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # --- scheduler: linear warmup then cosine ---
        lr_lambda = _warmup_cosine_schedule(config.warmup_epochs, config.max_epochs)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # --- EMA: averaged weights used for validation + checkpointing ---
        self.ema = ModelEMA(self.model, config.ema_decay) if config.use_ema else None

        # --- AMP ---
        self.scaler = GradScaler(enabled=config.amp)

        # --- data loader ---
        # When skip_bg_chunks is on, a BackgroundSkipSampler both shuffles and drops
        # pure-background chunks per epoch (sampler and shuffle are mutually exclusive).
        # drop_last=False keeps the leftover chunks as a smaller final batch.
        if config.skip_bg_chunks:
            self.train_sampler = BackgroundSkipSampler(
                self.train_dataset.bg_chunk_flags, config.skip_bg_prob, seed=config.seed
            )
            loader_kwargs = dict(sampler=self.train_sampler)
        else:
            self.train_sampler = None
            # Seeded generator so the shuffle order is reproducible run-to-run.
            shuffle_gen = torch.Generator()
            shuffle_gen.manual_seed(config.seed)
            loader_kwargs = dict(shuffle=True, generator=shuffle_gen)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_padded,
            worker_init_fn=_seed_worker,
            **loader_kwargs,
        )

    def train_epoch(self, epoch: Optional[int] = None) -> Dict:
        # Re-roll the per-epoch background-chunk skip set (reproducible: seed + epoch).
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch if epoch is not None else 0)

        self.model.train()
        total_ce = 0.0
        total_lovasz = 0.0
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        self.optimizer.zero_grad(set_to_none=True)
        accum = self.config.grad_accum_steps
        w = self.criterion.class_weights.to(self.device) if self.criterion.class_weights is not None else None

        desc = f"Epoch {epoch} [train]" if epoch is not None else "train"
        pbar = tqdm(self.train_loader, desc=desc, leave=False, unit="batch")
        running_loss = 0.0
        for step, (feats, labels, mask) in enumerate(pbar):
            # feats: (B, M, 17), labels: (B, M), mask: (B, M)
            feats = feats.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            B = feats.shape[0]
            step_ce = 0.0
            step_lovasz = 0.0
            valid_chunks = 0

            # Process each chunk independently so KNN is built per-chunk, not across
            # the whole batch. Each chunk uses its own local (XY-centred) coordinate
            # system, so mixing them in one flat tensor would give physically
            # meaningless neighbours. This matches inference, which also runs one
            # chunk at a time.
            for i in range(B):
                mask_i = mask[i]                    # (M,) bool
                feats_i = feats[i, mask_i]          # (M_actual, 17) — no padding
                labels_i = labels[i, mask_i]        # (M_actual,)

                if feats_i.shape[0] == 0:
                    continue

                xyz_i = feats_i[:, :3]

                with autocast(enabled=self.config.amp):
                    logits_i = self.model(feats_i, xyz_i)           # (M_actual, K)
                    ce = classification_term(
                        logits_i, labels_i, weight=w,
                        loss_type=self.config.loss_type,
                        focal_gamma=self.config.focal_gamma,
                    )
                    ll = lovász_softmax(logits_i, labels_i)
                    # Divide by B*accum: average over chunks in the batch and
                    # over gradient-accumulation steps so the effective scale
                    # matches the original single-forward-pass formulation.
                    loss_i = (self.config.loss_ce_weight * ce + self.config.loss_lovasz_weight * ll) / (B * accum)

                self.scaler.scale(loss_i).backward()
                step_ce += ce.item()
                step_lovasz += ll.item()
                valid_chunks += 1

                all_preds.append(logits_i.argmax(dim=-1).detach().cpu().numpy())
                all_labels.append(labels_i.cpu().numpy())

            if valid_chunks > 0:
                step_ce_avg = step_ce / valid_chunks
                step_lovasz_avg = step_lovasz / valid_chunks
                total_ce += step_ce_avg
                total_lovasz += step_lovasz_avg
                running_loss = (
                    self.config.loss_ce_weight * step_ce_avg
                    + self.config.loss_lovasz_weight * step_lovasz_avg
                )
                pbar.set_postfix(loss=f"{running_loss:.4f}")

            if (step + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)

        # final accum flush
        if len(self.train_loader) % accum != 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)

        n = max(len(self.train_loader), 1)
        metrics = compute_iou(np.concatenate(all_preds), np.concatenate(all_labels), self.config.num_classes)
        metrics["ce_loss"] = total_ce / n
        metrics["lovasz_loss"] = total_lovasz / n
        metrics["loss"] = (self.config.loss_ce_weight * total_ce + self.config.loss_lovasz_weight * total_lovasz) / n
        return metrics

    @torch.no_grad()
    def validate(self, epoch: Optional[int] = None) -> Dict:
        """
        Single-orientation (0°, no TTA) evaluation used for per-epoch checkpoint
        ranking and early stopping: emit val tower chunks (no augmentation) →
        forward → softmax-merge overlapping chunks → per-point labels → IoU +
        boundary diagnostic.

        Also computes the validation loss (CE + Lovász, per-chunk averaged with the
        same class weights as training) purely as a monitoring signal — it is NOT
        used for early stopping or checkpoint selection. The faithful full-TTA
        inference metric is computed once at the end of run() (final_tta_val_mIoU).
        """
        self.model.eval()
        w = self.criterion.class_weights.to(self.device) if self.criterion.class_weights is not None else None

        # Validation requires ground truth on every val tower (mIoU is pooled across
        # them). Fail loudly here rather than crashing later in concatenate / loss.
        unlabelled = [vd.tower_id for vd in self.val_datas if vd.gt_labels is None]
        if unlabelled:
            raise ValueError(
                f"Validation tower(s) have no ground-truth labels: {unlabelled}. "
                "Use labelled towers for --val-dir / --val-stems."
            )

        total_ce = 0.0
        total_lovasz = 0.0
        n_loss_chunks = 0

        # pooled across all val towers (IoU is computed once on the concatenation)
        gt_all: List[np.ndarray] = []
        merged_all: List[np.ndarray] = []
        counts_all: List[np.ndarray] = []

        total_chunks = sum(len(vd.chunks) for vd in self.val_datas)
        desc = f"Epoch {epoch} [val]" if epoch is not None else "val"
        vbar = tqdm(total=total_chunks, desc=desc, leave=False, unit="chunk")

        for val_data in self.val_datas:
            per_chunk_probs: List[np.ndarray] = []
            per_chunk_indices: List[np.ndarray] = []
            per_chunk_counts: List[np.ndarray] = []

            for chunk in val_data.chunks:
                feats = torch.from_numpy(chunk.features).to(self.device)
                xyz = feats[:, :3]

                with autocast(enabled=self.config.amp):
                    logits = self.model(feats, xyz)              # (M, K)

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                per_chunk_probs.append(probs)
                per_chunk_indices.append(chunk.point_indices)
                per_chunk_counts.append(chunk.chunk_counts)

                # validation loss (monitoring only) — reuse the logits already computed
                if chunk.labels is not None:
                    labels_t = torch.from_numpy(chunk.labels).long().to(self.device)
                    with autocast(enabled=self.config.amp):
                        ce = classification_term(
                            logits, labels_t, weight=w,
                            loss_type=self.config.loss_type,
                            focal_gamma=self.config.focal_gamma,
                        )
                        ll = lovász_softmax(logits, labels_t)
                    total_ce += ce.item()
                    total_lovasz += ll.item()
                    n_loss_chunks += 1

                vbar.update(1)

            # merge overlapping chunks → one label per downsampled point (per tower)
            merged_labels = merge_predictions(
                per_chunk_probs, per_chunk_indices, val_data.N, self.config.num_classes
            )

            chunk_counts_full = np.zeros(val_data.N, dtype=np.int32)
            for idx, cc in zip(per_chunk_indices, per_chunk_counts):
                if cc is not None:
                    chunk_counts_full[idx] = cc

            gt_all.append(val_data.gt_labels)
            merged_all.append(merged_labels)
            counts_all.append(chunk_counts_full)

        vbar.close()

        gt = np.concatenate(gt_all)
        merged_labels = np.concatenate(merged_all)
        chunk_counts_full = np.concatenate(counts_all)

        metrics = compute_iou(merged_labels, gt, self.config.num_classes)

        # validation loss (per-chunk averaged across all towers; not used for selection)
        if n_loss_chunks > 0:
            metrics["ce_loss"] = total_ce / n_loss_chunks
            metrics["lovasz_loss"] = total_lovasz / n_loss_chunks
            metrics["loss"] = (
                self.config.loss_ce_weight * metrics["ce_loss"]
                + self.config.loss_lovasz_weight * metrics["lovasz_loss"]
            )

        # boundary vs interior diagnostic (never in loss)
        diag = boundary_interior_iou(gt, merged_labels, chunk_counts_full, self.config.num_classes)
        metrics["boundary_interior"] = diag

        return metrics

    def _composite_score(self, entry: Dict) -> float:
        """Blended selector score for one epoch's metrics (see module-level fn)."""
        return composite_score_from_entry(
            entry, getattr(self.config, "composite_weights", None)
        )

    def _build_selectors(self, early_stop_metric: str):
        """
        Build the active checkpoint selectors from config.save_checkpoints, always
        including the early_stop_metric's selector (needed to drive early stopping).
        Returns (selectors_dict, stop_name).

        Each selector: getter(entry)->float, maximize(bool), filename, running best.
        """
        defs = {
            "val-miou":   (lambda e: e.get("val_mIoU", float("nan")), True),
            "val-loss":   (lambda e: e.get("val_loss", float("nan")), False),
            "train-loss": (lambda e: e.get("train_loss", float("nan")), False),
            "train-miou": (lambda e: e.get("train_mIoU", float("nan")), True),
            "composite":  (lambda e: self._composite_score(e), True),
        }
        stop_name = {"val_miou": "val-miou", "train_miou": "train-miou",
                     "train_loss": "train-loss"}[early_stop_metric]

        want = list(getattr(self.config, "save_checkpoints",
                            ["val-miou", "val-loss", "train-loss", "composite"]))
        if stop_name not in want:
            want.append(stop_name)

        selectors = {}
        for name in want:
            if name not in defs:
                raise ValueError(
                    f"Unknown checkpoint selector {name!r}; choose from {list(defs)}"
                )
            getter, maximize = defs[name]
            selectors[name] = {
                "getter": getter,
                "maximize": maximize,
                "filename": _SELECTOR_FILES[name],
                "best": -math.inf if maximize else math.inf,
                "best_epoch": None,
            }
        return selectors, stop_name

    def _build_ckpt(self, epoch: int, selector_name: str, score: float,
                    val_miou: float, commit: str) -> Dict:
        """Assemble the checkpoint dict for the current model state."""
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "num_features": self.model.num_features,
            "num_classes": self.model.num_classes,
            "feature_order": [
                "xyz_local_0", "xyz_local_1", "xyz_local_2",
                "height_norm",
                "nx", "ny", "nz",
                "linearity_s", "planarity_s", "scattering_s",
                "planarity_l", "scattering_l",
                "planarity_diff", "curvature",
                "R", "G", "B",
            ],
            "best_metric_name": selector_name,
            "best_metric_value": score,
            "val_mIoU": val_miou,
            "fold": self.fold_id,
            "git_commit": commit,
        }

    def run(self, save_dir: str) -> List[Dict]:
        """
        Full training loop with AdamW + cosine/warmup LR, MLflow logging, early
        stopping on `early_stop_metric`, and multi-criterion checkpointing: each
        selector in config.save_checkpoints persists its own best_*.pt (plus a
        composite blend), so the deployable model isn't just the early-stop metric's.
        """
        os.makedirs(save_dir, exist_ok=True)
        max_epochs = self.config.max_epochs
        patience = self.config.early_stop_patience
        commit = _git_commit()

        mlflow.set_tracking_uri(self.config.log_dir)
        mlflow.set_experiment(self.config.mlflow_experiment)

        run_name = f"fold_{self.fold_id}" if self.fold_id is not None else "single"
        metric_name = self.config.early_stop_metric  # "val_miou" | "train_miou" | "train_loss"
        _valid_metrics = ("val_miou", "train_miou", "train_loss")
        if metric_name not in _valid_metrics:
            raise ValueError(
                f"early_stop_metric={metric_name!r} is not supported. "
                f"Choose one of: {_valid_metrics}"
            )
        # Independent checkpoint selectors: each persists its own best_*.pt so the
        # deployable model isn't whatever the early-stop metric happened to favour
        # (train_loss keeps falling as the model overfits). Early stopping is driven
        # ONLY by `stop_name` (the early_stop_metric selector); the others purely save.
        selectors, stop_name = self._build_selectors(metric_name)

        log: List[Dict] = []
        no_improve = 0

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("git_commit", commit)
            if self.fold_id is not None:
                mlflow.set_tag("fold", str(self.fold_id))
            mlflow.log_params({k: v for k, v in self.config.__dict__.items()
                               if not isinstance(v, (list, dict))})

            for epoch in range(1, max_epochs + 1):
                train_m = self.train_epoch(epoch=epoch)
                # Validate and checkpoint on the EMA weights; the raw training weights
                # are restored after checkpointing (below) so the next epoch continues
                # from them.
                if self.ema is not None:
                    self.ema.store(self.model)
                    self.ema.copy_to(self.model)
                val_m = self.validate(epoch=epoch)
                self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]
                val_miou = val_m.get("mIoU", float("nan"))
                train_miou = train_m.get("mIoU", float("nan"))
                gap = train_miou - val_miou if not (math.isnan(train_miou) or math.isnan(val_miou)) else float("nan")

                entry = {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": train_m["loss"],
                    "train_ce_loss": train_m["ce_loss"],
                    "train_lovasz_loss": train_m["lovasz_loss"],
                    "train_mIoU": train_miou,
                    "val_mIoU": val_miou,
                    "val_accuracy": val_m.get("accuracy", float("nan")),
                    "val_loss": val_m.get("loss", float("nan")),
                    "val_ce_loss": val_m.get("ce_loss", float("nan")),
                    "val_lovasz_loss": val_m.get("lovasz_loss", float("nan")),
                    "train_val_gap": gap,
                    # dict value — skipped by the float-only mlflow.log_metrics
                    # below, but kept so the end-of-run log_dict can find it.
                    "val_boundary_interior": val_m.get("boundary_interior"),
                }
                # per-class val IoU
                for c in range(self.config.num_classes):
                    entry[f"val_iou_class{c}"] = val_m.get(c, float("nan"))
                # per-class train IoU (already computed by compute_iou under int keys)
                for c in range(self.config.num_classes):
                    entry[f"train_iou_class{c}"] = train_m.get(c, float("nan"))
                log.append(entry)
                # Flush the log every epoch so a partial record survives a crash/kill
                # (the end-of-run dump below only fires if run() reaches completion).
                with open(os.path.join(save_dir, "training_log.json"), "w") as f:
                    json.dump(log, f, indent=2)

                mlflow.log_metrics({k: v for k, v in entry.items() if isinstance(v, float)}, step=epoch)

                val_loss = val_m.get("loss", float("nan"))
                print(
                    f"Epoch {epoch:4d}/{max_epochs} | "
                    f"loss={train_m['loss']:.4f} train_mIoU={train_miou:.4f} | "
                    f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} gap={gap:.4f} lr={current_lr:.2e}"
                )
                train_cls = " ".join(f"c{c}={train_m.get(c, float('nan')):.3f}"
                                     for c in range(self.config.num_classes))
                print(f"           train IoU/class: {train_cls}")

                # Evaluate every selector on this epoch's metrics; each saves its own
                # best_*.pt independently. The early-stop selector (`stop_name`) also
                # drives the patience counter and is aliased to best_model.pt for
                # backward-compat (e.g. older tooling that loads best_model.pt).
                improved_stop = False
                for name, s in selectors.items():
                    score = s["getter"](entry)
                    if score is None or math.isnan(score):
                        continue
                    better = (score > s["best"]) if s["maximize"] else (score < s["best"])
                    if not better:
                        continue
                    s["best"] = score
                    s["best_epoch"] = epoch
                    ckpt = self._build_ckpt(epoch, name, score, val_miou, commit)
                    torch.save(ckpt, os.path.join(save_dir, s["filename"]))
                    mlflow.log_metric(f"best_{name}", score, step=epoch)
                    print(f"  -> Saved {s['filename']} (epoch {epoch}, {name}={score:.4f})")
                    if name == stop_name:
                        improved_stop = True
                        torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))

                # Checkpoints (which capture self.model's state) are now saved with the
                # EMA weights; restore the raw training weights before the next epoch.
                if self.ema is not None:
                    self.ema.restore(self.model)

                if improved_stop:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stop at epoch {epoch} (no {stop_name} improvement "
                              f"for {patience} epochs).")
                        break

            # Per-selector best epoch + value, so the good checkpoints are discoverable
            # without re-reading the full log.
            summary = {
                name: {"best_epoch": s["best_epoch"], "best_value": s["best"],
                       "file": s["filename"]}
                for name, s in selectors.items()
            }
            with open(os.path.join(save_dir, "checkpoint_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_metric(f"final_best_{stop_name}", selectors[stop_name]["best"])
            diag = log[-1].get("val_boundary_interior") if log else None
            if diag:
                mlflow.log_dict(diag, "boundary_interior_iou.json")

            # Honest deployment metric: evaluate the best checkpoint through the
            # SAME path production inference uses (full TTA), so the reported
            # number reflects what is deployed rather than the fast 0°-only
            # per-epoch validation used for checkpoint ranking.
            tta_metrics = self._final_tta_eval(save_dir)
            if tta_metrics is not None:
                mlflow.log_metric("final_tta_val_mIoU", tta_metrics["mIoU"])
                for c in range(self.config.num_classes):
                    iou_c = tta_metrics.get(c, float("nan"))
                    if not math.isnan(iou_c):
                        mlflow.log_metric(f"final_tta_val_iou_class{c}", iou_c)
                vm_best = selectors.get("val-miou", {}).get("best", float("nan"))
                print(
                    f"Final TTA val mIoU: {tta_metrics['mIoU']:.4f} "
                    f"(best per-epoch val_mIoU: {vm_best:.4f})"
                )

        with open(os.path.join(save_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        print(f"Done. Best {metric_name}: {selectors[stop_name]['best']:.4f}")
        return log

    @torch.no_grad()
    def _final_tta_eval(self, save_dir: str) -> Optional[Dict]:
        """
        Reload the best checkpoint and evaluate the val towers through the real
        inference path (full TTA via predict_probs), pooling predictions across all
        val towers into one mIoU. Returns per-class IoU + mIoU, or None if there is no
        checkpoint / no labelled val tower.
        """
        from pointcloud_seg.infer import predict_probs

        # Evaluate the checkpoint that will actually be deployed. Defaults to the
        # best val_mIoU model (deployment-honest), falling back to best_model.pt.
        sel = getattr(self.config, "tta_eval_checkpoint", "val-miou")
        ckpt_path = os.path.join(save_dir, _SELECTOR_FILES.get(sel, "best_model.pt"))
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(save_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            return None

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        labels_all: List[np.ndarray] = []
        gt_all: List[np.ndarray] = []
        for val_data in tqdm(self.val_datas, desc="Final TTA eval", unit="tower"):
            if val_data.gt_labels is None:
                continue
            tta_angles = self.config.tta_rotations if self.config.use_tta else [0]
            probs = predict_probs(
                self.model,
                val_data.xyz,
                val_data.rgb,
                self.config,
                self.device,
                tta_rotations=tta_angles,
                amp=self.config.amp,
            )
            labels_all.append(np.argmax(probs, axis=1).astype(np.int32))
            gt_all.append(val_data.gt_labels)

        if not labels_all:
            return None
        return compute_iou(np.concatenate(labels_all), np.concatenate(gt_all),
                           self.config.num_classes)


def _find_npz_for_tower(cache_dir: str, tower_stem: str) -> List[str]:
    """Sorted chunk_*.npz paths for one tower inside a feature cache directory."""
    tower_dir = Path(cache_dir) / tower_stem
    return sorted(str(p) for p in tower_dir.glob("chunk_*.npz"))


def train_single_fold(
    config: Config,
    train_paths: List[str],
    val_paths: List[str],
    save_dir: str,
    device: str = "cuda",
    from_cache: bool = False,
    cache_dir: Optional[str] = None,
    init_weights: Optional[str] = None,
) -> List[Dict]:
    """
    Train a single model on an explicit train/val tower split. Any number of val
    towers is supported; validation metrics are pooled across them.

    from_cache=False (default): train_paths/val_paths are downsampled .las files.
    TowerDataset computes features at 0° (augmentation per-epoch in __getitem__);
    validation builds InferenceTowerData from the raw .las val towers.

    from_cache=True: train_paths are .las whose stems map to cached chunk_*.npz via
    cache_dir; val_paths are per-tower cache directories rebuilt with
    InferenceTowerData.from_cache. Behaviourally identical training distribution — the
    same augment_features + class-weighted dropout run per-epoch.

    Class weights are derived from the RAW (pre-aug, pre-dropout) chunk labels so they
    stay stable across epochs.

    init_weights: optional path to a .pt checkpoint. When set, the model weights are loaded
    (strict) as initialization AFTER the fresh model is built and BEFORE the Trainer — a
    warm-start. Training still runs from epoch 1 with a fresh optimizer/schedule; only the
    starting weights differ. Strict load requires the current config to build the identical
    architecture as the checkpoint.
    """
    print(f"\n=== Single-fold training ===")
    print(f"  Train towers : {len(train_paths)}")
    print(f"  Val towers   : {len(val_paths)} -> {[Path(p).stem for p in val_paths]}")
    print(f"  From cache   : {from_cache}")

    cached_dataset = None
    if from_cache:
        train_npz: List[str] = []
        for p in train_paths:
            pth = Path(p)
            if pth.is_dir():
                # folder-based: train_paths are per-tower cache directories
                train_npz.extend(sorted(str(c) for c in pth.glob("chunk_*.npz")))
                train_npz.extend(sorted(str(c) for c in pth.glob("aug_*.npz")))
            elif cache_dir:
                # legacy: train_paths are .las whose stems live under cache_dir
                train_npz.extend(_find_npz_for_tower(cache_dir, pth.stem))
            else:
                raise ValueError(
                    "from_cache=True needs train_paths to be tower cache dirs, "
                    "or a cache_dir to resolve tower stems against"
                )
        if not train_npz:
            raise FileNotFoundError(
                f"No .npz chunks found for training towers in {cache_dir}. "
                "Run scripts/02_extract_features.py first."
            )
        print(f"  Train chunks : {len(train_npz)}")
        cached_dataset = CachedChunkDataset(train_npz, config, augment=False)

    # Seed BEFORE building the model so weight initialization is reproducible.
    # (Trainer.__init__ also calls set_seed, which re-seeds before the dataloader
    # is built so shuffle order is independent of how much RNG init consumed.)
    set_seed(config.seed)
    model = CellTowerSegmenter(config)

    # Warm-start: load preexisting weights as initialization (strict). Optimizer, LR
    # schedule and epoch counter still start fresh — only the initial weights change.
    if init_weights:
        ckpt = torch.load(init_weights, map_location="cpu")   # mirror infer.py / _final_tta_eval
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        extra = (f" (epoch {ckpt['epoch']}, val_mIoU {ckpt.get('val_mIoU')})"
                 if isinstance(ckpt, dict) and "epoch" in ckpt else "")
        print(f"  Warm-start   : loaded weights from {init_weights}{extra}")

    trainer = Trainer(
        config=config,
        model=model,
        train_tower_paths=train_paths,
        val_tower_paths=val_paths,
        device=device,
        train_dataset=cached_dataset,   # None → Trainer builds TowerDataset from .las
        val_from_cache=from_cache,
    )

    return trainer.run(save_dir=save_dir)
