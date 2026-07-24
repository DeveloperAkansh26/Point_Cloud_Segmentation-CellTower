import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Lovász-softmax (Berman et al., 2018) — kept for direct IoU optimisation
# ---------------------------------------------------------------------------

def lovász_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovász_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
) -> torch.Tensor:
    if probas.numel() == 0:
        return probas * 0.0

    losses = []
    class_to_sum = list(range(probas.shape[1])) if classes == "all" else labels.unique().tolist()
    for c in class_to_sum:
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovász_grad(fg_sorted)))

    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovász_softmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probas = F.softmax(logits, dim=1)
    if mask is not None:
        probas = probas[mask]
        labels = labels[mask]
    return lovász_softmax_flat(probas, labels, classes=classes)


# ---------------------------------------------------------------------------
# Focal loss (Lin et al., 2017) — class-weighted, with hard-example modulation
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Multi-class focal loss:

        FL = -alpha_y * (1 - p_y)^gamma * log(p_y)

    where p_y is the softmax probability of the true class and alpha_y is the
    per-class weight (the SAME `weight` vector used by weighted cross-entropy,
    e.g. from compute_class_weights). gamma>0 additionally down-weights
    well-classified points so the gradient concentrates on hard / rare points.

    With gamma == 0 this reduces EXACTLY to weighted cross-entropy, so the
    `loss_type` toggle in classification_term is mathematically clean.
    Mean-reduced over points.
    """
    logp = F.log_softmax(logits, dim=1)
    logp_y = logp.gather(1, labels.unsqueeze(1)).squeeze(1)   # log p_y, (N,)
    pt = logp_y.exp()                                         # p_y, (N,)
    loss = -(1.0 - pt).pow(gamma) * logp_y                    # (N,)
    if weight is not None:
        a = weight[labels]                                    # alpha_y, (N,)
        # Normalise by sum of weights — matches F.cross_entropy(weight=...)
        # so that gamma == 0 reproduces weighted CE EXACTLY and the focal/CE
        # terms stay on the same scale when mixed with Lovász.
        return (loss * a).sum() / a.sum().clamp_min(1e-12)
    return loss.mean()


def classification_term(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Per-point classification term, switchable between weighted cross-entropy and
    focal loss. Both use the same class-weight vector `weight`; focal additionally
    applies the (1 - p_y)^gamma hard-example modulation.

    loss_type: "ce"    -> F.cross_entropy(logits, labels, weight=weight, label_smoothing=...)
               "focal" -> focal_loss(logits, labels, weight=weight, gamma=focal_gamma)

    label_smoothing applies to the "ce" branch only (F.cross_entropy has native
    support); focal loss has no native smoothing and ignores this argument.
    """
    if loss_type == "focal":
        return focal_loss(logits, labels, weight=weight, gamma=focal_gamma)
    if loss_type == "ce":
        return F.cross_entropy(logits, labels, weight=weight, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss_type={loss_type!r}; expected 'ce' or 'focal'")


# ---------------------------------------------------------------------------
# Combined loss: (weighted CE | focal) + Lovász-softmax
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """
    loss = ce_weight * classification_term(logits, labels, class_weights)
         + lovasz_weight * LovászSoftmax(logits, labels)

    The classification term is either plain weighted cross-entropy
    (loss_type="ce") or class-weighted focal loss (loss_type="focal"); both share
    the same class-weight vector. Lovász is retained to directly optimise mIoU on
    rare classes.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        logits: (N, K)
        labels: (N,) long
        mask:   (N,) bool — True = real point, False = padding. Applied before both terms.
        """
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        ce = classification_term(
            logits, labels, weight=w,
            loss_type=self.loss_type, focal_gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing,
        )
        ll = lovász_softmax(logits, labels)
        return self.ce_weight * ce + self.lovasz_weight * ll


# ---------------------------------------------------------------------------
# Class weight computation — matches previous architecture exactly
# ---------------------------------------------------------------------------

def compute_class_weights(
    label_counts: np.ndarray,
    num_classes: int,
    method: str = "log",
    mu: float = 0.15,
) -> torch.Tensor:
    """
    Matches calculate_class_weights() from the previous architecture.

    method='log' (default):
        weight_c = 1 / log(1.0 + mu + count_c / total)
        normalized so mean == 1.0

    method='median': median-frequency balancing
    method='inverse': total / count_c
    """
    counts = np.array(label_counts, dtype=np.float64)
    counts = np.where(counts == 0, 1, counts)

    if method == "log":
        total = counts.sum()
        percentage = counts / total
        weights = 1.0 / np.log(1.0 + mu + percentage)

    elif method == "median":
        frequencies = counts / counts.sum()
        median_freq = np.median(frequencies)
        weights = median_freq / frequencies

    else:  # inverse
        total = counts.sum()
        weights = total / (counts + 1e-6)

    weights = weights / weights.mean()

    print("--- Class weights ---")
    for i, (c, w) in enumerate(zip(counts, weights)):
        print(f"  Class {i}: count={int(c):>10d}  weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)
