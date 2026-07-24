import numpy as np
from typing import Dict, List


def merge_probs(
    per_chunk_probs: List[np.ndarray],   # list of (M_i, K) float32
    per_chunk_indices: List[np.ndarray], # list of (M_i,) int
    N_total: int,
    num_classes: int,
) -> np.ndarray:
    """Softmax-average overlapping chunks. Returns (N_total, K) float32 probabilities."""
    prob_sum = np.zeros((N_total, num_classes), dtype=np.float32)
    count = np.zeros(N_total, dtype=np.int32)

    for probs, indices in zip(per_chunk_probs, per_chunk_indices):
        np.add.at(prob_sum, indices, probs)
        np.add.at(count, indices, 1)

    return prob_sum / (count[:, None] + 1e-10)


def merge_predictions(
    per_chunk_probs: List[np.ndarray],   # list of (M_i, K) float32
    per_chunk_indices: List[np.ndarray], # list of (M_i,) int
    N_total: int,
    num_classes: int,
) -> np.ndarray:
    """Softmax-average overlapping chunks, then argmax. Returns (N_total,) int labels."""
    prob_avg = merge_probs(per_chunk_probs, per_chunk_indices, N_total, num_classes)
    return np.argmax(prob_avg, axis=1).astype(np.int32)


def boundary_interior_iou(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    chunk_counts: np.ndarray,
    num_classes: int,
) -> Dict[str, Dict]:
    """
    Interior: chunk_counts == 1. Boundary: chunk_counts >= 2.
    Returns {'interior': {class: iou, 'mIoU': ...}, 'boundary': {...}}
    DIAGNOSTIC ONLY — never part of the loss.
    """
    results = {}
    for region, mask in [("interior", chunk_counts == 1), ("boundary", chunk_counts >= 2)]:
        if mask.sum() == 0:
            results[region] = {"mIoU": float("nan")}
            continue

        gt = labels_true[mask]
        pred = labels_pred[mask]
        per_class = {}
        ious = []
        for c in range(num_classes):
            tp = ((gt == c) & (pred == c)).sum()
            fp = ((gt != c) & (pred == c)).sum()
            fn = ((gt == c) & (pred != c)).sum()
            denom = tp + fp + fn
            iou = float(tp) / float(denom) if denom > 0 else float("nan")
            per_class[c] = iou
            if denom > 0:
                ious.append(iou)
        per_class["mIoU"] = float(np.mean(ious)) if ious else float("nan")
        results[region] = per_class

    return results
