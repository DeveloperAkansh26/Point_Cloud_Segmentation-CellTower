import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from pointcloud_seg.model.norm import FeatureNorm


def build_knn(xyz: np.ndarray, k: int) -> np.ndarray:
    """
    Compute KNN on CPU. Returns (N, k_eff) int64 indices (excludes self).

    k is clamped to N-1 so a point set with fewer than k+1 points never produces an
    out-of-bounds neighbour index (cKDTree.query fills missing neighbours with index N).
    EdgeConvBlock reads k from the index shape, so a smaller k_eff per level is fine.
    """
    N = xyz.shape[0]
    tree = cKDTree(xyz)
    k_eff = min(k, N - 1)               # at most N-1 real neighbours
    if k_eff < 1:                       # single point: self is its only neighbour
        return np.zeros((N, 1), dtype=np.int64)
    _, idx = tree.query(xyz, k=k_eff + 1, workers=-1)
    return idx[:, 1:].astype(np.int64)  # (N, k_eff), excludes self


def build_knn_tensor(xyz_tensor: torch.Tensor, k: int) -> torch.LongTensor:
    """Accepts a GPU float tensor (N, 3), computes KNN on CPU, returns GPU LongTensor (N, k)."""
    xyz_np = xyz_tensor.detach().cpu().numpy()
    idx_np = build_knn(xyz_np, k)
    return torch.from_numpy(idx_np).to(xyz_tensor.device)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 16,
                 use_layernorm: bool = True, use_batchnorm: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.k = k
        self.norm = FeatureNorm(in_channels, use_layernorm, use_batchnorm)
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        # feature dropout on the transformed branch (residual/identity path stays clean)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, knn_idx: torch.LongTensor) -> torch.Tensor:
        # x: (N, in_channels), knn_idx: (N, k)
        N, C = x.shape
        k = knn_idx.shape[1]

        x_norm = self.norm(x)

        # gather neighbour features: (N, k, C)
        x_nbr = x_norm[knn_idx.view(-1)].view(N, k, C)

        # broadcast self: (N, 1, C) → (N, k, C)
        x_self = x_norm.unsqueeze(1).expand(N, k, C)

        # edge feature: concat(nbr - self, self)
        edge_feat = torch.cat([x_nbr - x_self, x_self], dim=-1)  # (N, k, 2C)

        h = self.mlp(edge_feat)            # (N, k, out_channels)
        out = h.max(dim=1)[0]             # (N, out_channels)

        return self.dropout(out) + self.residual(x)
