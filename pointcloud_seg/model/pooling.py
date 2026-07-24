import torch
import torch.nn as nn


class GridPool(nn.Module):
    """
    Voxel-grid pooling with a selectable aggregation over the points in each voxel.

    pool_type:
      "max"       (default, current model): channel-wise voxel max. No learnable params,
                  numerically identical to the original implementation.
      "meanmax"   : concat[segment-mean, segment-max] (M, 2C) → Linear(2C, C). The mean keeps
                  every point's contribution (order-invariant, no Frankenstein vector), the max
                  keeps the salient peak.
      "attention" : learned segment-softmax weighted sum over each voxel's points, concatenated
                  with the max, then projected. The only mode that can up-weight a rare-class
                  point regardless of how few points share its voxel.

    xyz is mean-pooled in every mode (unchanged). forward() returns
    (xyz_pooled (M,3), feats_pooled (M,C), inverse_map (N,)).
    """

    def __init__(self, voxel_size: float, channels: int = None, pool_type: str = "max"):
        super().__init__()
        self.voxel_size = voxel_size
        self.pool_type = pool_type

        if pool_type == "max":
            pass  # no params
        elif pool_type == "meanmax":
            assert channels is not None, "meanmax pooling needs channels"
            self.proj = nn.Linear(2 * channels, channels)
        elif pool_type == "attention":
            assert channels is not None, "attention pooling needs channels"
            self.score_mlp = nn.Linear(channels, 1)
            self.proj = nn.Linear(2 * channels, channels)
        else:
            raise ValueError(f"unknown pool_type {pool_type!r}")

    def _segment_max(self, feats, inverse_map, M, C):
        out = torch.full((M, C), float("-inf"), dtype=feats.dtype, device=feats.device)
        out.scatter_reduce_(0, inverse_map.unsqueeze(1).expand(-1, C), feats, reduce="amax",
                            include_self=True)
        return out

    def _segment_sum(self, feats, inverse_map, M, C):
        out = torch.zeros(M, C, dtype=feats.dtype, device=feats.device)
        out.scatter_add_(0, inverse_map.unsqueeze(1).expand(-1, C), feats)
        return out

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor):
        """
        xyz: (N, 3) float on GPU
        feats: (N, C) float on GPU
        Returns: xyz_pooled (M, 3), feats_pooled (M, C), inverse_map (N,) long
        """
        # quantize to voxel grid and group identical voxels directly (collision-free;
        # no manual coordinate hashing / offset ceiling)
        voxel_coords = torch.floor(xyz / self.voxel_size).long()  # (N, 3)
        _, inverse_map = torch.unique(voxel_coords, dim=0, return_inverse=True)
        M = int(inverse_map.max()) + 1
        C = feats.shape[1]

        # mean-pool xyz (unchanged in all modes)
        xyz_pooled = torch.zeros(M, 3, dtype=xyz.dtype, device=xyz.device)
        counts = torch.zeros(M, dtype=xyz.dtype, device=xyz.device)
        xyz_pooled.scatter_add_(0, inverse_map.unsqueeze(1).expand(-1, 3), xyz)
        counts.scatter_add_(0, inverse_map, torch.ones(xyz.shape[0], dtype=xyz.dtype, device=xyz.device))
        counts = counts.clamp(min=1)
        xyz_pooled = xyz_pooled / counts.unsqueeze(1)

        if self.pool_type == "max":
            feats_pooled = self._segment_max(feats, inverse_map, M, C)

        elif self.pool_type == "meanmax":
            mean_pooled = self._segment_sum(feats, inverse_map, M, C) / counts.unsqueeze(1)
            max_pooled = self._segment_max(feats, inverse_map, M, C)
            feats_pooled = self.proj(torch.cat([mean_pooled, max_pooled], dim=-1))

        else:  # "attention"
            score = self.score_mlp(feats)                                   # (N, 1)
            # numerically-stable segment softmax over variable-size voxels
            seg_max = torch.full((M, 1), float("-inf"), dtype=score.dtype, device=score.device)
            seg_max.scatter_reduce_(0, inverse_map.unsqueeze(1), score, reduce="amax",
                                    include_self=True)
            ex = (score - seg_max[inverse_map]).exp()                       # (N, 1)
            den = torch.zeros(M, 1, dtype=ex.dtype, device=ex.device)
            den.scatter_add_(0, inverse_map.unsqueeze(1), ex)               # (M, 1)
            w = ex / den[inverse_map].clamp_min(1e-12)                      # (N, 1), sums to 1 / voxel
            attn_pooled = self._segment_sum(feats * w, inverse_map, M, C)   # (M, C)
            max_pooled = self._segment_max(feats, inverse_map, M, C)
            feats_pooled = self.proj(torch.cat([attn_pooled, max_pooled], dim=-1))

        return xyz_pooled, feats_pooled, inverse_map


def grid_unpool(
    feats_coarse: torch.Tensor,
    inverse_map: torch.LongTensor,
    unpool_type: str = "nearest",
    knn_idx: torch.LongTensor = None,
    xyz_fine: torch.Tensor = None,
    xyz_coarse: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    inverse_map: (N_fine,) long — for each fine point, which coarse voxel it came from.
    feats_coarse: (M, C)
    Returns: (N_fine, C)

    unpool_type:
      "nearest" (default, current model): copy each fine point its own voxel's coarse feature
                — `feats_coarse[inverse_map]`. Bit-for-bit identical to the original.
      "interp"  : inverse-distance blend of the fine point's own-voxel feature with the coarse
                features of its fine KNN neighbours' voxels, so a rare point near a voxel boundary
                pulls from the nearest coarse centroids rather than only its (majority-dominated)
                own voxel. Reuses the encoder's fine-level KNN graph (knn_idx) — no new KDTree.
                Requires knn_idx (N, k), xyz_fine (N, 3), xyz_coarse (M, 3).
    """
    base = feats_coarse[inverse_map]                       # (N, C) self voxel
    if unpool_type == "nearest":
        return base

    if unpool_type != "interp":
        raise ValueError(f"unknown unpool_type {unpool_type!r}")

    assert knn_idx is not None and xyz_fine is not None and xyz_coarse is not None, (
        "interp unpool needs knn_idx, xyz_fine, xyz_coarse"
    )
    nbr_vox = inverse_map[knn_idx]                          # (N, k) neighbour voxels
    nbr_feat = feats_coarse[nbr_vox]                        # (N, k, C)
    d = (xyz_fine.unsqueeze(1) - xyz_coarse[nbr_vox]).norm(dim=-1)   # (N, k)
    w = 1.0 / (d + eps)
    w = w / w.sum(dim=1, keepdim=True)                     # (N, k), inverse-distance, sums to 1
    feats_interp = (nbr_feat * w.unsqueeze(-1)).sum(dim=1)  # (N, C)
    return 0.5 * base + 0.5 * feats_interp
