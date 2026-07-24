import torch
import torch.nn as nn


class FeatureNorm(nn.Module):
    """Last-dim (feature) normalization with independently toggleable LayerNorm + BatchNorm.

    The model carries features in the last dimension: (N, C) for the per-point blocks and
    (B, N, C) for the global-attention block. LayerNorm normalizes that last dim directly;
    BatchNorm1d normalizes each channel over the points, so it takes (N, C) as-is (N is the
    batch) and a transpose for the 3D case.

    When both are enabled the order is LayerNorm, then BatchNorm. When neither is enabled this
    is an identity. With use_layernorm=True, use_batchnorm=False it is functionally identical to
    nn.LayerNorm(dim) — the current model's default.
    """

    def __init__(self, dim: int, use_layernorm: bool = True, use_batchnorm: bool = False):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if use_layernorm else None
        self.bn = nn.BatchNorm1d(dim) if use_batchnorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ln is not None:
            x = self.ln(x)
        if self.bn is not None:
            if x.dim() == 2:                                   # (N, C)
                x = self.bn(x)
            else:                                              # (B, N, C) -> (B, C, N) -> back
                x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        return x
