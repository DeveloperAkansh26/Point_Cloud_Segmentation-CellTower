import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcloud_seg.model.norm import FeatureNorm


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C) or (B, N, C)
        z = x.mean(dim=-2)                       # global avg pool over points
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))           # channel-wise gate
        # broadcast back: z is (..., C), x is (..., N, C)
        return x * z.unsqueeze(-2)


class GlobalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int,
                 use_layernorm: bool = True, use_batchnorm: bool = False):
        super().__init__()
        self.norm1 = FeatureNorm(dim, use_layernorm, use_batchnorm)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = FeatureNorm(dim, use_layernorm, use_batchnorm)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, dim)
        x = x.unsqueeze(0)                          # (1, N, dim)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out                            # residual
        x = x + self.ffn(self.norm2(x))             # FFN + residual
        return x.squeeze(0)                         # (N, dim)
