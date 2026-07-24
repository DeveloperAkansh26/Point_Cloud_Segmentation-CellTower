import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from pointcloud_seg.config import Config
from pointcloud_seg.model.edgeconv import EdgeConvBlock, build_knn_tensor
from pointcloud_seg.model.pooling import GridPool, grid_unpool
from pointcloud_seg.model.attention import SEBlock, GlobalSelfAttentionBlock
from pointcloud_seg.model.norm import FeatureNorm


class CellTowerSegmenter(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.num_features >= 1, (
            f"CellTowerSegmenter requires num_features>=1, got {config.num_features}"
        )
        self.config = config
        self._num_features = config.num_features
        self._num_classes = config.num_classes
        self.knn_k = config.knn_k

        assert (len(config.encoder_widths) == len(config.encoder_depths)
                == len(config.decoder_depths) == len(config.pool_voxel_sizes)), (
            "encoder_widths, encoder_depths, decoder_depths and pool_voxel_sizes "
            f"must have equal length; got {len(config.encoder_widths)}, "
            f"{len(config.encoder_depths)}, {len(config.decoder_depths)}, "
            f"{len(config.pool_voxel_sizes)}"
        )

        widths = config.encoder_widths    # [64, 128, 256, 512]
        enc_depths = config.encoder_depths
        dec_depths = config.decoder_depths

        ln, bn = config.use_layernorm, config.use_batchnorm

        # per-feature input standardization (BatchNorm1d over the raw input features) BEFORE
        # the embedding Linear. The features live on very different scales; without this
        # the first projection is dominated by large-magnitude dims and the net latches onto
        # train-specific input statistics that don't transfer. Identity when disabled.
        self.input_norm = (
            FeatureNorm(config.num_features, use_layernorm=False, use_batchnorm=True)
            if config.standardize_input else None
        )

        # input embedding
        self.embed = nn.Sequential(
            nn.Linear(config.num_features, widths[0]),
            FeatureNorm(widths[0], ln, bn),
            nn.ReLU(inplace=True),
        )

        # encoder stages
        self.encoder_stages = nn.ModuleList()
        in_w = widths[0]
        for stage_idx, (out_w, depth) in enumerate(zip(widths, enc_depths)):
            blocks = nn.ModuleList()
            for d in range(depth):
                blocks.append(EdgeConvBlock(in_w if d == 0 else out_w, out_w, k=self.knn_k,
                                            use_layernorm=ln, use_batchnorm=bn,
                                            dropout=config.block_dropout))
            self.encoder_stages.append(blocks)
            in_w = out_w

        # grid pooling layers (one per encoder stage). channels = the encoder width at that
        # stage (the feature width AFTER the stage's EdgeConv blocks, i.e. what gets pooled).
        # pool_type defaults to "max" → identical to the original parameter-free pooling.
        self.grid_pools = nn.ModuleList([
            GridPool(vs, channels=w, pool_type=config.pool_type)
            for vs, w in zip(config.pool_voxel_sizes, widths)
        ])

        # bottleneck
        bottleneck_dim = widths[-1]  # 512
        self.bottleneck_se = SEBlock(bottleneck_dim, reduction=config.se_reduction)
        self.bottleneck_attn = GlobalSelfAttentionBlock(bottleneck_dim, config.num_attention_heads,
                                                        use_layernorm=ln, use_batchnorm=bn)

        # decoder stages (reverse order)
        self.decoder_stages = nn.ModuleList()
        self.skip_se = nn.ModuleList()
        self.skip_proj = nn.ModuleList()

        # decoder widths: each stage takes (coarse_feat + skip_feat) → decoder_width
        # we go from bottleneck (widths[-1]) back down to widths[0]
        dec_widths = list(reversed(widths))  # [512, 256, 128, 64]
        for stage_idx in range(len(widths)):
            skip_w = dec_widths[stage_idx]      # same as encoder width at this level
            coarse_w = dec_widths[stage_idx]    # coarse features have same width (from prev decoder or bottleneck)
            out_w = dec_widths[stage_idx + 1] if stage_idx + 1 < len(dec_widths) else dec_widths[-1]

            # skip_proj merges coarse (from unpool) + skip connection
            # at stage 0: coarse = bottleneck (512), skip = enc_stage[-1] (512) -> 512 -> 256
            concat_w = coarse_w + skip_w
            self.skip_se.append(SEBlock(skip_w, reduction=config.se_reduction))
            self.skip_proj.append(nn.Linear(concat_w, out_w, bias=False))

            depth = dec_depths[stage_idx]
            blocks = nn.ModuleList()
            for d in range(depth):
                blocks.append(EdgeConvBlock(out_w, out_w, k=self.knn_k,
                                            use_layernorm=ln, use_batchnorm=bn,
                                            dropout=config.block_dropout))
            self.decoder_stages.append(blocks)

        # segmentation head
        # Fix 3B: when head_fuse_fine, concat the un-pooled level-0 embedding (widths[0]) with the
        # final decoder features so every point reaches the head with a feature that never passed
        # through any pooling. Default OFF → head_in = dec_widths[-1] (current model).
        self.head_fuse_fine = config.head_fuse_fine
        head_in = dec_widths[-1] + (widths[0] if config.head_fuse_fine else 0)  # 64 or 128
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.head_dropout),
            nn.Linear(64, config.num_classes),
        )

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _run_stage(self, blocks: nn.ModuleList, x: torch.Tensor, knn_idx: torch.LongTensor) -> torch.Tensor:
        for block in blocks:
            x = block(x, knn_idx)
        return x

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        feats: (N, 17), xyz: (N, 3)
        Returns: (N, num_classes) logits
        """
        assert feats.shape[-1] == self._num_features, (
            f"Expected {self._num_features} features, got {feats.shape[-1]}"
        )

        if self.input_norm is not None:
            feats = self.input_norm(feats)   # per-feature standardization (N, 17)

        x = self.embed(feats)   # (N, 64)
        fine_embed = x          # un-pooled level-0 features, kept for optional head fusion (Fix 3B)

        # encoder
        enc_skips: List[torch.Tensor] = []
        enc_knns: List[torch.LongTensor] = []
        inverse_maps: List[torch.LongTensor] = []
        xyz_levels: List[torch.Tensor] = [xyz]

        cur_xyz = xyz
        cur_x = x

        for stage_idx, (blocks, pool) in enumerate(zip(self.encoder_stages, self.grid_pools)):
            knn_idx = build_knn_tensor(cur_xyz, self.knn_k)
            enc_knns.append(knn_idx)
            cur_x = self._run_stage(blocks, cur_x, knn_idx)
            enc_skips.append(cur_x)

            # pool
            cur_xyz, cur_x, inv_map = pool(cur_xyz, cur_x)
            inverse_maps.append(inv_map)
            xyz_levels.append(cur_xyz)

        # bottleneck
        cur_x = self.bottleneck_se(cur_x)
        cur_x = self.bottleneck_attn(cur_x)

        # decoder (reverse)
        n_stages = len(self.encoder_stages)
        for stage_idx in range(n_stages):
            enc_level = n_stages - 1 - stage_idx

            # unpool back to this level's fine resolution. "interp" reuses the encoder's
            # fine-level KNN graph (built on xyz_levels[enc_level]) + the fine/coarse xyz so no
            # new KDTree is needed; "nearest" (default) ignores the extra args.
            inv_map = inverse_maps[enc_level]
            cur_x = grid_unpool(
                cur_x, inv_map,
                unpool_type=self.config.unpool_type,
                knn_idx=enc_knns[enc_level][:, : self.config.unpool_interp_k],
                xyz_fine=xyz_levels[enc_level],
                xyz_coarse=xyz_levels[enc_level + 1],
            )

            # SE-gate the skip, then concat + project
            skip = enc_skips[enc_level]
            skip_gated = self.skip_se[stage_idx](skip)
            cur_x = torch.cat([cur_x, skip_gated], dim=-1)
            cur_x = self.skip_proj[stage_idx](cur_x)
            cur_x = F.relu(cur_x, inplace=True)

            # reuse matching encoder KNN graph
            knn_idx = enc_knns[enc_level]
            cur_x = self._run_stage(self.decoder_stages[stage_idx], cur_x, knn_idx)

        if self.head_fuse_fine:
            cur_x = torch.cat([cur_x, fine_embed], dim=-1)   # (N, dec_widths[-1] + widths[0])

        logits = self.head(cur_x)   # (N, num_classes)
        return logits
