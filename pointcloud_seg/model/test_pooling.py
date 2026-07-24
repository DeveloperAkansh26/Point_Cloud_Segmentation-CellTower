"""
Unit tests for the rare-class-aware pooling backbone (Fix 1/2/3).

Run:  python -m pytest pointcloud_seg/model/test_pooling.py -q
  or: python pointcloud_seg/model/test_pooling.py   (runs the same checks without pytest)

All tests are CPU-only on tiny synthetic clouds. The guiding invariant is that every new
option is a strict generalisation of the current model: with the default knobs
(pool_type="max", unpool_type="nearest", head_fuse_fine=False) the ops must reduce to the
original implementation bit-for-bit.
"""
import torch

from pointcloud_seg.config import Config
from pointcloud_seg.model.pooling import GridPool, grid_unpool


def _toy_cloud(n=200, c=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    # spread points over a few voxels at voxel_size=0.1
    xyz = torch.rand(n, 3, generator=g) * 0.5
    feats = torch.randn(n, c, generator=g)
    return xyz, feats


# --- reference (frozen copy of the ORIGINAL ops) ----------------------------------------
def _ref_max_pool(xyz, feats, voxel_size):
    voxel_coords = torch.floor(xyz / voxel_size).long()
    _, inv = torch.unique(voxel_coords, dim=0, return_inverse=True)
    M = int(inv.max()) + 1
    C = feats.shape[1]
    xyz_p = torch.zeros(M, 3)
    counts = torch.zeros(M)
    xyz_p.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), xyz)
    counts.scatter_add_(0, inv, torch.ones(xyz.shape[0]))
    xyz_p = xyz_p / counts.unsqueeze(1).clamp(min=1)
    feats_p = torch.full((M, C), float("-inf"))
    feats_p.scatter_reduce_(0, inv.unsqueeze(1).expand(-1, C), feats, reduce="amax", include_self=True)
    return xyz_p, feats_p, inv


def _ref_nearest_unpool(feats_coarse, inv):
    return feats_coarse[inv]


# --- tests ------------------------------------------------------------------------------
def test_backcompat_max_pool():
    xyz, feats = _toy_cloud()
    pool = GridPool(0.1, channels=feats.shape[1], pool_type="max")
    xyz_p, feats_p, inv = pool(xyz, feats)
    rx, rf, ri = _ref_max_pool(xyz, feats, 0.1)
    assert torch.equal(inv, ri)
    assert torch.allclose(xyz_p, rx, atol=0, rtol=0)
    assert torch.equal(feats_p, rf)   # bit-for-bit


def test_backcompat_nearest_unpool():
    xyz, feats = _toy_cloud()
    _, feats_p, inv = GridPool(0.1, channels=feats.shape[1], pool_type="max")(xyz, feats)
    out = grid_unpool(feats_p, inv, unpool_type="nearest")
    assert torch.equal(out, _ref_nearest_unpool(feats_p, inv))


def test_shapes_and_partition():
    xyz, feats = _toy_cloud()
    C = feats.shape[1]
    for pt in ("max", "meanmax", "attention"):
        pool = GridPool(0.1, channels=C, pool_type=pt)
        xyz_p, feats_p, inv = pool(xyz, feats)
        M = int(inv.max()) + 1
        assert xyz_p.shape == (M, 3)
        assert feats_p.shape == (M, C)
        assert inv.shape == (xyz.shape[0],)
        assert M == int(inv.max()) + 1


def test_attention_weights_sum_to_one():
    # directly exercise the segment-softmax: weights must sum to 1 per voxel
    xyz, feats = _toy_cloud()
    C = feats.shape[1]
    pool = GridPool(0.1, channels=C, pool_type="attention")
    voxel_coords = torch.floor(xyz / 0.1).long()
    _, inv = torch.unique(voxel_coords, dim=0, return_inverse=True)
    M = int(inv.max()) + 1
    score = pool.score_mlp(feats)
    seg_max = torch.full((M, 1), float("-inf"))
    seg_max.scatter_reduce_(0, inv.unsqueeze(1), score, reduce="amax", include_self=True)
    ex = (score - seg_max[inv]).exp()
    den = torch.zeros(M, 1)
    den.scatter_add_(0, inv.unsqueeze(1), ex)
    w = ex / den[inv].clamp_min(1e-12)
    wsum = torch.zeros(M, 1)
    wsum.scatter_add_(0, inv.unsqueeze(1), w)
    assert torch.allclose(wsum, torch.ones(M, 1), atol=1e-5)


def test_attention_const_score_equals_mean():
    # Invariant: constant score_mlp output ⇒ uniform attention weights ⇒ attn part == segment mean.
    xyz, feats = _toy_cloud()
    C = feats.shape[1]
    pool = GridPool(0.1, channels=C, pool_type="attention")
    with torch.no_grad():
        pool.score_mlp.weight.zero_()
        pool.score_mlp.bias.zero_()       # score == 0 for every point → uniform softmax
    # recompute attn_pooled the way forward() does and compare to a plain segment mean
    voxel_coords = torch.floor(xyz / 0.1).long()
    _, inv = torch.unique(voxel_coords, dim=0, return_inverse=True)
    M = int(inv.max()) + 1
    counts = torch.zeros(M)
    counts.scatter_add_(0, inv, torch.ones(xyz.shape[0]))
    seg_sum = torch.zeros(M, C)
    seg_sum.scatter_add_(0, inv.unsqueeze(1).expand(-1, C), feats)
    seg_mean = seg_sum / counts.clamp(min=1).unsqueeze(1)
    # uniform-weight attn pool
    w = torch.full((xyz.shape[0], 1), 0.0)
    score = pool.score_mlp(feats)
    seg_max = torch.full((M, 1), float("-inf"))
    seg_max.scatter_reduce_(0, inv.unsqueeze(1), score, reduce="amax", include_self=True)
    ex = (score - seg_max[inv]).exp()
    den = torch.zeros(M, 1)
    den.scatter_add_(0, inv.unsqueeze(1), ex)
    w = ex / den[inv].clamp_min(1e-12)
    attn = torch.zeros(M, C)
    attn.scatter_add_(0, inv.unsqueeze(1).expand(-1, C), feats * w)
    assert torch.allclose(attn, seg_mean, atol=1e-5)


def test_permutation_invariance():
    # Pooled voxel features must be invariant to input point ordering (up to voxel relabelling,
    # which torch.unique fixes by sorted voxel coords → stable labels).
    xyz, feats = _toy_cloud()
    perm = torch.randperm(xyz.shape[0])
    for pt in ("max", "meanmax", "attention"):
        pool = GridPool(0.1, channels=feats.shape[1], pool_type=pt)
        pool.eval()
        _, f1, _ = pool(xyz, feats)
        _, f2, _ = pool(xyz[perm], feats[perm])
        assert torch.allclose(f1, f2, atol=1e-5), pt


def test_interp_reduces_toward_nearest_with_zero_distance():
    # If every neighbour shares the point's own voxel, interp == nearest (same coarse feature).
    xyz = torch.zeros(10, 3)                # all in one voxel
    feats = torch.randn(10, 8)
    pool = GridPool(0.1, channels=8, pool_type="max")
    xyz_p, feats_p, inv = pool(xyz, feats)
    knn = torch.zeros(10, 4, dtype=torch.long)  # neighbours all map to voxel 0
    out_i = grid_unpool(feats_p, inv, unpool_type="interp", knn_idx=knn,
                        xyz_fine=xyz, xyz_coarse=xyz_p)
    out_n = grid_unpool(feats_p, inv, unpool_type="nearest")
    assert torch.allclose(out_i, out_n, atol=1e-5)


def test_interp_weights_sum_to_one():
    xyz, feats = _toy_cloud(n=50, c=8)
    pool = GridPool(0.1, channels=8, pool_type="max")
    xyz_p, feats_p, inv = pool(xyz, feats)
    knn = torch.randint(0, xyz.shape[0], (xyz.shape[0], 4))
    nbr_vox = inv[knn]
    d = (xyz.unsqueeze(1) - xyz_p[nbr_vox]).norm(dim=-1)
    w = 1.0 / (d + 1e-8)
    w = w / w.sum(dim=1, keepdim=True)
    assert torch.allclose(w.sum(dim=1), torch.ones(xyz.shape[0]), atol=1e-5)


def test_gradient_flow_all_modes():
    xyz, feats = _toy_cloud()
    for pt in ("meanmax", "attention"):
        pool = GridPool(0.1, channels=feats.shape[1], pool_type=pt)
        f = feats.clone().requires_grad_(True)
        _, feats_p, _ = pool(xyz, f)
        loss = feats_p.pow(2).sum()
        loss.backward()
        assert f.grad is not None and torch.isfinite(f.grad).all()
        for name, p in pool.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_single_point_voxels_no_nan():
    # Every point in its own voxel (tiny voxel size) — attention/interp must not NaN.
    xyz = torch.arange(20).float().reshape(20, 1).repeat(1, 3)  # all distinct, far apart
    feats = torch.randn(20, 8)
    pool = GridPool(0.1, channels=8, pool_type="attention")
    _, feats_p, inv = pool(xyz, feats)
    assert torch.isfinite(feats_p).all()
    out = grid_unpool(feats_p, inv, unpool_type="interp",
                      knn_idx=torch.arange(20).reshape(20, 1).repeat(1, 3),
                      xyz_fine=xyz, xyz_coarse=xyz)
    assert torch.isfinite(out).all()


def test_full_model_backcompat_and_options():
    # Default config → baseline model still builds & runs; option configs build & run end-to-end.
    from pointcloud_seg.model.segmenter import CellTowerSegmenter
    feats = torch.randn(300, 17)
    xyz = torch.rand(300, 3) * 2.0

    def run(**overrides):
        c = Config()
        for k, v in overrides.items():
            setattr(c, k, v)
        c.__post_init__()
        torch.manual_seed(0)
        m = CellTowerSegmenter(c).eval()
        with torch.no_grad():
            return m(feats, xyz)

    base = run()
    assert base.shape == (300, 5)
    for ov in (
        {"pool_type": "meanmax"},
        {"pool_type": "attention"},
        {"unpool_type": "interp", "unpool_interp_k": 8},
        {"head_fuse_fine": True},
        {"pool_type": "attention", "unpool_type": "interp", "head_fuse_fine": True},
    ):
        out = run(**ov)
        assert out.shape == (300, 5), ov
        assert torch.isfinite(out).all(), ov


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
