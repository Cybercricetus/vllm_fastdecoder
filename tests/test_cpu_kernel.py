"""CPU-only tests for the C++ kernels (no GPU needed).

    python -m pytest tests/test_cpu_kernel.py -q
"""

import math

import pytest
import torch

from fastdecode import cpu_ops


def _rand_case(n, H, KVH, D, BS, C, dtype, max_len, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, H, D, generator=g).to(dtype)
    k_cache = torch.randn(C, KVH, BS, D, generator=g).to(dtype)
    v_cache = torch.randn(C, KVH, BS, D, generator=g).to(dtype)
    lens = torch.randint(0, max_len + 1, (n,), generator=g, dtype=torch.int32)
    lens[0] = max_len  # exercise the widest row
    if n > 1:
        lens[1] = 1  # single token
    W = (max_len + BS - 1) // BS
    # distinct random blocks per request, like a real allocator
    perm = torch.randperm(C, generator=g)
    bt = torch.zeros(n, W, dtype=torch.int32)
    pos = 0
    for i in range(n):
        nb = (int(lens[i]) + BS - 1) // BS
        if pos + nb > C:
            perm = torch.randperm(C, generator=g)
            pos = 0
        bt[i, :nb] = perm[pos : pos + nb].to(torch.int32)
        pos += nb
    return q, k_cache, v_cache, bt, lens


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("H,KVH", [(8, 8), (8, 2), (4, 1)])
@pytest.mark.parametrize("D", [64, 128, 72])
def test_decode_attn_matches_reference(dtype, H, KVH, D):
    ext = cpu_ops.load_ext()
    n, BS, C = 7, 16, 64
    q, kc, vc, bt, lens = _rand_case(n, H, KVH, D, BS, C, dtype, max_len=93, seed=D + H)
    lens[3] = 0  # empty row -> zeros and lse=-inf
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    lse = torch.empty(H, n, dtype=torch.float32)
    ext.decode_attn(q, kc, vc, bt, lens, scale, out, lse)
    ref_out, ref_lse = cpu_ops.ref_decode_attn(q, kc, vc, bt, lens, scale)

    nz = lens > 0
    tol = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(out[nz].double(), ref_out[nz], atol=tol, rtol=tol)
    torch.testing.assert_close(lse[:, nz].double(), ref_lse[:, nz], atol=1e-4, rtol=1e-5)
    assert torch.all(out[~nz] == 0)
    assert torch.all(torch.isneginf(lse[:, ~nz]))


def test_decode_attn_large_scores():
    """Large logits must not overflow (max subtraction)."""
    ext = cpu_ops.load_ext()
    n, H, KVH, D, BS, C = 2, 4, 4, 64, 16, 8
    q, kc, vc, bt, lens = _rand_case(n, H, KVH, D, BS, C, torch.float32, 40, 0)
    q = q * 300.0
    out = torch.empty_like(q)
    lse = torch.empty(H, n, dtype=torch.float32)
    ext.decode_attn(q, kc, vc, bt, lens, 1.0, out, lse)
    ref_out, ref_lse = cpu_ops.ref_decode_attn(q, kc, vc, bt, lens, 1.0)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(out.double(), ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse.double(), ref_lse, atol=1e-3, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_store_kv_matches_reference(dtype):
    ext = cpu_ops.load_ext()
    C, KVH, BS, D, m = 10, 4, 16, 128, 37
    g = torch.Generator().manual_seed(1)
    kc = torch.zeros(C, KVH, BS, D, dtype=dtype)
    vc = torch.zeros_like(kc)
    kc_ref, vc_ref = kc.clone(), vc.clone()
    k = torch.randn(m, KVH, D, generator=g).to(dtype)
    v = torch.randn(m, KVH, D, generator=g).to(dtype)
    slots = torch.randperm(C * BS, generator=g)[:m].to(torch.int64)
    ext.store_kv(k, v, slots, kc, vc)
    cpu_ops.ref_store_kv(k, v, slots, kc_ref, vc_ref)
    assert torch.equal(kc, kc_ref) and torch.equal(vc, vc_ref)


def test_store_kv_rejects_out_of_range():
    ext = cpu_ops.load_ext()
    kc = torch.zeros(2, 1, 16, 64, dtype=torch.float16)
    k = torch.zeros(1, 1, 64, dtype=torch.float16)
    with pytest.raises(RuntimeError):
        ext.store_kv(k, k, torch.tensor([32], dtype=torch.int64), kc, kc.clone())


def test_split_and_merge_equals_full_attention():
    """Core FastDecode invariant: attention over all tokens == LSE-merge of
    attention over any partition of the tokens into two sets."""
    ext = cpu_ops.load_ext()
    n, H, KVH, D, BS, C = 5, 8, 2, 128, 16, 64
    q, kc, vc, bt, lens = _rand_case(n, H, KVH, D, BS, C, torch.float16, 80, 7)
    lens.clamp_(min=1)
    scale = D ** -0.5
    full_out, full_lse = cpu_ops.ref_decode_attn(q, kc, vc, bt, lens, scale)

    # split each request's blocks: even block positions -> "GPU", odd -> "CPU"
    W = bt.shape[1]
    bt_a = torch.zeros_like(bt)
    bt_b = torch.zeros_like(bt)
    len_a = torch.zeros_like(lens)
    len_b = torch.zeros_like(lens)
    for i in range(n):
        L = int(lens[i])
        nb = (L + BS - 1) // BS
        a_blocks, b_blocks = [], []
        la = lb = 0
        for j in range(nb):
            ntok = BS if j < nb - 1 else L - (nb - 1) * BS
            if j % 2 == 0:
                a_blocks.append(int(bt[i, j]))
                la += ntok
            else:
                b_blocks.append(int(bt[i, j]))
                lb += ntok
        bt_a[i, : len(a_blocks)] = torch.tensor(a_blocks, dtype=torch.int32)
        if b_blocks:
            bt_b[i, : len(b_blocks)] = torch.tensor(b_blocks, dtype=torch.int32)
        len_a[i], len_b[i] = la, lb

    o_a, lse_a = cpu_ops.ref_decode_attn(q, kc, vc, bt_a, len_a, scale)
    o_b = torch.empty_like(q)
    lse_b = torch.empty(H, n, dtype=torch.float32)
    ext.decode_attn(q, kc, vc, bt_b, len_b, scale, o_b, lse_b)
    merged = cpu_ops.merge_partials_torch(o_a.half(), lse_a.float(), o_b, lse_b)
    torch.testing.assert_close(merged.double(), full_out, atol=2e-3, rtol=2e-3)


def test_merge_handles_fa_empty_convention():
    """FlashAttention reports +inf LSE for an empty row; merge must ignore it."""
    o_a = torch.zeros(3, 2, 8)
    lse_a = torch.full((2, 3), float("inf"))
    o_b = torch.randn(3, 2, 8)
    lse_b = torch.randn(2, 3)
    torch.testing.assert_close(cpu_ops.merge_partials_torch(o_a, lse_a, o_b, lse_b), o_b)
