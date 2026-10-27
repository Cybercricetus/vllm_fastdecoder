"""CPU-only tests of the planner and of FastDecodeImpl._forward_split.

The GPU primitives (FlashAttention, reshape_and_cache_flash, merge_attn_states,
CUDA events) are replaced by straightforward PyTorch reference versions, so
everything except the CUDA kernels themselves -- block routing, KV writes to
both tiers, index bookkeeping, LSE layout and merging -- runs exactly as on the
GPU and is checked against plain attention over a single unified KV cache.

    python -m pytest tests/test_plan_and_forward.py -q
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch

from fastdecode import backend, cpu_ops, plan, state

# ----------------------------------------------------------------------------
# CPU mocks of the GPU ops
# ----------------------------------------------------------------------------


def _attend(q, k, v, scale, causal):
    """q [Lq, H, D], k/v [Lk, KVH, D] -> out [Lq, H, D] (float64), lse [H, Lq]."""
    H, KVH = q.shape[1], k.shape[1]
    G = H // KVH
    if k.shape[0] == 0:
        return (torch.zeros(q.shape, dtype=torch.float64),
                torch.full((H, q.shape[0]), float("inf"), dtype=torch.float64))  # FA convention
    k = k.double().repeat_interleave(G, dim=1)
    v = v.double().repeat_interleave(G, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.double(), k) * scale
    if causal:
        Lq, Lk = q.shape[0], k.shape[0]
        qi = torch.arange(Lq)[:, None] + (Lk - Lq)  # bottom-right aligned
        kj = torch.arange(Lk)[None, :]
        s = s.masked_fill((kj > qi)[None], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)  # [H, Lq]
    out = torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), v)
    return out, lse


class CpuOps:
    @staticmethod
    def fa(q, k, v, cu_seqlens_q, max_seqlen_q, max_seqlen_k, softmax_scale,
           causal, cu_seqlens_k=None, seqused_k=None, block_table=None,
           return_softmax_lse=False, **_):
        n = cu_seqlens_q.numel() - 1
        outs, lses = [], []
        for b in range(n):
            q0, q1 = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
            if block_table is not None:
                L = int(seqused_k[b])
                bs = k.shape[1]
                nb = (L + bs - 1) // bs
                blks = block_table[b, :nb].long()
                kk = k[blks].reshape(-1, k.shape[2], k.shape[3])[:L]
                vv = v[blks].reshape(-1, v.shape[2], v.shape[3])[:L]
            else:
                k0, k1 = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
                kk, vv = k[k0:k1], v[k0:k1]
            o, l = _attend(q[q0:q1], kk, vv, softmax_scale, causal)
            outs.append(o)
            lses.append(l)
        out = torch.cat(outs).to(q.dtype)
        lse = torch.cat(lses, dim=1).float()
        return (out, lse) if return_softmax_lse else out

    @staticmethod
    def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, *_):
        bs = key_cache.shape[1]
        for i in range(slot_mapping.numel()):
            s = int(slot_mapping[i])
            if s < 0:
                continue
            key_cache[s // bs, s % bs] = key[i]
            value_cache[s // bs, s % bs] = value[i]

    @staticmethod
    def merge(out, o_a, lse_a, o_b, lse_b):
        out.copy_(cpu_ops.merge_partials_torch(o_a, lse_a, o_b, lse_b).to(out.dtype))

    @staticmethod
    def event():
        return SimpleNamespace(synchronize=lambda: None)


@pytest.fixture(autouse=True)
def _cpu_mode(monkeypatch):
    monkeypatch.setattr(backend, "OPS", CpuOps)
    st = state.PinnedStaging.__new__(state.PinnedStaging)
    st._bufs, st._pin = {}, False
    monkeypatch.setattr(state, "_STAGING", st)
    yield
    state.STORE = None


# ----------------------------------------------------------------------------
# Scenario: requests with explicit block lists over a unified id space
# ----------------------------------------------------------------------------

BS = 4


def _scenario(reqs, G, C, H, KVH, D, dtype, seed):
    """reqs: list of (ctx_len, q_len, block_ids). Returns everything a forward
    call needs plus the reference output."""
    g = torch.Generator().manual_seed(seed)
    # unified logical cache over ids [0, G + C): [N, BS, KVH, D]
    uk = torch.randn(G + C, BS, KVH, D, generator=g).to(dtype)
    uv = torch.randn(G + C, BS, KVH, D, generator=g).to(dtype)
    n_tok = sum(r[1] for r in reqs)
    query = torch.randn(n_tok, H, D, generator=g).to(dtype)
    key = torch.randn(n_tok, KVH, D, generator=g).to(dtype)
    value = torch.randn(n_tok, KVH, D, generator=g).to(dtype)

    qsl, seq_lens, slots = [0], [], []
    W = max(len(r[2]) for r in reqs)
    bt = torch.zeros(len(reqs), W, dtype=torch.int32)
    for i, (ctx, ql, blocks) in enumerate(reqs):
        L = ctx + ql
        assert len(blocks) == (L + BS - 1) // BS
        bt[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        for p in range(ctx, L):
            slots.append(blocks[p // BS] * BS + p % BS)
        qsl.append(qsl[-1] + ql)
        seq_lens.append(L)
    qsl = torch.tensor(qsl, dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    slots = torch.tensor(slots, dtype=torch.int64)

    # reference: write new tokens into the unified cache, attend causally
    rk, rv = uk.clone(), uv.clone()
    for t, s in enumerate(slots.tolist()):
        rk[s // BS, s % BS] = key[t]
        rv[s // BS, s % BS] = value[t]
    scale = 1.0 / math.sqrt(D)
    ref = torch.zeros(n_tok, H, D, dtype=torch.float64)
    for i, (ctx, ql, blocks) in enumerate(reqs):
        L = ctx + ql
        kk = rk[torch.tensor(blocks)].reshape(-1, KVH, D)[:L]
        vv = rv[torch.tensor(blocks)].reshape(-1, KVH, D)[:L]
        o, _ = _attend(query[qsl[i] : qsl[i + 1]], kk, vv, scale, causal=True)
        ref[qsl[i] : qsl[i + 1]] = o

    # split the unified cache into the two tiers in their real layouts
    kv_cache = torch.zeros(G, KVH, BS, 2 * D, dtype=dtype)  # vLLM logical [B,H,N,2D]
    kv_cache[..., :D] = uk[:G].transpose(1, 2)
    kv_cache[..., D:] = uv[:G].transpose(1, 2)
    cpu_k = uk[G:].transpose(1, 2).contiguous()  # [C, KVH, BS, D]
    cpu_v = uv[G:].transpose(1, 2).contiguous()
    return SimpleNamespace(
        query=query, key=key, value=value, qsl=qsl, seq_lens=seq_lens, bt=bt,
        slots=slots, kv_cache=kv_cache, cpu_k=cpu_k, cpu_v=cpu_v, ref=ref,
        ref_k=rk, ref_v=rv, scale=scale,
    )


def _impl(H, KVH, D, scale):
    impl = backend.FastDecodeImpl.__new__(backend.FastDecodeImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = H, KVH, D
    impl.scale = scale
    impl.logits_soft_cap = 0
    impl.vllm_flash_attn_version = 2
    impl.kv_cache_dtype = "auto"
    impl.kv_sharing_target_layer_name = None
    return impl


REQS = [
    (8, 1, [1, 2, 12]),            # S: decode, last block on CPU
    (4, 1, [3, 4]),                # P: decode, all GPU
    (0, 6, [13, 5]),               # F: fresh prefill touching CPU
    (0, 3, [6]),                   # P: fresh prefill, all GPU
    (3, 1, [15]),                  # S: decode, no GPU tokens at all
    (11, 1, [16, 7, 17]),          # S: GPU block in the middle
    (0, 1, [18]),                  # S: single-token prompt on CPU
]


def test_plan_classification():
    G, C = 10, 20
    sc = _scenario(REQS, G, C, 4, 4, 16, torch.float32, 0)
    hp = plan.build_host_plan(sc.qsl, sc.seq_lens, sc.bt, sc.slots, G, C, BS)
    assert hp is not None
    assert hp.p_req.tolist() == [1, 3]
    assert hp.s_tok.tolist() == [0, 11, 12, 13]  # token rows of reqs 0, 4, 5, 6
    assert hp.f_cu.tolist() == [0, 6]
    assert hp.s_gpu_lens.tolist() == [8, 0, 4, 0]
    assert hp.s_cpu_lens.tolist() == [1, 4, 8, 1]
    assert hp.s_gpu_bt[2, 0].item() == 7
    assert hp.s_cpu_bt[2, :2].tolist() == [16 - G, 17 - G]
    # every CPU slot is masked out of the GPU write and routed to the store
    cpu_mask = sc.slots >= G * BS
    assert torch.equal(hp.gpu_slot_mapping[cpu_mask], torch.full((int(cpu_mask.sum()),), -1))
    assert torch.equal(hp.cpu_w_slot, sc.slots[cpu_mask] - G * BS)


def test_plan_none_when_all_gpu():
    reqs = [(4, 1, [1, 2]), (0, 3, [3])]
    sc = _scenario(reqs, 10, 5, 4, 4, 16, torch.float32, 0)
    assert plan.build_host_plan(sc.qsl, sc.seq_lens, sc.bt, sc.slots, 10, 5, BS) is None


def test_plan_rejects_chunked_continuation_on_cpu():
    reqs = [(4, 3, [1, 12])]  # ctx>0, q_len>1, touches CPU
    sc = _scenario(reqs, 10, 5, 4, 4, 16, torch.float32, 0)
    with pytest.raises(NotImplementedError):
        plan.build_host_plan(sc.qsl, sc.seq_lens, sc.bt, sc.slots, 10, 5, BS)


@pytest.mark.parametrize("H,KVH", [(4, 4), (8, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seed", [0, 1])
def test_forward_split_matches_unified_attention(H, KVH, dtype, seed):
    G, C, D = 10, 20, 64
    sc = _scenario(REQS, G, C, H, KVH, D, dtype, seed)
    hp = plan.build_host_plan(sc.qsl, sc.seq_lens, sc.bt, sc.slots, G, C, BS)
    dp = plan.to_device(hp, sc.bt, "cpu")
    state.STORE = state.CpuKVStore(
        num_gpu_blocks=G, num_cpu_blocks=C, block_size=BS, num_kv_heads=KVH,
        head_size=D, dtype=dtype, k={"l0": sc.cpu_k}, v={"l0": sc.cpu_v},
    )
    impl = _impl(H, KVH, D, sc.scale)
    layer = SimpleNamespace(layer_name="l0", _k_scale=None, _v_scale=None)
    output = torch.full_like(sc.query, float("nan"))
    impl._forward_split(layer, sc.query, sc.key, sc.value, sc.kv_cache, dp, output)

    tol = 1e-5 if dtype == torch.float32 else 3e-3
    torch.testing.assert_close(output.double(), sc.ref, atol=tol, rtol=tol)

    # both tiers hold exactly the unified reference cache afterwards
    D_ = D
    gk = sc.kv_cache[..., :D_].transpose(1, 2)
    gv = sc.kv_cache[..., D_:].transpose(1, 2)
    assert torch.equal(gk, sc.ref_k[:G]) and torch.equal(gv, sc.ref_v[:G])
    assert torch.equal(sc.cpu_k.transpose(1, 2), sc.ref_k[G:])
    assert torch.equal(sc.cpu_v.transpose(1, 2), sc.ref_v[G:])


def test_engine_core_patch(monkeypatch):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    from fastdecode import patches

    spec = FullAttentionSpec(block_size=16, num_kv_heads=32, head_size=128, dtype=torch.float16)
    cfg = KVCacheConfig(
        num_blocks=100, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[f"l{i}" for i in range(32)], kv_cache_spec=spec)],
    )
    monkeypatch.setattr(EngineCore, "_initialize_kv_caches", lambda self, vc: cfg)
    monkeypatch.setenv("FASTDECODE_CPU_KV_GB", "2")
    patches.apply_engine_core_patch()
    vc = SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
        parallel_config=SimpleNamespace(world_size=1),
    )
    out = EngineCore._initialize_kv_caches(None, vc)
    per_block = 2 * 32 * 16 * 128 * 2 * 32  # K+V, heads, tokens, dim, fp16, layers
    assert out.num_blocks == 100 + (2 << 30) // per_block == 100 + 256
    assert cfg.num_blocks == 100  # the workers' config is untouched

    vc.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="prefix_caching"):
        EngineCore._initialize_kv_caches(None, vc)


def test_ensure_store_matches_engine_side(monkeypatch):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from fastdecode import config

    monkeypatch.setenv("FASTDECODE_CPU_KV_GB", "0.01")
    monkeypatch.setattr(cpu_ops, "load_ext", lambda: SimpleNamespace(has_avx2=lambda: True, get_num_threads=lambda: 1))
    spec = FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=64, dtype=torch.float16)
    names = ["a", "b"]
    vc = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks=None))
    state.ensure_store(vc, spec, names)
    assert state.STORE is None  # profiling-time builder: no-op
    vc.cache_config.num_gpu_blocks = 50
    state.ensure_store(vc, spec, names)
    st = state.STORE
    assert st is not None and st.num_gpu_blocks == 50
    expected = int(0.01 * (1 << 30)) // (spec.page_size_bytes * len(names))
    assert st.num_cpu_blocks == expected
    assert st.k["a"].shape == (expected, 4, 16, 64)


def test_block_pool_hands_out_gpu_blocks_first(monkeypatch):
    """Drive vLLM's real BlockPool through allocate/free churn: a request must
    never receive a CPU block while a GPU block is free."""
    import random

    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue

    from fastdecode import patches

    G, C = 20, 30
    monkeypatch.setattr(FreeKVCacheBlockQueue, "prepend_n", FreeKVCacheBlockQueue.prepend_n)
    monkeypatch.setattr(patches, "_GPU_BLOCKS", G)
    patches._apply_gpu_first_free_queue()

    pool = BlockPool(num_gpu_blocks=G + C, enable_caching=False, hash_block_size=16)
    rng = random.Random(0)
    live: list[list] = []
    for _ in range(2000):
        if live and (rng.random() < 0.45 or pool.get_num_free_blocks() < 4):
            blocks = live.pop(rng.randrange(len(live)))
            pool.free_blocks(reversed(blocks))
            continue
        n = rng.randint(1, 4)
        if n > pool.get_num_free_blocks():
            continue
        free_gpu_before = sum(
            1 for b in pool.free_block_queue.get_all_free_blocks() if b.block_id < G
        )
        got = pool.get_new_blocks(n)
        n_gpu = sum(1 for b in got if b.block_id < G)
        assert n_gpu == min(n, free_gpu_before), (n, free_gpu_before, [b.block_id for b in got])
        live.append(got)


@pytest.mark.parametrize("with_cpu_seq_lens", [True, False])
def test_builder_build_then_forward(monkeypatch, with_cpu_seq_lens):
    """FastDecodeMetadataBuilder.build on a real CommonAttentionMetadata with
    runner-style padding (extra block-table columns, padded slot mapping)."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

    monkeypatch.setattr(FlashAttentionMetadataBuilder, "build",
                        lambda self, cpl, cm, fast_build=False: SimpleNamespace())
    G, C, H, KVH, D, dtype = 10, 20, 8, 2, 64, torch.float16
    sc = _scenario(REQS, G, C, H, KVH, D, dtype, 3)
    state.STORE = state.CpuKVStore(
        num_gpu_blocks=G, num_cpu_blocks=C, block_size=BS, num_kv_heads=KVH,
        head_size=D, dtype=dtype, k={"l0": sc.cpu_k}, v={"l0": sc.cpu_v},
    )
    R, n_tok = sc.bt.shape[0], int(sc.qsl[-1])
    bt_wide = torch.zeros(R + 2, 64, dtype=torch.int32)  # runner buffers are wider
    bt_wide[:R, : sc.bt.shape[1]] = sc.bt
    slot_pad = torch.full((n_tok + 5,), -1, dtype=torch.int64)
    slot_pad[:n_tok] = sc.slots
    cm = CommonAttentionMetadata(
        query_start_loc=sc.qsl, query_start_loc_cpu=sc.qsl, seq_lens=sc.seq_lens,
        num_reqs=R, num_actual_tokens=n_tok + 5, max_query_len=6,
        max_seq_len=int(sc.seq_lens.max()), block_table_tensor=bt_wide,
        slot_mapping=slot_pad,
        seq_lens_cpu_upper_bound=sc.seq_lens if with_cpu_seq_lens else None,
    )
    b = backend.FastDecodeMetadataBuilder.__new__(backend.FastDecodeMetadataBuilder)
    b.kernel_block_size, b.device, b._acc = None, "cpu", [0] * 5
    md = b.build(0, cm)
    assert isinstance(md.fd_plan, plan.DevicePlan)

    impl = _impl(H, KVH, D, sc.scale)
    layer = SimpleNamespace(layer_name="l0", _k_scale=None, _v_scale=None)
    pad = 5  # padded rows like a piecewise-cudagraph batch
    q = torch.cat([sc.query, torch.zeros(pad, H, D, dtype=dtype)])
    k = torch.cat([sc.key, torch.zeros(pad, KVH, D, dtype=dtype)])
    v = torch.cat([sc.value, torch.zeros(pad, KVH, D, dtype=dtype)])
    out = torch.full_like(q, float("nan"))
    impl._forward_split(layer, q, k, v, sc.kv_cache, md.fd_plan, out)
    torch.testing.assert_close(out[:n_tok].double(), sc.ref, atol=3e-3, rtol=3e-3)


def test_capacity_check_counts_cpu_tier(monkeypatch):
    """vLLM's 'one max_model_len request must fit' check must count CPU blocks."""
    from functools import partial

    import vllm.v1.core.kv_cache_utils as kvu
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    from fastdecode import patches

    monkeypatch.setattr(kvu, "_check_enough_kv_cache_memory", kvu._check_enough_kv_cache_memory)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=32, head_size=128, dtype=torch.float16)
    groups = [KVCacheGroupSpec(layer_names=[f"l{i}" for i in range(32)], kv_cache_spec=spec)]
    per_block = kvu._pool_bytes_per_block(groups)  # 8 MiB for this spec
    needed_blocks = 2048 // 16

    def need(_cfg, _groups):
        return needed_blocks * per_block

    get_needed = partial(need, None, groups)
    gpu_mem = 32 * per_block  # GPU tier holds only 512 tokens
    with pytest.raises(ValueError):
        kvu._check_enough_kv_cache_memory(gpu_mem, get_needed, 2048, lambda m: 0)

    monkeypatch.setenv("FASTDECODE_CPU_KV_GB", "2")  # +256 blocks
    patches._apply_capacity_check_patch()
    kvu._check_enough_kv_cache_memory(gpu_mem, get_needed, 2048, lambda m: 0)  # passes now


def test_verifier_clean_run_and_detects_corruption(monkeypatch):
    """FASTDECODE_VERIFY's online check: silent on a correct step, loud when the
    CPU-tier KV or the output is corrupted."""
    from fastdecode import verify

    G, C, H, KVH, D, dtype = 10, 20, 8, 2, 64, torch.float16
    sc = _scenario(REQS, G, C, H, KVH, D, dtype, 5)
    state.STORE = state.CpuKVStore(
        num_gpu_blocks=G, num_cpu_blocks=C, block_size=BS, num_kv_heads=KVH,
        head_size=D, dtype=dtype, k={"l0": sc.cpu_k}, v={"l0": sc.cpu_v},
    )
    R, n_tok = sc.bt.shape[0], int(sc.qsl[-1])
    md = SimpleNamespace(
        query_start_loc=sc.qsl, seq_lens=sc.seq_lens, slot_mapping=sc.slots,
        block_table=sc.bt, num_actual_tokens=n_tok, fd_num_reqs=R,
        fd_seq_cpu=sc.seq_lens,
    )
    vf = verify.Verifier()
    # fingerprint the context KV as if earlier steps had written it
    ctx_slots, ctx_k, ctx_qsl, ctx_lens = [], [], [0], []
    for (ctx, _, blocks) in REQS:
        for p in range(ctx):
            s = blocks[p // BS] * BS + p % BS
            ctx_slots.append(s)
            ctx_k.append(sc.ref_k[s // BS, s % BS])
        ctx_qsl.append(ctx_qsl[-1] + ctx)
        ctx_lens.append(ctx)
    prev = SimpleNamespace(
        query_start_loc=torch.tensor(ctx_qsl), seq_lens=torch.tensor(ctx_lens),
        slot_mapping=torch.tensor(ctx_slots), num_actual_tokens=len(ctx_slots),
        fd_num_reqs=R,
    )
    vf.record("l0", torch.stack(ctx_k), prev)

    hp = plan.build_host_plan(sc.qsl, sc.seq_lens, sc.bt, sc.slots, G, C, BS)
    dp = plan.to_device(hp, sc.bt, "cpu")
    impl = _impl(H, KVH, D, sc.scale)
    layer = SimpleNamespace(layer_name="l0", _k_scale=None, _v_scale=None)
    out = torch.full_like(sc.query, float("nan"))
    impl._forward_split(layer, sc.query, sc.key, sc.value, sc.kv_cache, dp, out)
    vf.record("l0", sc.key, md)
    vf.check(impl, layer, sc.query, sc.key, sc.value, sc.kv_cache, md, out)
    assert vf.total_bad == 0, vf.cnt

    # corrupt one CPU-resident context position of request 5 (block 16 -> CPU idx 6)
    sc.cpu_k[16 - G, :, 1] += 1.0
    out[0] += 0.5  # and the output row of request 0
    vf.check(impl, layer, sc.query, sc.key, sc.value, sc.kv_cache, md, out)
    assert vf.cnt["ksum_mismatch"] >= 1
    assert vf.cnt["out_mismatch"] >= 1
