"""Loader for the C++ CPU kernels, plus pure-PyTorch reference versions.

The extension is JIT-compiled with torch.utils.cpp_extension on first use and
cached under ~/.cache/torch_extensions, so the first engine start takes ~30 s
longer. Set FASTDECODE_MARCH to override `-march=native` (e.g. "x86-64-v3" when
building on one machine and running on another).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

_EXT = None
_LOCK = threading.Lock()


def load_ext(verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load

        src = Path(__file__).resolve().parent / "csrc" / "cpu_attn.cpp"
        cflags = ["-O3", "-fopenmp", "-std=c++17"]
        march = os.environ.get("FASTDECODE_MARCH", "native")
        if march:
            cflags.append(f"-march={march}")
        name = "fastdecode_cpu" + ("" if march == "native" else "_" + march.replace("-", "_"))
        _EXT = load(
            name=name,
            sources=[str(src)],
            extra_cflags=cflags,
            extra_ldflags=["-fopenmp"],
            verbose=verbose,
        )
        nt = os.environ.get("FASTDECODE_CPU_THREADS")
        if nt:
            _EXT.set_num_threads(int(nt))
        return _EXT


# ----------------------------------------------------------------------------
# Reference implementations (float64 math; for tests only)
# ----------------------------------------------------------------------------


def ref_decode_attn(
    q: torch.Tensor,  # [n, H, D]
    k_cache: torch.Tensor,  # [C, KVH, BS, D]
    v_cache: torch.Tensor,
    block_table: torch.Tensor,  # [n, W] int32
    lens: torch.Tensor,  # [n] int32
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, H, D = q.shape
    _, KVH, BS, _ = k_cache.shape
    G = H // KVH
    out = torch.zeros(n, H, D, dtype=torch.float64)
    lse = torch.full((H, n), float("-inf"), dtype=torch.float64)
    for i in range(n):
        L = int(lens[i])
        if L == 0:
            continue
        nb = (L + BS - 1) // BS
        blks = block_table[i, :nb].long()
        # [nb, KVH, BS, D] -> [KVH, nb*BS, D] -> first L tokens
        k = k_cache[blks].permute(1, 0, 2, 3).reshape(KVH, nb * BS, D)[:, :L].double()
        v = v_cache[blks].permute(1, 0, 2, 3).reshape(KVH, nb * BS, D)[:, :L].double()
        k = k.repeat_interleave(G, dim=0)  # [H, L, D]
        v = v.repeat_interleave(G, dim=0)
        s = torch.einsum("hd,hld->hl", q[i].double(), k) * scale
        lse[:, i] = torch.logsumexp(s, dim=-1)
        out[i] = torch.einsum("hl,hld->hd", torch.softmax(s, dim=-1), v)
    return out, lse


def ref_store_kv(k, v, slots, k_cache, v_cache) -> None:
    BS = k_cache.shape[2]
    blk = slots // BS
    off = slots % BS
    # advanced indices separated by a slice -> result dims are [m, KVH, D]
    k_cache[blk, :, off] = k
    v_cache[blk, :, off] = v


def merge_partials_torch(
    o_a: torch.Tensor,  # [n, H, D]
    lse_a: torch.Tensor,  # [H, n]; +inf or -inf means "empty"
    o_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Same rule as vllm.v1.attention.ops.merge_attn_states (torch fallback)."""
    pos_inf = float("inf")
    neg_inf = float("-inf")
    lse_a = torch.where(lse_a == pos_inf, torch.full_like(lse_a, neg_inf), lse_a).float()
    lse_b = torch.where(lse_b == pos_inf, torch.full_like(lse_b, neg_inf), lse_b).float()
    m = torch.maximum(lse_a, lse_b)
    both_empty = m == neg_inf
    m = torch.where(both_empty, torch.zeros_like(m), m)
    wa = torch.exp(lse_a - m)
    wb = torch.exp(lse_b - m)
    tot = wa + wb
    tot = torch.where(both_empty, torch.ones_like(tot), tot)
    wa = (wa / tot).t().unsqueeze(-1)
    wb = (wb / tot).t().unsqueeze(-1)
    return o_a.float() * wa + o_b.float() * wb
