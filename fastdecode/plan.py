"""Per-step execution plan: which tokens/requests go where.

Block ids < G live in the GPU KV cache, ids >= G live in the CPU store (at
index id - G). Every request in the batch falls in exactly one class:

  P  all of its KV blocks are on the GPU      -> normal paged FlashAttention
  S  decode (query_len == 1) with >=1 CPU block -> GPU partial + CPU partial,
                                                   merged by log-sum-exp
  F  fresh prefill (no prior context) with CPU  -> FlashAttention over the new
     slots                                         q/k/v (no cache read needed)

A request with query_len > 1 *and* prior context *and* CPU blocks (chunked-prefill
continuation or a prefix-cache hit) is not supported and raises; run with
--no-enable-prefix-caching and a max_num_batched_tokens >= max_model_len.

Everything here runs on CPU tensors so it can be unit-tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HostPlan:
    n_tok: int
    # KV writes
    gpu_slot_mapping: torch.Tensor  # [n_tok] int64, CPU slots replaced by -1
    cpu_w_tok: torch.Tensor  # [nw] int64 token rows whose KV goes to the CPU store
    cpu_w_slot: torch.Tensor  # [nw] int64 CPU-relative slot ids
    # P
    p_tok: torch.Tensor  # [nP_tok] int64
    p_cu: torch.Tensor  # [nP + 1] int32
    p_seqlens: torch.Tensor  # [nP] int32
    p_req: torch.Tensor  # [nP] int64 rows of the block table
    p_max_q: int
    p_max_k: int
    # S
    s_tok: torch.Tensor  # [nS] int64
    s_gpu_bt: torch.Tensor  # [nS, Wg] int32 (GPU block ids, compacted)
    s_gpu_lens: torch.Tensor  # [nS] int32
    s_max_gpu_len: int
    s_cpu_bt: torch.Tensor  # [nS, Wc] int32 (CPU-relative ids, compacted)
    s_cpu_lens: torch.Tensor  # [nS] int32
    # F
    f_tok: torch.Tensor  # [nF_tok] int64
    f_cu: torch.Tensor  # [nF + 1] int32
    f_max: int

    @property
    def n_s(self) -> int:
        return int(self.s_tok.numel())

    @property
    def n_p_tok(self) -> int:
        return int(self.p_tok.numel())

    @property
    def n_f_tok(self) -> int:
        return int(self.f_tok.numel())


def _ranges(starts: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """concat(arange(s, s + l) for s, l in zip(starts, lens)) without a loop."""
    total = int(lens.sum())
    if total == 0:
        return torch.zeros(0, dtype=torch.int64)
    offs = torch.cumsum(lens, 0) - lens
    within = torch.arange(total, dtype=torch.int64) - torch.repeat_interleave(offs, lens)
    return torch.repeat_interleave(starts, lens) + within


def _cu(lens: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(lens.numel() + 1, dtype=torch.int32)
    if lens.numel():
        out[1:] = torch.cumsum(lens, 0).to(torch.int32)
    return out


def _compact(bt: torch.Tensor, keep: torch.Tensor, offset: int, pad_to: int = 1) -> torch.Tensor:
    """Move the kept block ids of each row to the front (stable), minus offset.

    The width is rounded up to a multiple of `pad_to` (padding ids are 0).
    FlashAttention resolves page addresses for a whole KV tile (up to 256
    tokens = 16 pages) before masking rows past seqused_k, so it reads table
    entries beyond a row's last block; padding keeps those reads inside the row.
    """
    if bt.numel() == 0:
        return torch.zeros(bt.shape[0], pad_to, dtype=torch.int32)
    order = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices
    vals = torch.gather(bt, 1, order) - offset
    kept = torch.gather(keep, 1, order)
    vals = torch.where(kept, vals, torch.zeros_like(vals))
    width = max(1, int(keep.sum(1).max()))
    width = -(-width // pad_to) * pad_to
    if width > vals.shape[1]:
        vals = torch.nn.functional.pad(vals, (0, width - vals.shape[1]))
    return vals[:, :width].to(torch.int32).contiguous()


def build_host_plan(
    query_start_loc: torch.Tensor,  # [R + 1] (cpu)
    seq_lens: torch.Tensor,  # [R] (cpu), total tokens incl. this step's
    block_table: torch.Tensor,  # [R, W] int32 (cpu), W >= ceil(max seq/bs)
    slot_mapping: torch.Tensor,  # [n_tok] int64 (cpu)
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    block_size: int,
) -> HostPlan | None:
    G, bs = num_gpu_blocks, block_size
    qsl = query_start_loc.to(torch.int64)
    seq_lens = seq_lens.to(torch.int64)
    qlens = qsl[1:] - qsl[:-1]
    ctx = seq_lens - qlens
    n_tok = int(qsl[-1]) if qsl.numel() else 0
    slots = slot_mapping[:n_tok].to(torch.int64)

    nblk = (seq_lens + bs - 1) // bs
    W = block_table.shape[1]
    col = torch.arange(W, dtype=torch.int64)
    valid = col[None, :] < nblk[:, None]
    bt = block_table.to(torch.int64)
    on_cpu = (bt >= G) & valid
    on_gpu = (bt < G) & valid
    req_has_cpu = on_cpu.any(1)
    cpu_slot = slots >= G * bs

    if not bool(req_has_cpu.any()) and not bool(cpu_slot.any()):
        return None  # pure-GPU step: use plain FlashAttention

    if bool(on_cpu.any()) and int(bt[on_cpu].max()) >= G + num_cpu_blocks:
        raise RuntimeError(
            "FastDecode: block id beyond GPU+CPU capacity; engine and worker "
            "disagree on FASTDECODE_CPU_BLOCKS / FASTDECODE_CPU_KV_GB"
        )

    active = qlens > 0
    is_s = active & req_has_cpu & (qlens == 1)
    is_f = active & req_has_cpu & (qlens > 1) & (ctx == 0)
    bad = active & req_has_cpu & (qlens > 1) & (ctx > 0)
    if bool(bad.any()):
        raise NotImplementedError(
            "FastDecode: a request with prior context is being extended by >1 token "
            "while part of its KV is on the CPU (chunked prefill or prefix-cache hit). "
            "Run with enable_prefix_caching=False and "
            "max_num_batched_tokens >= max_model_len."
        )
    is_p = active & ~req_has_cpu

    gpu_slot_mapping = torch.where(cpu_slot, torch.full_like(slots, -1), slots)
    cpu_w_tok = torch.nonzero(cpu_slot).flatten()
    cpu_w_slot = slots[cpu_w_tok] - G * bs

    # P: all-GPU requests (any query length)
    p_req = torch.nonzero(is_p).flatten()
    p_q = qlens[p_req]
    p_tok = _ranges(qsl[p_req], p_q)
    p_seqlens = seq_lens[p_req].to(torch.int32)

    # S: decodes with CPU blocks
    s_req = torch.nonzero(is_s).flatten()
    s_tok = qsl[s_req]
    last_len = seq_lens - (nblk - 1) * bs
    tpb = torch.where(col[None, :] == (nblk - 1)[:, None], last_len[:, None], torch.full_like(bt, bs))
    tpb = torch.where(valid, tpb, torch.zeros_like(tpb))
    s_bt, s_gpu, s_cpu, s_tpb = bt[s_req], on_gpu[s_req], on_cpu[s_req], tpb[s_req]
    s_gpu_lens = (s_tpb * s_gpu).sum(1).to(torch.int32)
    s_cpu_lens = (s_tpb * s_cpu).sum(1).to(torch.int32)
    s_gpu_bt = _compact(s_bt, s_gpu, 0, pad_to=16)
    s_cpu_bt = _compact(s_bt, s_cpu, G)

    # F: fresh prefills touching CPU slots
    f_req = torch.nonzero(is_f).flatten()
    f_q = qlens[f_req]
    f_tok = _ranges(qsl[f_req], f_q)

    return HostPlan(
        n_tok=n_tok,
        gpu_slot_mapping=gpu_slot_mapping,
        cpu_w_tok=cpu_w_tok,
        cpu_w_slot=cpu_w_slot.contiguous(),
        p_tok=p_tok,
        p_cu=_cu(p_q),
        p_seqlens=p_seqlens,
        p_req=p_req,
        p_max_q=int(p_q.max()) if p_q.numel() else 0,
        p_max_k=int(p_seqlens.max()) if p_seqlens.numel() else 0,
        s_tok=s_tok,
        s_gpu_bt=s_gpu_bt,
        s_gpu_lens=s_gpu_lens,
        s_max_gpu_len=int(s_gpu_lens.max()) if s_gpu_lens.numel() else 0,
        s_cpu_bt=s_cpu_bt,
        s_cpu_lens=s_cpu_lens.contiguous(),
        f_tok=f_tok,
        f_cu=_cu(f_q),
        f_max=int(f_q.max()) if f_q.numel() else 0,
    )


@dataclass
class DevicePlan:
    """HostPlan with the tensors the GPU needs moved to the device."""

    host: HostPlan
    gpu_slot_mapping: torch.Tensor
    cpu_w_tok: torch.Tensor
    p_tok: torch.Tensor
    p_cu: torch.Tensor
    p_seqlens: torch.Tensor
    p_bt: torch.Tensor  # [nP, W] int32 (rows of the batch block table)
    s_tok: torch.Tensor
    # GPU partial of split decodes: only the S rows that have GPU tokens
    # (FlashAttention must never see a row with seqused_k == 0, see backend)
    s_gpu_rows: torch.Tensor  # [nG] int64, indices into the S rows
    s_gpu_cu: torch.Tensor  # arange(nG + 1) int32
    s_gpu_bt: torch.Tensor  # [nG, Wg] int32
    s_gpu_lens: torch.Tensor  # [nG] int32, all > 0
    f_tok: torch.Tensor
    f_cu: torch.Tensor


def to_device(hp: HostPlan, block_table_dev: torch.Tensor, device) -> DevicePlan:
    def d(t: torch.Tensor) -> torch.Tensor:
        return t.to(device, non_blocking=False)

    p_req = d(hp.p_req)
    g_rows = torch.nonzero(hp.s_gpu_lens > 0).flatten()
    return DevicePlan(
        host=hp,
        gpu_slot_mapping=d(hp.gpu_slot_mapping),
        cpu_w_tok=d(hp.cpu_w_tok),
        p_tok=d(hp.p_tok),
        p_cu=d(hp.p_cu),
        p_seqlens=d(hp.p_seqlens),
        p_bt=block_table_dev.index_select(0, p_req) if hp.p_req.numel() else block_table_dev[:0],
        s_tok=d(hp.s_tok),
        s_gpu_rows=d(g_rows),
        s_gpu_cu=torch.arange(g_rows.numel() + 1, dtype=torch.int32, device=device),
        s_gpu_bt=d(hp.s_gpu_bt[g_rows]),
        s_gpu_lens=d(hp.s_gpu_lens[g_rows]),
        f_tok=d(hp.f_tok),
        f_cu=d(hp.f_cu),
    )
