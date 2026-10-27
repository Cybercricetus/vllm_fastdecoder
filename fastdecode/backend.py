"""FastDecode attention backend for vLLM v0.31 (FlashAttention + CPU KV tier).

Registered as AttentionBackendEnum.CUSTOM; select it with
`--attention-backend CUSTOM` (or `LLM(..., attention_backend="CUSTOM")`).

With no CPU blocks configured, or in a step where no request touches a CPU
block, it runs exactly the stock FlashAttention path. Otherwise each layer:

  1. writes new K/V: GPU slots via reshape_and_cache_flash, CPU slots via a
     D2H copy + `store_kv` on the host;
  2. launches all GPU attention (all-GPU requests, the GPU part of split
     decodes, fresh prefills) and, concurrently, runs the CPU part of split
     decodes with the C++ kernel;
  3. merges the GPU and CPU partials of split decodes with merge_attn_states.
"""

from __future__ import annotations

import logging

import torch

from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends import flash_attn as _fa
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

from fastdecode import config, cpu_ops, sr, state, verify
from fastdecode.plan import DevicePlan, build_host_plan, to_device

logger = logging.getLogger("fastdecode")

# FASTDECODE_VERIFY=1: slow online self-check (see fastdecode/verify.py)
_VERIFIER = verify.Verifier() if verify.enabled() else None


class _CudaOps:
    """GPU primitives used by the split path (swapped for CPU mocks in tests)."""

    @staticmethod
    def fa(**kw):
        return _fa.flash_attn_varlen_func(**kw)

    @staticmethod
    def reshape_and_cache(*args):
        return _fa.reshape_and_cache_flash(*args)

    @staticmethod
    def merge(out, o_a, lse_a, o_b, lse_b):
        return _fa.merge_attn_states(out, o_a, lse_a, o_b, lse_b)

    @staticmethod
    def event():
        ev = torch.cuda.Event()
        ev.record()
        return ev


OPS = _CudaOps


class FastDecodeBackend(FlashAttentionBackend):
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]
    # The layer must not call the separate unified_kv_cache_update op: CPU-bound
    # K/V have to be written from inside forward(), which runs eagerly (it is a
    # piecewise-cudagraph splitting op).
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["FastDecodeImpl"]:
        return FastDecodeImpl

    @staticmethod
    def get_builder_cls() -> type["FastDecodeMetadataBuilder"]:
        return FastDecodeMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_rswa(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class FastDecodeMetadataBuilder(FlashAttentionMetadataBuilder):
    # The CPU work cannot be captured in a CUDA graph; vLLM downgrades
    # FULL_AND_PIECEWISE to PIECEWISE (attention runs eagerly between graphs).
    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table: bool = False
    supports_draft_decode_metadata_update: bool = False

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        return AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.supports_draft_decode_metadata_update = False
        state.ensure_store(vllm_config, kv_cache_spec, layer_names)
        self._n_builds = 0
        self._acc = [0, 0, 0, 0, 0]  # steps, P reqs, S reqs, F reqs, CPU tokens

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        cm = common_attn_metadata
        md: FlashAttentionMetadata = super().build(0, cm, fast_build)
        md.fd_plan = None  # type: ignore[attr-defined]
        md.fd_num_reqs = cm.num_reqs  # type: ignore[attr-defined]
        md.fd_seq_cpu = None  # type: ignore[attr-defined]
        store = state.STORE
        if store is None:
            return md
        if self.kernel_block_size not in (None, store.block_size):
            raise NotImplementedError(
                "FastDecode needs kernel block size == KV block size "
                f"({self.kernel_block_size} vs {store.block_size})"
            )
        bs = store.block_size
        num_reqs = cm.num_reqs
        if num_reqs == 0:
            return md
        qsl = cm.query_start_loc_cpu[: num_reqs + 1]
        if cm.seq_lens_cpu_upper_bound is not None:
            seq = cm.seq_lens_cpu_upper_bound[:num_reqs]
        else:
            seq = cm.seq_lens[:num_reqs].cpu()
        max_blocks = int((int(seq.max()) + bs - 1) // bs)
        width = max(1, min(max_blocks, cm.block_table_tensor.shape[1]))
        bt_dev = cm.block_table_tensor[:num_reqs]
        n_tok = int(qsl[-1])
        # One small D2H sync per step (not per layer).
        bt = bt_dev[:, :width].cpu()
        slots = cm.slot_mapping[:n_tok].cpu()
        md.fd_seq_cpu = seq  # type: ignore[attr-defined]
        hp = build_host_plan(
            qsl, seq, bt, slots, store.num_gpu_blocks, store.num_cpu_blocks, bs
        )
        if hp is not None:
            md.fd_plan = to_device(hp, bt_dev[:, :width], self.device)  # type: ignore[attr-defined]
        if config.stats_enabled():
            self._log_stats(hp, num_reqs)
        return md

    def _log_stats(self, hp, num_reqs) -> None:
        a = self._acc
        a[0] += 1
        if hp is None:
            a[1] += num_reqs
        else:
            a[1] += int(hp.p_req.numel())
            a[2] += hp.n_s
            a[3] += int(hp.f_cu.numel()) - 1
            a[4] += int(hp.s_cpu_lens.sum())
        if a[0] % 50 == 0:
            logger.warning(
                "FastDecode stats (last 50 steps): avg reqs/step all-GPU=%.1f "
                "split-decode=%.1f fresh-prefill-on-CPU=%.1f, avg CPU KV tokens "
                "read per layer per step=%.0f",
                a[1] / 50, a[2] / 50, a[3] / 50, a[4] / 50,
            )
            self._acc = [a[0], 0, 0, 0, 0]


class FastDecodeImpl(FlashAttentionImpl):
    supports_dcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.alibi_slopes is not None:
            raise NotImplementedError("FastDecode: ALiBi is not supported")
        if tuple(self.sliding_window) != (-1, -1):
            raise NotImplementedError("FastDecode: sliding window is not supported")
        if self.sinks is not None:
            raise NotImplementedError("FastDecode: attention sinks are not supported")
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("FastDecode: quantized KV cache is not supported")
        if self.dcp_world_size > 1:
            raise NotImplementedError("FastDecode: DCP is not supported")

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("FastDecode: fused output quantization")
        if attn_metadata is None:  # profiling run
            return output.fill_(0)
        plan = getattr(attn_metadata, "fd_plan", None)
        overlap = sr.active()
        vf = None if overlap else _VERIFIER  # the verifier assumes one batch per step
        if vf is not None and not (
            state.STORE is not None and key is not None and vf.selected(layer.layer_name)
        ):
            vf = None
        if plan is None:
            # Stock path. KV update normally runs as a separate op, which this
            # backend disables (forward_includes_kv_cache_update=True).
            if self.kv_sharing_target_layer_name is None and key is not None:
                FlashAttentionImpl.do_kv_cache_update(
                    self, layer, key, value, kv_cache, attn_metadata.slot_mapping
                )
            out = super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output
            )
            if vf is not None:
                vf.record(layer.layer_name, key, attn_metadata)
            if overlap:
                sr.yield_to_other()  # no CPU work, but the handoff must stay paired
            return out
        out = self._forward_split(layer, query, key, value, kv_cache, plan, output, overlap)
        if vf is not None:
            vf.record(layer.layer_name, key, attn_metadata)
            vf.check(self, layer, query, key, value, kv_cache, attn_metadata, output)
        return out

    def _forward_split(self, layer, query, key, value, kv_cache, dp: DevicePlan, output,
                       overlap: bool = False):
        hp = dp.host
        store = state.STORE
        assert store is not None
        k_cpu = store.k[layer.layer_name]
        v_cpu = store.v[layer.layer_name]
        fa = OPS.fa
        H, D = self.num_heads, self.head_size
        key_cache, value_cache = kv_cache.transpose(1, 2).split(D, dim=-1)
        st = state.staging(sr.ubatch_id() if overlap else 0)

        # 1) KV writes for GPU-resident slots (CPU slots are -1 -> skipped)
        if self.kv_sharing_target_layer_name is None:
            OPS.reshape_and_cache(
                key, value, key_cache, value_cache, dp.gpu_slot_mapping,
                self.kv_cache_dtype, layer._k_scale, layer._v_scale,
            )

        # 2) Stage CPU-bound tensors (async D2H into pinned buffers)
        nw = int(hp.cpu_w_tok.numel())
        ns = hp.n_s
        do_kv = nw > 0 and self.kv_sharing_target_layer_name is None
        if do_kv:
            kw = key.index_select(0, dp.cpu_w_tok)
            vw = value.index_select(0, dp.cpu_w_tok)
            hk = st.get("k", tuple(kw.shape), kw.dtype)
            hv = st.get("v", tuple(vw.shape), vw.dtype)
            hk.copy_(kw, non_blocking=True)
            hv.copy_(vw, non_blocking=True)
        if ns:
            qs = query.index_select(0, dp.s_tok)
            hq = st.get("q", tuple(qs.shape), qs.dtype)
            hq.copy_(qs, non_blocking=True)
        staged = OPS.event()

        # 3) GPU attention that does not need the CPU (overlaps with step 4)
        if ns:
            # Rows without GPU tokens keep (out=0, lse=-inf): the merge then
            # takes the CPU partial alone.
            o_gpu = torch.zeros_like(qs)
            lse_gpu = torch.full((H, ns), float("-inf"), dtype=torch.float32, device=qs.device)
            if hp.s_max_gpu_len > 0:
                # Only rows with GPU tokens go to FlashAttention. For a row with
                # seqused_k == 0, FA2's paged (split-KV) kernel exits early and
                # writes its LSE at a padded-layout offset, i.e. over OTHER rows'
                # LSE; the merge then silently drops their GPU partial.
                # num_splits=1 also keeps FA off the split-KV combine path, so the
                # LSE always comes from the regular [heads, total_q] epilogue.
                rows = dp.s_gpu_rows
                o_g, lse_g = fa(
                    q=qs.index_select(0, rows), k=key_cache, v=value_cache,
                    cu_seqlens_q=dp.s_gpu_cu, max_seqlen_q=1,
                    seqused_k=dp.s_gpu_lens, max_seqlen_k=hp.s_max_gpu_len,
                    softmax_scale=self.scale, causal=False,
                    block_table=dp.s_gpu_bt, softcap=self.logits_soft_cap,
                    return_softmax_lse=True, num_splits=1,
                    fa_version=self.vllm_flash_attn_version,
                )
                o_gpu.index_copy_(0, rows, o_g)
                lse_gpu.index_copy_(1, rows, lse_g.float())
        if hp.n_p_tok:
            q_p = query.index_select(0, dp.p_tok)
            o_p = fa(
                q=q_p, k=key_cache, v=value_cache,
                cu_seqlens_q=dp.p_cu, max_seqlen_q=hp.p_max_q,
                seqused_k=dp.p_seqlens, max_seqlen_k=hp.p_max_k,
                softmax_scale=self.scale, causal=True,
                block_table=dp.p_bt, softcap=self.logits_soft_cap,
                fa_version=self.vllm_flash_attn_version,
            )
            output.index_copy_(0, dp.p_tok, o_p)
        if hp.n_f_tok:
            q_f = query.index_select(0, dp.f_tok)
            k_f = key.index_select(0, dp.f_tok)
            v_f = value.index_select(0, dp.f_tok)
            o_f = fa(
                q=q_f, k=k_f, v=v_f,
                cu_seqlens_q=dp.f_cu, cu_seqlens_k=dp.f_cu,
                max_seqlen_q=hp.f_max, max_seqlen_k=hp.f_max,
                softmax_scale=self.scale, causal=True,
                softcap=self.logits_soft_cap,
                fa_version=self.vllm_flash_attn_version,
            )
            output.index_copy_(0, dp.f_tok, o_f)

        # 4) CPU side: store new K/V, then partial attention over CPU KV
        ext = cpu_ops.load_ext()

        def cpu_part():
            staged.synchronize()
            if do_kv:
                ext.store_kv(hk, hv, hp.cpu_w_slot, k_cpu, v_cpu)
            if not ns:
                return None, None
            ho = st.get("o", (ns, H, D), query.dtype)
            hl = st.get("lse", (H, ns), torch.float32)
            ext.decode_attn(hq, k_cpu, v_cpu, hp.s_cpu_bt, hp.s_cpu_lens,
                            float(self.scale), ho, hl)
            return ho, hl

        if overlap:
            # S/R: the other micro-batch's GPU work (out-proj, MLP, next QKV)
            # runs while this micro-batch's CPU part does. The buffers in `st`
            # belong to this micro-batch, and its thread does not touch them
            # again until the job is done.
            job = sr.submit(cpu_part) if (do_kv or ns) else None
            sr.yield_to_other()
            ho, hl = job.result() if job is not None else (None, None)
        else:
            ho, hl = cpu_part()
        if ns:
            o_cpu = ho.to(query.device, non_blocking=True)
            lse_cpu = hl.to(query.device, non_blocking=True)
            merged = torch.empty_like(o_gpu)
            OPS.merge(merged, o_gpu, lse_gpu, o_cpu, lse_cpu)
            output.index_copy_(0, dp.s_tok, merged)
        return output
