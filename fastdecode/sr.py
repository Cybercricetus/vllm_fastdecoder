"""SPIKE (throwaway, feat/sr-split): FastDecode's S/R pipeline on one GPU.

The synchronous prototype makes the GPU wait for the CPU in every layer. The
paper instead runs two mini-batches out of phase, so the GPU works on one
mini-batch's shared-weight layers (S part) while the CPU computes the other
one's attention over CPU-resident KV (R part):

    GPU: | A: out-proj, MLP, next QKV | B: out-proj, MLP, next QKV | ...
    CPU: |       B: attention         |       A: attention         | ...

vLLM's dual batch overlap (DBO) already runs a step as two micro-batch threads
that hand control to each other at `dbo_yield()`. FastDecodeImpl starts its CPU
part on a worker thread, yields, and collects the result when control returns.

vLLM 0.31.0 only micro-batches MoE models under data parallelism, so three
pieces of the V2 model runner are patched to allow it on one GPU:

  1. maybe_build_ubatch_runner: drop the data_parallel_size > 1 condition.
  2. dispatch_cg_and_sync_dp: with one DP rank, split decode-only batches of at
     least --dbo-decode-token-threshold tokens into two eager micro-batches.
     Steps with a prefill are never split: a prompt cut across micro-batches
     is the chunked-prefill case that plan.py rejects.
  3. UBatchRunner._make_forward_contexts: build the per-micro-batch forward
     contexts without DP metadata (DPMetadata.make asserts DP > 1 or MoE).

Engine settings it needs (scripts/common.py arm "fastdecode_sr"):
    env  VLLM_USE_V2_MODEL_RUNNER=1 FASTDECODE_SR=1 FASTDECODE_CPU_KV_GB=...
    enable_dbo=True, enforce_eager=True (micro-batched steps cannot use graphs),
    all2all_backend="deepep_low_latency" (only to pass VllmConfig's DBO assert;
    no all-to-all is created without expert parallelism)
"""

from __future__ import annotations

import functools
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_enabled, dbo_yield

logger = logging.getLogger("fastdecode")

_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_STEPS = [0, 0]  # dispatched steps, micro-batched steps


def active() -> bool:
    """True inside a micro-batch thread, i.e. this step is being split."""
    return dbo_enabled()


def ubatch_id() -> int:
    return dbo_current_ubatch_id()


def yield_to_other() -> None:
    """Hand control to the other micro-batch. Every micro-batch must call this
    exactly once per attention layer, or the two threads deadlock."""
    dbo_yield()


def submit(fn) -> Future:
    """Run fn on the CPU-attention worker thread. One worker, FIFO: the CPU parts
    of the two micro-batches never compete for cores."""
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:
                _EXECUTOR = ThreadPoolExecutor(1, thread_name_prefix="fastdecode-cpu")
    return _EXECUTOR.submit(fn)


def apply_worker_patches() -> None:
    import vllm.v1.worker.gpu.model_runner as mr
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import create_forward_context
    from vllm.v1.worker.gpu import ubatch_utils as uu
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds, get_num_ubatches

    orig_dispatch = mr.dispatch_cg_and_sync_dp
    if getattr(orig_dispatch, "_fastdecode_patched", False):
        return

    # 1) build the micro-batch runner even with a single DP rank
    def maybe_build_ubatch_runner(vllm_config, device, model_state, attn_groups,
                                  kv_cache_config, max_num_reqs):
        if not vllm_config.parallel_config.use_ubatching:
            return None
        logger.warning("FastDecode S/R: micro-batch runner enabled on one GPU")
        return uu.UBatchRunner(vllm_config, device, model_state, attn_groups,
                               kv_cache_config, max_num_reqs)

    # 2) decide per step; the stock version returns early when dp_size == 1
    @functools.wraps(orig_dispatch)
    def dispatch_cg_and_sync_dp(cudagraph_manager, num_reqs, num_tokens,
                                uniform_token_count, dp_size, dp_rank, *args, **kwargs):
        desc, sync = orig_dispatch(cudagraph_manager, num_reqs, num_tokens,
                                   uniform_token_count, dp_size, dp_rank, *args, **kwargs)
        pc = kwargs.get("parallel_config")
        if dp_size != 1 or pc is None:
            return desc, sync
        _STEPS[0] += 1
        if (
            kwargs.get("allow_ubatching")
            and kwargs.get("uniform_decode")
            and not kwargs.get("need_eager")
            and num_reqs >= 2
            and check_ubatch_thresholds(pc, num_tokens, uniform_decode=True)
        ):
            _STEPS[1] += 1
            desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=kwargs.get("num_active_loras", 0),
                num_ubatches=get_num_ubatches(pc),
            )
        if _STEPS[0] % 50 == 0:
            logger.warning("FastDecode S/R: %d of the last 50 steps micro-batched",
                           _STEPS[1])
            _STEPS[1] = 0
        return desc, sync

    dispatch_cg_and_sync_dp._fastdecode_patched = True  # type: ignore[attr-defined]

    # 3) per-micro-batch forward contexts without DP metadata
    orig_contexts = uu.UBatchRunner._make_forward_contexts

    @functools.wraps(orig_contexts)
    def _make_forward_contexts(self, ubatch_slices, attn_metadata,
                               slot_mappings_by_layer, is_padding):
        if self.parallel_config.data_parallel_size > 1:
            return orig_contexts(self, ubatch_slices, attn_metadata,
                                 slot_mappings_by_layer, is_padding)
        return [
            create_forward_context(
                attn_metadata[i],
                self.vllm_config,
                dp_metadata=None,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=slot_mappings_by_layer[i],
                is_padding=is_padding[i],
            )
            for i in range(len(ubatch_slices))
        ]

    mr.maybe_build_ubatch_runner = maybe_build_ubatch_runner
    mr.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    uu.UBatchRunner._make_forward_contexts = _make_forward_contexts
    logger.info("FastDecode S/R: V2 model runner patched for single-GPU micro-batching")
