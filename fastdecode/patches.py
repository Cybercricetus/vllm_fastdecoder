"""The only change to vLLM itself: the scheduler sees G + C KV blocks.

EngineCore._initialize_kv_caches profiles GPU memory, initializes the workers'
GPU KV cache with G blocks, and returns the KVCacheConfig the scheduler is
built from. We return a copy with num_blocks = G + C, so the unmodified
scheduler / KV cache manager hand out block ids in [0, G + C); ids >= G are
served from the CPU store by the attention backend.
"""

from __future__ import annotations

import copy
import functools
import logging

from fastdecode import config, verify

logger = logging.getLogger("fastdecode")

# Number of GPU blocks, set in the engine-core process once known.
_GPU_BLOCKS: int | None = None


def _apply_gpu_first_free_queue() -> None:
    """Keep every free GPU block ahead of every free CPU block.

    With prefix caching off, BlockPool.free_blocks() prepends freed blocks
    (LIFO) and get_new_blocks() pops from the front, so a just-freed CPU block
    would be handed out before an idle GPU block. Prepending only GPU blocks and
    appending CPU blocks preserves the "GPU ids first" invariant the free queue
    starts with (it is built in id order), so CPU blocks are used only when the
    GPU tier is full.
    """
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue

    orig = FreeKVCacheBlockQueue.prepend_n
    if getattr(orig, "_fastdecode_patched", False):
        return

    @functools.wraps(orig)
    def prepend_n(self, blocks):
        g = _GPU_BLOCKS
        if g is None or not blocks:
            return orig(self, blocks)
        gpu = [b for b in blocks if b.block_id < g]
        cpu = [b for b in blocks if b.block_id >= g]
        orig(self, gpu)
        self.append_n(cpu)

    prepend_n._fastdecode_patched = True  # type: ignore[attr-defined]
    FreeKVCacheBlockQueue.prepend_n = prepend_n


def _apply_capacity_check_patch() -> None:
    """vLLM refuses to start unless the *GPU* KV cache can hold one
    max_model_len request. With a CPU tier a request may span both tiers, so
    count the CPU blocks in that admission check too."""
    import vllm.v1.core.kv_cache_utils as kvu

    orig = kvu._check_enough_kv_cache_memory
    if getattr(orig, "_fastdecode_patched", False):
        return

    @functools.wraps(orig)
    def check(available_memory, get_needed_memory, max_model_len, estimate_max_model_len):
        extra = 0
        try:
            groups = get_needed_memory.args[1]  # partial(_max_memory_usage..., cfg, groups)
            per_block = kvu._pool_bytes_per_block(groups)
            extra = config._resolve(per_block) * per_block
        except Exception:
            logger.warning("FastDecode: could not size the CPU tier for the capacity check")
        return orig(available_memory + extra, get_needed_memory, max_model_len,
                    estimate_max_model_len)

    check._fastdecode_patched = True  # type: ignore[attr-defined]
    kvu._check_enough_kv_cache_memory = check


def apply_engine_core_patch() -> None:
    from vllm.v1.engine.core import EngineCore

    _apply_capacity_check_patch()
    orig = EngineCore._initialize_kv_caches
    if getattr(orig, "_fastdecode_patched", False):
        return

    @functools.wraps(orig)
    def _initialize_kv_caches(self, vllm_config):
        kv_cfg = orig(self, vllm_config)
        num_cpu = config.resolve_num_cpu_blocks(kv_cfg)
        if num_cpu <= 0:
            return kv_cfg
        config.validate_kv_cache_config(kv_cfg)
        sched = vllm_config.scheduler_config
        cache = vllm_config.cache_config
        problems = []
        if cache.enable_prefix_caching:
            problems.append("enable_prefix_caching=False (--no-enable-prefix-caching)")
        if sched.enable_chunked_prefill:
            problems.append("enable_chunked_prefill=False (--no-enable-chunked-prefill)")
        if vllm_config.speculative_config is not None:
            problems.append("no speculative decoding")
        if vllm_config.parallel_config.world_size > 1:
            problems.append("tensor/pipeline parallel size 1")
        if config.sr_enabled():
            if not vllm_config.parallel_config.enable_dbo:
                problems.append("enable_dbo=True (FASTDECODE_SR=1)")
            if not vllm_config.use_v2_model_runner:
                problems.append("VLLM_USE_V2_MODEL_RUNNER=1 (FASTDECODE_SR=1)")
            if not vllm_config.model_config.enforce_eager:
                # micro-batched steps hand off mid-layer; never capture them
                problems.append("enforce_eager=True (FASTDECODE_SR=1)")
            if verify.enabled():
                # the verifier assumes one batch per step and is skipped in
                # split steps, so its write table would go stale
                problems.append("FASTDECODE_VERIFY unset (FASTDECODE_SR=1)")
        if problems:
            raise ValueError(
                "FastDecode prototype requires: " + "; ".join(problems)
                + ". (A request extended by >1 token while part of its KV is on "
                "the CPU is not implemented.)"
            )
        global _GPU_BLOCKS
        _GPU_BLOCKS = kv_cfg.num_blocks
        _apply_gpu_first_free_queue()
        new_cfg = copy.copy(kv_cfg)
        new_cfg.num_blocks = kv_cfg.num_blocks + num_cpu
        bs = kv_cfg.kv_cache_groups[0].kv_cache_spec.block_size
        logger.warning(
            "FastDecode: scheduler KV capacity %d GPU + %d CPU blocks = %d tokens "
            "(GPU-only would be %d tokens)",
            kv_cfg.num_blocks, num_cpu, new_cfg.num_blocks * bs, kv_cfg.num_blocks * bs,
        )
        return new_cfg

    _initialize_kv_caches._fastdecode_patched = True  # type: ignore[attr-defined]
    EngineCore._initialize_kv_caches = _initialize_kv_caches
