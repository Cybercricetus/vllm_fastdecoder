"""Environment-driven configuration.

Both the engine-core process (scheduler) and the worker process read these, so
they must be set in the environment that launches vLLM (they are inherited by
the spawned processes).

FASTDECODE_CPU_BLOCKS   number of extra KV blocks held in CPU memory, or
FASTDECODE_CPU_KV_GB    the same expressed as GiB of CPU memory (ignored if
                        FASTDECODE_CPU_BLOCKS is set)
FASTDECODE_CPU_THREADS  OpenMP threads for the CPU attention kernel
                        (default: OpenMP default; set it to your P-core count)
FASTDECODE_MARCH        -march for the JIT build (default "native")
FASTDECODE_STATS        "1" to log per-step GPU/CPU token split every 50 steps
FASTDECODE_SR           "1" for the S/R pipeline spike (see fastdecode/sr.py)
"""

from __future__ import annotations

import os

ENV_CPU_BLOCKS = "FASTDECODE_CPU_BLOCKS"
ENV_CPU_KV_GB = "FASTDECODE_CPU_KV_GB"
ENV_STATS = "FASTDECODE_STATS"
ENV_SR = "FASTDECODE_SR"


def enabled() -> bool:
    return bool(os.environ.get(ENV_CPU_BLOCKS) or os.environ.get(ENV_CPU_KV_GB))


def stats_enabled() -> bool:
    return os.environ.get(ENV_STATS, "0") not in ("", "0", "false", "False")


def sr_enabled() -> bool:
    return enabled() and os.environ.get(ENV_SR, "0") not in ("", "0", "false", "False")


def kv_bytes_per_block(kv_cache_config) -> int:
    """Bytes one KV block occupies across all layers (K and V)."""
    total = 0
    for group in kv_cache_config.kv_cache_groups:
        total += group.kv_cache_spec.page_size_bytes * len(group.layer_names)
    return total


def _resolve(per_block_bytes: int) -> int:
    blocks = os.environ.get(ENV_CPU_BLOCKS)
    if blocks:
        return max(0, int(blocks))
    gb = os.environ.get(ENV_CPU_KV_GB)
    if gb:
        return max(0, int(float(gb) * (1 << 30)) // per_block_bytes)
    return 0


def resolve_num_cpu_blocks(kv_cache_config) -> int:
    """Engine-core side (has the whole KVCacheConfig)."""
    return _resolve(kv_bytes_per_block(kv_cache_config))


def resolve_num_cpu_blocks_from_spec(spec, num_layers: int) -> int:
    """Worker side (metadata builder sees one group's spec and layer names).
    Must agree with resolve_num_cpu_blocks for single-group models."""
    return _resolve(spec.page_size_bytes * num_layers)


def validate_kv_cache_config(kv_cache_config) -> None:
    """The prototype supports exactly one full-attention KV cache group."""
    groups = kv_cache_config.kv_cache_groups
    if len(groups) != 1:
        raise NotImplementedError(
            f"FastDecode supports models with a single KV cache group, got {len(groups)} "
            "(hybrid / sliding-window / MLA models are out of scope)."
        )
    validate_spec(groups[0].kv_cache_spec)


def validate_spec(spec) -> None:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

    if not isinstance(spec, FullAttentionSpec) or spec.sliding_window is not None:
        raise NotImplementedError(f"FastDecode needs full attention, got {type(spec).__name__}")
    if spec.kv_quant_mode != KVQuantMode.NONE:
        raise NotImplementedError("FastDecode does not support quantized KV cache")
    if spec.head_size_v != spec.head_size:
        raise NotImplementedError("FastDecode needs head_size_v == head_size")
