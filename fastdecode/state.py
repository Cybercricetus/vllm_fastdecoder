"""Worker-process state: the CPU-resident KV cache and pinned staging buffers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from fastdecode import config

logger = logging.getLogger("fastdecode")


@dataclass
class CpuKVStore:
    num_gpu_blocks: int  # block ids [0, G) live in the GPU KV cache
    num_cpu_blocks: int  # block ids [G, G + C) live here
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    k: dict[str, torch.Tensor] = field(default_factory=dict)  # [C, KVH, BS, D]
    v: dict[str, torch.Tensor] = field(default_factory=dict)
    steps: int = 0


STORE: CpuKVStore | None = None


def ensure_store(vllm_config, spec, layer_names: list[str]) -> None:
    """Create the CPU KV store in the worker (idempotent).

    Called from the metadata builder's constructor, which runs inside
    GPUModelRunner.initialize_kv_cache, after Worker.initialize_from_config has
    set cache_config.num_gpu_blocks to the real GPU block count G. Builders
    created earlier (memory profiling) see None and do nothing.
    """
    global STORE
    if not config.enabled():
        return
    num_gpu = vllm_config.cache_config.num_gpu_blocks
    if num_gpu is None:
        return
    num_cpu = config.resolve_num_cpu_blocks_from_spec(spec, len(layer_names))
    if num_cpu <= 0:
        STORE = None
        return
    if (
        STORE is not None
        and STORE.num_gpu_blocks == num_gpu
        and STORE.num_cpu_blocks == num_cpu
        and all(n in STORE.k for n in layer_names)
    ):
        return
    config.validate_spec(spec)
    store = CpuKVStore(
        num_gpu_blocks=int(num_gpu),
        num_cpu_blocks=num_cpu,
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=spec.dtype,
    )
    shape = (num_cpu, spec.num_kv_heads, spec.block_size, spec.head_size)
    t0 = time.time()
    for name in layer_names:
        # zeros (not empty) so the pages are really committed now, not mid-run
        store.k[name] = torch.zeros(shape, dtype=spec.dtype)
        store.v[name] = torch.zeros(shape, dtype=spec.dtype)
    elem = torch.empty((), dtype=spec.dtype).element_size()
    gib = 2 * len(layer_names) * torch.Size(shape).numel() * elem / (1 << 30)

    from fastdecode import cpu_ops

    ext = cpu_ops.load_ext()  # compile now so failures surface at startup
    logger.warning(
        "FastDecode: %d GPU blocks + %d CPU blocks (block_size=%d, %.1f GiB host KV "
        "for %d layers, allocated in %.1fs). CPU kernel: avx2=%s threads=%d",
        store.num_gpu_blocks,
        num_cpu,
        spec.block_size,
        gib,
        len(layer_names),
        time.time() - t0,
        ext.has_avx2(),
        ext.get_num_threads(),
    )
    STORE = store


class PinnedStaging:
    """Grow-only host buffers for the per-layer D2H / H2D transfers.

    Reused across layers: every layer finishes its CPU work (and enqueues its
    H2D copies) before the next layer's D2H copies are enqueued on the same
    stream, and the CPU only touches a buffer after waiting on the event that
    follows those copies, so a single set of buffers is race-free. That holds per
    micro-batch only: with S/R micro-batching the two micro-batches of a step
    overlap, so each gets its own set (staging(ubatch)).
    """

    def __init__(self) -> None:
        self._bufs: dict[str, torch.Tensor] = {}
        try:
            from vllm.utils.torch_utils import PIN_MEMORY

            self._pin = bool(PIN_MEMORY)
        except Exception:  # pragma: no cover
            self._pin = True
        if not self._pin:
            logger.warning(
                "FastDecode: pinned memory unavailable (WSL?); transfers will be slower"
            )

    def get(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        numel = 1
        for s in shape:
            numel *= s
        buf = self._bufs.get(name)
        if buf is None or buf.dtype != dtype or buf.numel() < numel:
            cap = max(numel, 1, 2 * (buf.numel() if buf is not None else 0))
            buf = torch.empty(cap, dtype=dtype, pin_memory=self._pin)
            self._bufs[name] = buf
        return buf[:numel].view(shape)


_STAGING: PinnedStaging | None = None
_STAGING_UBATCH: dict[int, PinnedStaging] = {}  # micro-batches 1.. (S/R spike)


def staging(ubatch: int = 0) -> PinnedStaging:
    global _STAGING
    if ubatch:
        st = _STAGING_UBATCH.get(ubatch)
        if st is None:
            st = _STAGING_UBATCH[ubatch] = PinnedStaging()
        return st
    if _STAGING is None:
        _STAGING = PinnedStaging()
    return _STAGING
