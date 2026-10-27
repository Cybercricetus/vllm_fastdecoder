"""FastDecode for vLLM: a CPU tier for the KV cache with CPU-side attention.

Loaded automatically by vLLM through the `vllm.general_plugins` entry point,
in every vLLM process (front end, engine core, workers).
"""

__version__ = "0.1.0"

BACKEND_CLASS = "fastdecode.backend.FastDecodeBackend"


def register() -> None:
    import logging

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    from fastdecode import config

    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_CLASS)
    if config.enabled():
        from fastdecode.patches import apply_engine_core_patch

        apply_engine_core_patch()
        logging.getLogger("fastdecode").info("FastDecode plugin active")
