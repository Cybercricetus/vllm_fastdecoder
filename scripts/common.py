"""Shared helpers for the GPU scripts: experiment arms and child-process runs.

Every arm runs in its own Python process. That gives each arm a clean CUDA
context and lets the FASTDECODE_* environment differ per arm (vLLM reads plugins
and their env once per process).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Engine settings shared by ALL arms, so the only difference between baseline and
# FastDecode is the attention backend + CPU tier. Chunked prefill and prefix
# caching are off in every arm because the prototype requires it.
COMMON_ENGINE = dict(
    dtype="auto",
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    seed=0,
    disable_log_stats=False,
)

ARMS = {
    # stock vLLM (FlashAttention, default FULL_AND_PIECEWISE cudagraphs)
    "baseline": dict(attention_backend="FLASH_ATTN", env={}),
    # stock vLLM with piecewise cudagraphs only (what FastDecode forces)
    "baseline_pw": dict(attention_backend="FLASH_ATTN", cudagraph_mode="PIECEWISE", env={}),
    # FastDecode code path with an empty CPU tier: isolates our overhead
    "fd_c0": dict(attention_backend="CUSTOM", env={}),
    # FastDecode with the CPU tier (size from --cpu-kv-gib)
    "fastdecode": dict(attention_backend="CUSTOM", env={"FASTDECODE_CPU_KV_GB": None}),
    # --- S/R spike (feat/sr-split). Micro-batched steps cannot use CUDA graphs,
    # so these arms all run eagerly on the V2 model runner; compare
    # fastdecode_sr with fastdecode_eager to see what the overlap buys.
    "baseline_eager": dict(
        attention_backend="FLASH_ATTN", engine=dict(enforce_eager=True),
        env={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    ),
    "fastdecode_eager": dict(
        attention_backend="CUSTOM", engine=dict(enforce_eager=True),
        env={"FASTDECODE_CPU_KV_GB": None, "VLLM_USE_V2_MODEL_RUNNER": "1"},
    ),
    "fastdecode_sr": dict(
        attention_backend="CUSTOM",
        engine=dict(enforce_eager=True, enable_dbo=True, dbo_decode_token_threshold=16,
                    all2all_backend="deepep_low_latency"),  # see fastdecode/sr.py
        env={"FASTDECODE_CPU_KV_GB": None, "FASTDECODE_SR": "1",
             "VLLM_USE_V2_MODEL_RUNNER": "1"},
    ),
}


def add_engine_args(ap) -> None:
    ap.add_argument("--model", default="lmsys/vicuna-7b-v1.5")
    ap.add_argument("--max-model-len", type=int, default=1024,
                    help="stock-vLLM arms need GPU KV >= max_model_len + 16 tokens")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-kv-gib", type=float, default=1.0,
                    help="GPU KV cache size, identical for every arm (kv_cache_memory_bytes)")
    ap.add_argument("--cpu-kv-gib", type=float, default=16.0,
                    help="CPU KV tier size for the 'fastdecode' arm")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="FASTDECODE_CPU_THREADS (0 = OpenMP default)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)


def engine_kwargs(args, arm: str) -> dict:
    spec = ARMS[arm]
    kw = dict(COMMON_ENGINE)
    kw.update(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max(args.max_model_len, 4096),
        kv_cache_memory_bytes=int(args.gpu_kv_gib * (1 << 30)),
        gpu_memory_utilization=args.gpu_mem_util,
        attention_backend=spec["attention_backend"],
    )
    if spec.get("cudagraph_mode"):
        kw["compilation_config"] = {"cudagraph_mode": spec["cudagraph_mode"]}
    kw.update(spec.get("engine", {}))
    return kw


def arm_env(args, arm: str) -> dict:
    env = dict(os.environ)
    for k in list(env):
        if k in ("FASTDECODE_CPU_BLOCKS", "FASTDECODE_CPU_KV_GB", "FASTDECODE_SR"):
            del env[k]
    for k, v in ARMS[arm]["env"].items():
        env[k] = str(args.cpu_kv_gib if v is None else v)
    if args.cpu_threads:
        env["FASTDECODE_CPU_THREADS"] = str(args.cpu_threads)
    return env


def run_child(script: str, arm: str, payload: dict, env: dict, out_path: Path) -> dict:
    """Run `script --_child` in a fresh process; return the JSON it writes."""
    payload = dict(payload, arm=arm, out=str(out_path))
    cmd = [sys.executable, script, "--_child", json.dumps(payload)]
    print(f"\n=== {arm}: {' '.join(cmd[:3])} ...", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"arm {arm} failed with exit code {proc.returncode}")
    return json.loads(out_path.read_text())


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        import vllm

        info.update(torch=torch.__version__, vllm=vllm.__version__)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    info["env"] = {k: v for k, v in os.environ.items() if k.startswith(("FASTDECODE", "VLLM", "OMP"))}
    return info
