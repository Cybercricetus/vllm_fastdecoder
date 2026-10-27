"""Measure how fast this machine's CPU runs FastDecode's decode attention.

Run this first on the target machine. It reports the KV bandwidth the kernel
achieves and the resulting upper bound on CPU-side decode throughput, which is
the number that decides whether FastDecode can beat the GPU-only baseline:
offloading pays off only when the CPU can add tokens/s faster than the GPU
loses them (see README, "When does it help").

    python scripts/probe_cpu_attn.py --heads 32 --kv-heads 32 --head-dim 128 \
        --layers 32 --ctx 1024 --batch 32 --threads 8
"""

import argparse
import time

import torch

from fastdecode import cpu_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=32, help="for the per-step estimate")
    ap.add_argument("--ctx", type=int, default=1024, help="CPU-resident tokens per request")
    ap.add_argument("--batch", type=int, default=32, help="requests with CPU-resident KV")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--threads", type=int, default=0, help="0 = OpenMP default")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    ext = cpu_ops.load_ext()
    if args.threads:
        ext.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    H, KVH, D, BS = args.heads, args.kv_heads, args.head_dim, args.block_size
    n, L = args.batch, args.ctx
    nb = (L + BS - 1) // BS
    C = n * nb
    kc = torch.randn(C, KVH, BS, D).to(dtype)
    vc = torch.randn(C, KVH, BS, D).to(dtype)
    bt = torch.randperm(C).to(torch.int32).view(n, nb).contiguous()
    lens = torch.full((n,), L, dtype=torch.int32)
    q = torch.randn(n, H, D).to(dtype)
    out = torch.empty_like(q)
    lse = torch.empty(H, n, dtype=torch.float32)

    for _ in range(3):
        ext.decode_attn(q, kc, vc, bt, lens, D ** -0.5, out, lse)
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        ext.decode_attn(q, kc, vc, bt, lens, D ** -0.5, out, lse)
        times.append(time.perf_counter() - t0)
    times.sort()
    t = times[len(times) // 2]

    kv_bytes = 2 * n * L * KVH * D * kc.element_size()
    gbps = kv_bytes / t / 1e9
    per_step = t * args.layers
    print(f"avx2={ext.has_avx2()} threads={ext.get_num_threads()}")
    print(f"one layer: {t * 1e3:.2f} ms for {n} reqs x {L} ctx  ->  {gbps:.1f} GB/s of KV")
    print(f"all {args.layers} layers: {per_step * 1e3:.1f} ms per decode step")
    print(f"=> CPU side can sustain at most ~{n / per_step:.0f} tokens/s at ctx={L} "
          f"(ignores PCIe + Python overhead)")


if __name__ == "__main__":
    main()
