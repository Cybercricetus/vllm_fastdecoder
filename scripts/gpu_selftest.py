"""GPU self-test of the split attention path with the REAL CUDA ops (~1 min).

tests/ check the routing logic on the CPU with reference stand-ins for
FlashAttention / reshape_and_cache_flash / merge_attn_states. This script runs
the same FastDecodeImpl._forward_split with vLLM's actual kernels, no model.

It simulates a continuous-batching run: requests arrive with random prompt
lengths, decode for a random number of steps, finish and free their blocks
(GPU-first allocation, LIFO reuse, like the patched vLLM block pool), so block
tables mix GPU and CPU blocks in every pattern, blocks get reused, and KV
persists across steps. Every step runs twice on the same q/k/v:

  fastdecode  G GPU blocks + C CPU blocks, through FastDecode's planner/backend
  stock       one GPU cache of G + C blocks, plain vLLM FlashAttention

and the outputs must agree up to fp16 rounding. The caches start filled with
different random garbage, so reading any slot that was never written shows up
as a mismatch. Every --ref-every steps both are also compared to an fp32
reference.

    python scripts/gpu_selftest.py
    python scripts/gpu_selftest.py --dtype bfloat16 --steps 200
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--max-reqs", type=int, default=64)
    ap.add_argument("--gpu-blocks", type=int, default=40)
    ap.add_argument("--cpu-blocks", type=int, default=700)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=32)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--max-prompt", type=int, default=400)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--ref-every", type=int, default=10)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-mock", action="store_true",
                    help="dev only: run on the CPU with the reference stand-ins from tests/")
    args = ap.parse_args()

    from fastdecode import backend, plan, state

    if args.cpu_mock:  # checks this script's own logic without a GPU
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        import test_plan_and_forward as t

        backend.OPS = t.CpuOps
        stg = state.PinnedStaging.__new__(state.PinnedStaging)
        stg._bufs, stg._pin = {}, False
        state._STAGING = stg
        dev = torch.device("cpu")
    else:
        assert torch.cuda.is_available(), "needs a GPU"
        assert backend.OPS is backend._CudaOps
        dev = torch.device("cuda")
    ops = backend.OPS

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dt = getattr(torch, args.dtype)
    G, C, BS = args.gpu_blocks, args.cpu_blocks, args.block_size
    H, KVH, D = args.heads, args.kv_heads, args.head_size
    scale = 1.0 / math.sqrt(D)
    name = "model.layers.0.self_attn.attn"
    one = torch.ones((), dtype=torch.float32, device=dev)
    layer = SimpleNamespace(layer_name=name, _k_scale=one, _v_scale=one)

    # vLLM's NHD physical layout, exposed as the logical [B, KVH, BS, 2D]
    phys = (torch.randn(G, BS, KVH, 2 * D, device=dev) * 10).to(dt)
    kv_cache = phys.permute(0, 2, 1, 3)
    st = state.CpuKVStore(num_gpu_blocks=G, num_cpu_blocks=C, block_size=BS,
                          num_kv_heads=KVH, head_size=D, dtype=dt)
    st.k[name] = (torch.randn(C, KVH, BS, D) * 10).to(dt)
    st.v[name] = (torch.randn(C, KVH, BS, D) * 10).to(dt)
    state.STORE = st
    uphys = (torch.randn(G + C, BS, KVH, 2 * D, device=dev) * 10).to(dt)
    ukv = uphys.permute(0, 2, 1, 3)

    impl = backend.FastDecodeImpl.__new__(backend.FastDecodeImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = H, KVH, D
    impl.scale, impl.logits_soft_cap = scale, 0.0
    impl.vllm_flash_attn_version = 2
    impl.kv_cache_dtype = "auto"
    impl.kv_sharing_target_layer_name = None

    def stock(q, k, v, kvc, qsl, seq, bt, slots, T, max_q, max_k):
        """What FlashAttentionImpl does: reshape_and_cache_flash + paged FA."""
        kc, vc = kvc.transpose(1, 2).split(D, dim=-1)
        ops.reshape_and_cache(k, v, kc, vc, slots, "auto", one, one)
        return ops.fa(
            q=q[:T], k=kc, v=vc, cu_seqlens_q=qsl, max_seqlen_q=max_q,
            seqused_k=seq, max_seqlen_k=max_k, softmax_scale=scale, causal=True,
            block_table=bt, softcap=0.0, fa_version=2,
        )

    def ref_fp32(q, rows, kvc_u, blocks, L, ql):
        kc, vc = kvc_u.transpose(1, 2).split(D, dim=-1)
        ids = torch.tensor(blocks, device=dev)
        k = kc[ids].reshape(-1, KVH, D)[:L].float().repeat_interleave(H // KVH, 1)
        v = vc[ids].reshape(-1, KVH, D)[:L].float().repeat_interleave(H // KVH, 1)
        s = torch.einsum("qhd,khd->hqk", q[rows].float(), k) * scale
        qi = torch.arange(ql, device=dev)[:, None] + (L - ql)
        s = s.masked_fill((torch.arange(L, device=dev)[None, :] > qi)[None], float("-inf"))
        return torch.einsum("hqk,khd->qhd", s.softmax(-1), v)

    free_gpu = list(range(1, G))  # block 0 is vLLM's null block
    free_cpu = list(range(G, G + C))

    def alloc():
        if free_gpu:
            return free_gpu.pop(0)
        if free_cpu:
            return free_cpu.pop(0)
        return None

    def release(blocks):
        free_gpu[:0] = [b for b in blocks if b < G]
        free_cpu.extend(b for b in blocks if b >= G)

    running = []
    worst = {k: 0.0 for k in ("P", "S_cpu_only", "S_mixed", "F")}
    count = {k: 0 for k in worst}
    worst_ref = {"fastdecode": 0.0, "stock": 0.0}
    n_split = 0
    failures = 0

    for step in range(args.steps):
        # ---- schedule: decodes (need a new block at a block boundary) + new prefills
        rows = []
        for r in list(running):
            if r["L"] % BS == 0:
                b = alloc()
                if b is None:  # out of blocks: drop it (stands in for preemption)
                    running.remove(r)
                    release(r["blocks"])
                    continue
                r["blocks"].append(b)
            rows.append((r, r["L"], 1))
        while len(running) < args.max_reqs:
            P = 1 if rng.random() < 0.05 else rng.randint(2, args.max_prompt)
            need = (P + BS - 1) // BS
            if need > len(free_gpu) + len(free_cpu) - 8:
                break
            r = {"blocks": [alloc() for _ in range(need)], "L": 0,
                 "left": rng.randint(1, args.max_new)}
            running.append(r)
            rows.append((r, 0, P))
        if not rows:
            continue
        rng.shuffle(rows)

        R = len(rows)
        qlens = [ql for _, _, ql in rows]
        seqs = [ctx + ql for _, ctx, ql in rows]
        T = sum(qlens)
        W = max(len(r["blocks"]) for r, _, _ in rows) + 2
        bt = torch.zeros(R, W, dtype=torch.int32)
        slots = []
        for i, (r, ctx, ql) in enumerate(rows):
            bl = r["blocks"]
            bt[i, : len(bl)] = torch.tensor(bl, dtype=torch.int32)
            bt[i, len(bl):] = torch.randint(0, G + C, (W - len(bl),), dtype=torch.int32)  # junk past the end
            for p in range(ctx, ctx + ql):
                slots.append(bl[p // BS] * BS + p % BS)
        qsl = torch.zeros(R + 1, dtype=torch.int32)
        qsl[1:] = torch.cumsum(torch.tensor(qlens), 0)
        seq = torch.tensor(seqs, dtype=torch.int32)
        slots = torch.tensor(slots, dtype=torch.int64)

        # q/k/v as strided views of one qkv tensor, plus cudagraph-style padding rows
        Tp = T + rng.randint(0, 7)
        qkv = torch.randn(Tp, (H + 2 * KVH) * D, device=dev).to(dt)
        q, k, v = qkv.split([H * D, KVH * D, KVH * D], dim=-1)
        q = (q * 2).view(Tp, H, D)
        k = k.view(Tp, KVH, D)
        v = v.view(Tp, KVH, D)

        qsl_d, seq_d, slots_d = qsl.to(dev), seq.to(dev), slots.to(dev)
        bt_d = bt.to(dev)
        max_q, max_k = max(qlens), max(seqs)

        # ---- FastDecode (same steps as FastDecodeMetadataBuilder.build + forward)
        width = max(1, min((max_k + BS - 1) // BS, W))
        hp = plan.build_host_plan(qsl, seq, bt[:, :width], slots, G, C, BS)
        out_fd = torch.full((Tp, H, D), float("nan"), dtype=dt, device=dev)
        if hp is None:
            out_fd[:T] = stock(q, k, v, kv_cache, qsl_d, seq_d, bt_d, slots_d, T, max_q, max_k)
        else:
            n_split += 1
            dp = plan.to_device(hp, bt_d[:, :width], dev)
            impl._forward_split(layer, q, k, v, kv_cache, dp, out_fd)
        # ---- stock vLLM over one big GPU cache
        out_st = stock(q, k, v, ukv, qsl_d, seq_d, bt_d, slots_d, T, max_q, max_k)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        # ---- compare
        if not bool(torch.isfinite(out_fd[:T]).all()):
            bad_rows = (~torch.isfinite(out_fd[:T]).flatten(1).all(1)).nonzero().flatten().tolist()
            print(f"step {step}: NaN/unwritten output rows {bad_rows[:10]}")
            failures += 1
        for i, (r, ctx, ql) in enumerate(rows):
            r0, r1 = int(qsl[i]), int(qsl[i + 1])
            L = ctx + ql
            nb = (L + BS - 1) // BS
            n_cpu = sum(1 for b in r["blocks"][:nb] if b >= G)
            cls = "P" if n_cpu == 0 else ("F" if ql > 1 else ("S_mixed" if n_cpu < nb else "S_cpu_only"))
            a, b = out_fd[r0:r1].float(), out_st[r0:r1].float()
            err = float((a - b).abs().max() / (b.abs().max() + 1e-3))
            count[cls] += 1
            worst[cls] = max(worst[cls], err) if err == err else float("inf")
            if not err <= args.tol:
                failures += 1
                if failures <= 20:
                    print(f"step {step} req {i}: {cls} L={L} q={ql} blocks={r['blocks'][:nb]} "
                          f"(G={G}) rel err vs stock {err:.3g}")
            if args.ref_every and step % args.ref_every == 0:
                ref = ref_fp32(q, slice(r0, r1), ukv, r["blocks"][:nb], L, ql)
                for key, o in (("fastdecode", a), ("stock", b)):
                    e = float((o - ref).abs().max() / (ref.abs().max() + 1e-3))
                    worst_ref[key] = max(worst_ref[key], e)

        # ---- KV written to the right tier and slot (bit-exact)
        ukc, uvc = ukv.transpose(1, 2).split(D, dim=-1)
        kc, vc = kv_cache.transpose(1, 2).split(D, dim=-1)
        sb, so = slots // BS, slots % BS
        g = sb < G
        if g.any():
            gi = g.nonzero().flatten()
            ok = torch.equal(kc[sb[gi].to(dev), so[gi].to(dev)], ukc[sb[gi].to(dev), so[gi].to(dev)])
            ok &= torch.equal(vc[sb[gi].to(dev), so[gi].to(dev)], uvc[sb[gi].to(dev), so[gi].to(dev)])
            if not ok:
                failures += 1
                print(f"step {step}: GPU-tier KV write mismatch")
        if (~g).any():
            ci = (~g).nonzero().flatten()
            cpu_k = st.k[name][sb[ci] - G, :, so[ci]]  # [n, KVH, D]
            cpu_v = st.v[name][sb[ci] - G, :, so[ci]]
            ok = torch.equal(cpu_k.to(dev), ukc[sb[ci].to(dev), so[ci].to(dev)])
            ok &= torch.equal(cpu_v.to(dev), uvc[sb[ci].to(dev), so[ci].to(dev)])
            if not ok:
                failures += 1
                print(f"step {step}: CPU-tier KV write mismatch")

        # ---- advance
        for r, ctx, ql in rows:
            r["L"] = ctx + ql
            r["left"] -= 1
            if r["left"] <= 0:
                running.remove(r)
                release(r["blocks"])

    print(f"\n{args.steps} steps, {n_split} took the split path; dtype {args.dtype}, "
          f"G={G} C={C} block_size={BS} H={H} KVH={KVH} D={D}")
    print("max rel. diff FastDecode vs stock FlashAttention, by request class:")
    for k in worst:
        print(f"  {k:>11}: n={count[k]:6d}  max {worst[k]:.2e}")
    print(f"max rel. err vs fp32 reference: fastdecode {worst_ref['fastdecode']:.2e}, "
          f"stock {worst_ref['stock']:.2e}")
    if failures:
        print(f"\nFAIL: {failures} problems (see above)")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
