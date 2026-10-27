"""Offline decode-throughput benchmark with a counterbalanced (ABBA) design.

Each *session* starts a fresh engine for one arm, runs one warm-up batch, then
--runs-per-session timed batches of the same fixed workload. Sessions are ordered
A B B A A B B A ... so slow drift (thermals, background load) hits both arms
equally. The session mean is the unit of analysis in analyze.py.

Workload: --num-prompts random-token prompts of --input-len tokens, each forced
to generate exactly --output-len tokens (ignore_eos), greedy. Identical across
arms and sessions (fixed seed), so per-run work is constant.

    python scripts/bench.py --arms baseline,fastdecode --sessions 4 \
        --gpu-kv-gib 1.0 --cpu-kv-gib 16 --cpu-threads 6 --out results/bench.jsonl
    python scripts/analyze.py results/bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def child(payload: dict) -> None:
    from vllm import LLM, SamplingParams

    class A:  # rebuild an args-like object for common.engine_kwargs
        pass

    args = A()
    args.__dict__.update(payload["args"])
    arm = payload["arm"]
    llm = LLM(**common.engine_kwargs(args, arm))
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(args.seed)
    prompts = [
        {"prompt_token_ids": [rng.randint(100, vocab - 100) for _ in range(args.input_len)]}
        for _ in range(args.num_prompts)
    ]
    sp = SamplingParams(
        temperature=0.0, max_tokens=args.output_len, min_tokens=args.output_len,
        ignore_eos=True,
    )
    # warm-up (compilation caches, CPU kernel JIT, allocator, pinned buffers)
    warm = SamplingParams(temperature=0.0, max_tokens=16, min_tokens=16, ignore_eos=True)
    llm.generate(prompts, warm, use_tqdm=False)

    runs = []
    for r in range(args.runs_per_session):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        tpots, ttfts = [], []
        for o in outs:
            m = getattr(o, "metrics", None)
            ng = len(o.outputs[0].token_ids)
            if m is not None and getattr(m, "last_token_ts", 0) and ng > 1:
                tpots.append((m.last_token_ts - m.first_token_ts) / (ng - 1))
                if getattr(m, "first_token_latency", 0):
                    ttfts.append(m.first_token_latency)
        tpots.sort()
        runs.append(
            dict(
                run=r, wall_s=dt, output_tokens=n_out, tok_per_s=n_out / dt,
                tpot_p50_ms=1e3 * tpots[len(tpots) // 2] if tpots else None,
                tpot_p99_ms=1e3 * tpots[int(0.99 * (len(tpots) - 1))] if tpots else None,
                ttft_p50_s=sorted(ttfts)[len(ttfts) // 2] if ttfts else None,
            )
        )
        print(f"[{arm}] run {r}: {n_out / dt:.1f} tok/s ({dt:.1f}s)", flush=True)
    Path(payload["out"]).write_text(json.dumps({"runs": runs, "env": common.environment_info()}))


def order(arms: list[str], sessions: int) -> list[str]:
    """Counterbalanced order: each round is the arm list rotated by one more
    position (a Latin square), which for two arms is A B | B A | A B | B A."""
    seq = []
    for i in range(sessions):
        k = i % len(arms)
        seq += arms[k:] + arms[:k]
    return seq


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--_child":
        child(json.loads(sys.argv[2]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_engine_args(ap)
    ap.add_argument("--arms", default="baseline,fastdecode")
    ap.add_argument("--sessions", type=int, default=4, help="sessions per arm")
    ap.add_argument("--runs-per-session", type=int, default=3)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bench.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in common.ARMS:
            raise SystemExit(f"unknown arm {a}; choose from {list(common.ARMS)}")
    seq = order(arms, args.sessions)
    print("session order:", " ".join(seq))
    if args.dry_run:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".child.json")
    for s, arm in enumerate(seq):
        res = common.run_child(__file__, arm, {"args": vars(args)}, common.arm_env(args, arm), tmp)
        with out.open("a") as f:
            for r in res["runs"]:
                rec = dict(arm=arm, session=s, **r, config={k: v for k, v in vars(args).items() if k != "out"})
                if s == 0 or r["run"] == 0:
                    rec["env"] = res["env"]
                f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {out}; now run: python scripts/analyze.py {out}")


if __name__ == "__main__":
    main()
