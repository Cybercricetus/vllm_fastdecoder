"""Statistics for bench.py output.

Unit of analysis = one engine session (mean of its timed runs); runs inside a
session share an engine and are not independent. For each arm vs the reference
arm it reports:

  * throughput ratio (mean of session means) with a 95% bootstrap CI
    (10k resamples of sessions within each arm);
  * Welch's t-test and the Mann-Whitney U test on session means (two-sided);
  * TPOT medians, since FastDecode trades latency for throughput.

A speed-up is called significant when the bootstrap CI excludes 1 AND both
tests give p < alpha. Decide --alpha and the workload *before* looking.

    python scripts/analyze.py results/bench.jsonl --ref baseline
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as stats
from collections import defaultdict


def load(path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    by_arm = defaultdict(lambda: defaultdict(list))
    tpot = defaultdict(list)
    for r in recs:
        by_arm[r["arm"]][r["session"]].append(r["tok_per_s"])
        if r.get("tpot_p50_ms") is not None:
            tpot[r["arm"]].append(r["tpot_p50_ms"])
    sessions = {a: [stats.fmean(v) for _, v in sorted(s.items())] for a, s in by_arm.items()}
    return sessions, tpot, recs


def welch(a, b):
    ma, mb = stats.fmean(a), stats.fmean(b)
    va, vb = stats.variance(a), stats.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    t = (ma - mb) / se if se > 0 else float("inf")
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    ) if se > 0 else float("inf")
    try:
        from scipy import stats as ss

        p = 2 * ss.t.sf(abs(t), df)
    except Exception:  # normal approximation
        p = math.erfc(abs(t) / math.sqrt(2))
    return t, df, p


def mann_whitney(a, b):
    try:
        from scipy import stats as ss

        return ss.mannwhitneyu(a, b, alternative="two-sided").pvalue
    except Exception:
        return float("nan")


def bootstrap_ratio(a, b, n=10_000, seed=0):
    rng = random.Random(seed)
    rs = []
    for _ in range(n):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        rs.append(stats.fmean(ra) / stats.fmean(rb))
    rs.sort()
    return rs[int(0.025 * n)], rs[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    sessions, tpot, recs = load(args.path)
    if recs and recs[0].get("env"):
        e = recs[0]["env"]
        print(f"GPU: {e.get('gpu')} | CPU: {e.get('cpu')} x{e.get('cpu_count')} | "
              f"vllm {e.get('vllm')} torch {e.get('torch')}")
    print(f"\n{'arm':<12} {'sessions':>8} {'mean tok/s':>11} {'sd':>8} {'CV%':>6} {'TPOT p50 ms':>12}")
    for arm, s in sessions.items():
        sd = stats.stdev(s) if len(s) > 1 else float("nan")
        tp = stats.median(tpot[arm]) if tpot[arm] else float("nan")
        print(f"{arm:<12} {len(s):>8} {stats.fmean(s):>11.1f} {sd:>8.1f} "
              f"{100 * sd / stats.fmean(s):>6.2f} {tp:>12.1f}")

    if args.ref not in sessions:
        raise SystemExit(f"reference arm {args.ref!r} not in results")
    ref = sessions[args.ref]
    for arm, s in sessions.items():
        if arm == args.ref:
            continue
        if len(s) < 2 or len(ref) < 2:
            print(f"\n{arm}: need >=2 sessions per arm for statistics")
            continue
        ratio = stats.fmean(s) / stats.fmean(ref)
        lo, hi = bootstrap_ratio(s, ref)
        t, df, p = welch(s, ref)
        pu = mann_whitney(s, ref)
        sig = (lo > 1 or hi < 1) and p < args.alpha and (math.isnan(pu) or pu < args.alpha)
        print(f"\n{arm} vs {args.ref}: ratio {ratio:.3f}x  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  Welch t={t:.2f} df={df:.1f} p={p:.2g} | Mann-Whitney p={pu:.2g}")
        if len(s) < 4 or len(ref) < 4:
            print("  note: with <4 sessions per arm Mann-Whitney cannot reach p<0.05")
        print(f"  => {'SIGNIFICANT' if sig else 'not significant'} at alpha={args.alpha}"
              + (f" ({'faster' if ratio > 1 else 'slower'})" if sig else ""))


if __name__ == "__main__":
    main()
