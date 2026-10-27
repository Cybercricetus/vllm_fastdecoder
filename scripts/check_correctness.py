"""Does FastDecode change model outputs? Two checks, both vs. stock vLLM.

1) tokens (default, ~10 min): greedy generation with top-k logprobs under heavy
   CPU offload (small GPU tier, many concurrent requests). Uses the criterion of
   vLLM's own test suite (tests/models/utils.py::check_logprobs_close): two
   outputs agree if identical, or if at the first differing position each
   sampled token is in the other run's top-k. Also measures the *noise floor*:
   stock vLLM vs stock vLLM with a different max_num_seqs, since vLLM is not
   batch-invariant and FastDecode necessarily changes batch composition.

2) gsm8k (~1 h on a laptop): task accuracy, paired per question.
   - McNemar exact test for a difference;
   - TOST equivalence test: "no meaningful loss" is claimed only if the 90% CI
     of the paired accuracy difference lies inside +/- --margin (pre-register!).
   Failing to find a significant difference is NOT evidence of equivalence; the
   TOST is what supports the claim.

    python scripts/check_correctness.py --task tokens --gpu-kv-gib 0.25 --cpu-kv-gib 8
    python scripts/check_correctness.py --task gsm8k --limit 1319 --margin 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FALLBACK_PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main causes of the French Revolution?",
    "Describe the process of photosynthesis step by step.",
    "Give me a recipe for a vegetarian lasagna.",
    "Compare TCP and UDP and say when each is appropriate.",
    "Summarize the plot of Romeo and Juliet in a paragraph.",
    "What is the difference between a virus and a bacterium?",
]


def load_prompts(task: str, limit: int):
    if task == "gsm8k":
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [ds[i] for i in range(min(limit, len(ds)))]
        prompts = [f"Question: {r['question']}\nAnswer: Let's think step by step." for r in rows]
        answers = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]
        return prompts, answers
    try:  # diverse real prompts if `datasets` is available
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompts = [f"Question: {ds[i]['question']}\nAnswer:" for i in range(limit)]
    except Exception:
        prompts = [FALLBACK_PROMPTS[i % len(FALLBACK_PROMPTS)] + f" (variant {i})" for i in range(limit)]
    return prompts, None


def child(payload: dict) -> None:
    from vllm import LLM, SamplingParams

    class A:
        pass

    args = A()
    args.__dict__.update(payload["args"])
    if payload.get("max_num_seqs"):
        args.max_num_seqs = payload["max_num_seqs"]
    llm = LLM(**common.engine_kwargs(args, payload["arm"]))
    prompts, _ = load_prompts(args.task, args.limit)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        logprobs=args.topk if args.task == "tokens" else None)
    outs = llm.generate(prompts, sp)
    res = []
    for o in outs:
        c = o.outputs[0]
        item = {"ids": list(c.token_ids), "text": c.text}
        if c.logprobs is not None:
            item["top"] = [[int(t) for t in pos.keys()] if pos else [] for pos in c.logprobs]
        res.append(item)
    Path(payload["out"]).write_text(json.dumps({"outputs": res, "env": common.environment_info()}))


def compare_tokens(a, b):
    """vLLM check_logprobs_close semantics. Returns (exact, close, fail, first_div)."""
    exact = close = fail = 0
    divs = []
    for x, y in zip(a, b):
        ids_x, ids_y = x["ids"], y["ids"]
        for pos, (tx, ty) in enumerate(zip(ids_x, ids_y)):
            if tx != ty:
                if tx in y["top"][pos] and ty in x["top"][pos]:
                    close += 1
                else:
                    fail += 1
                divs.append(pos)
                break
        else:
            if len(ids_x) == len(ids_y):
                exact += 1
            else:
                close += 1
                divs.append(min(len(ids_x), len(ids_y)))
    return exact, close, fail, divs


def fisher_two_sided(a, b, c, d):
    """Fisher exact test on [[a, b], [c, d]]."""
    from math import comb

    n1, n2, k = a + b, c + d, a + c
    def p(x):
        return comb(n1, x) * comb(n2, k - x) / comb(n1 + n2, k)
    p0 = p(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def extract_number(text: str):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm8k_stats(base, fd, answers, margin):
    cb = [extract_number(o["text"]) == ans for o, ans in zip(base, answers)]
    cf = [extract_number(o["text"]) == ans for o, ans in zip(fd, answers)]
    n = len(cb)
    b01 = sum(1 for x, y in zip(cb, cf) if x and not y)  # base right, fd wrong
    b10 = sum(1 for x, y in zip(cb, cf) if y and not x)
    # McNemar exact (binomial on discordant pairs)
    m = b01 + b10
    k = min(b01, b10)
    p_mc = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    # TOST on paired differences d_i in {-1, 0, 1}
    d = [int(y) - int(x) for x, y in zip(cb, cf)]
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if var > 0 else 1e-12
    z90 = 1.6448536269514722
    lo, hi = mean - z90 * se, mean + z90 * se
    p_lo = 0.5 * math.erfc(((mean + margin) / se) / math.sqrt(2))
    p_hi = 0.5 * math.erfc(((margin - mean) / se) / math.sqrt(2))
    return dict(n=n, acc_base=sum(cb) / n, acc_fd=sum(cf) / n, discordant=(b01, b10),
                p_mcnemar=p_mc, diff=mean, ci90=(lo, hi), p_tost=max(p_lo, p_hi))


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--_child":
        child(json.loads(sys.argv[2]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_engine_args(ap)
    ap.add_argument("--task", choices=["tokens", "gsm8k"], default="tokens")
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.02, help="TOST equivalence margin")
    ap.add_argument("--baseline-gpu-kv-gib", type=float, default=1.0,
                    help="GPU KV for the stock-vLLM arms (the fastdecode arm uses "
                         "--gpu-kv-gib, typically much smaller to force offloading)")
    ap.add_argument("--alt-gpu-kv-gib", type=float, default=0.6,
                    help="GPU KV for the noise-floor arm: a different size changes how "
                         "requests get batched, which is what FastDecode also does")
    ap.add_argument("--outdir", default="results/correctness")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    plan = [("baseline", None), ("fastdecode", None)]
    if args.task == "tokens":
        plan.insert(1, ("baseline_alt", max(1, args.max_num_seqs // 4)))
    for name, mns in plan:
        arm = "baseline" if name == "baseline_alt" else name
        a = dict(vars(args))
        if name == "baseline_alt":
            a["gpu_kv_gib"] = args.alt_gpu_kv_gib
        elif arm != "fastdecode":
            a["gpu_kv_gib"] = args.baseline_gpu_kv_gib
        payload = {"args": a, "max_num_seqs": mns}
        runs[name] = common.run_child(__file__, arm, payload, common.arm_env(args, arm),
                                      out / f"{args.task}_{name}.json")["outputs"]

    if args.task == "tokens":
        n = len(runs["baseline"])
        rows = {}
        for name in ("baseline_alt", "fastdecode"):
            e, c, f, divs = compare_tokens(runs["baseline"], runs[name])
            rows[name] = (e, c, f)
            md = sorted(divs)[len(divs) // 2] if divs else None
            print(f"{name:>13} vs baseline: exact {e}/{n}, close {c}, FAIL {f}, "
                  f"median first divergence at token {md}")
        (e0, _, f0), (e1, _, f1) = rows["baseline_alt"], rows["fastdecode"]
        p = fisher_two_sided(e1, n - e1, e0, n - e0)
        print(f"\nexact-match rate fastdecode {e1 / n:.3f} vs noise floor {e0 / n:.3f} "
              f"(Fisher p={p:.3g})")
        ok = f1 == 0 and (e1 >= e0 or p > 0.05)
        print("PASS: FastDecode is within vLLM's own batch-variance noise." if ok else
              "CHECK: FastDecode deviates more than the noise floor -- investigate.")
    else:
        _, answers = load_prompts("gsm8k", args.limit)
        s = gsm8k_stats(runs["baseline"], runs["fastdecode"], answers, args.margin)
        print(json.dumps(s, indent=2))
        eq = s["ci90"][0] > -args.margin and s["ci90"][1] < args.margin
        print(f"accuracy baseline {s['acc_base']:.4f} vs fastdecode {s['acc_fd']:.4f}; "
              f"McNemar p={s['p_mcnemar']:.3g}; 90% CI of diff [{s['ci90'][0]:+.4f}, "
              f"{s['ci90'][1]:+.4f}] -> {'EQUIVALENT' if eq else 'not shown equivalent'} "
              f"within +/-{args.margin} (TOST p={s['p_tost']:.3g})")


if __name__ == "__main__":
    main()
