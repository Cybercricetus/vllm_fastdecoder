#!/usr/bin/env bash
# SPIKE (feat/sr-split): does the S/R pipeline (fastdecode/sr.py) run, stay
# correct, and beat the synchronous version? Needs a CUDA GPU and the vllm==0.31.0
# environment from the README. Roughly 1.5-2 h on one GPU.
#   THREADS=6 bash scripts/sr_spike.sh
#
# THREADS is the CPU attention thread count. Leave a core or two free for the two
# micro-batch threads, which launch GPU work while the CPU kernel runs.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-lmsys/vicuna-7b-v1.5}
NPROC=$(nproc)
THREADS=${THREADS:-$(( NPROC / 2 > 2 ? NPROC / 2 - 1 : 1 ))}
GPU_KV=${GPU_KV:-1.0}      # GiB of GPU KV for the tiered comparison
CPU_KV=${CPU_KV:-32}       # GiB of CPU KV for the fastdecode arms
SESSIONS=${SESSIONS:-4}
OUT=${OUT:-results/sr-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
BENCH=(--model "$MODEL" --sessions "$SESSIONS" --cpu-kv-gib "$CPU_KV"
       --cpu-threads "$THREADS" --max-num-seqs 128 --num-prompts 128
       --input-len 256 --output-len 512)

echo "### 0. unit tests (CPU only; rebuilds the kernel, which now drops the GIL)"
python -m pytest tests -q

echo "### 1. smoke: does a micro-batched run start and finish? A hang here is a deadlock."
timeout 1800 python scripts/check_correctness.py --task tokens --model "$MODEL" \
  --arms fastdecode_sr --limit 64 --max-tokens 64 \
  --gpu-kv-gib 0.25 --cpu-kv-gib 8 --cpu-threads "$THREADS" \
  --outdir "$OUT/smoke" 2>&1 | tee "$OUT/smoke.log"
grep -q "micro-batch runner enabled" "$OUT/smoke.log" \
  || { echo "FAIL: the micro-batch runner was never built"; exit 1; }
grep "steps micro-batched" "$OUT/smoke.log" | tail -3 \
  || echo "WARN: no micro-batching report yet (fewer than 50 steps?)"

echo "### 2. token-level correctness under heavy offload"
timeout 5400 python scripts/check_correctness.py --task tokens --model "$MODEL" \
  --arms baseline,baseline_alt,fastdecode_eager,fastdecode_sr \
  --gpu-kv-gib 0.25 --cpu-kv-gib 8 --cpu-threads "$THREADS" --limit 128 \
  --outdir "$OUT/correctness" 2>&1 | tee "$OUT/correctness.txt"

echo "### 3. throughput, tiered placement (GPU KV ${GPU_KV} GiB, rest on CPU)"
python scripts/bench.py "${BENCH[@]}" --gpu-kv-gib "$GPU_KV" \
  --arms baseline,baseline_eager,fastdecode_eager,fastdecode_sr \
  --out "$OUT/bench_tiered.jsonl"
python scripts/analyze.py "$OUT/bench_tiered.jsonl" --ref baseline | tee "$OUT/analysis_tiered.txt"
python scripts/analyze.py "$OUT/bench_tiered.jsonl" --ref fastdecode_eager | tee -a "$OUT/analysis_tiered.txt"

echo "### 4. throughput, paper-like placement (almost all KV on the CPU)"
python scripts/bench.py "${BENCH[@]}" --gpu-kv-gib 0.25 \
  --arms fastdecode_eager,fastdecode_sr \
  --out "$OUT/bench_cpu_heavy.jsonl"
python scripts/analyze.py "$OUT/bench_cpu_heavy.jsonl" --ref fastdecode_eager | tee "$OUT/analysis_cpu_heavy.txt"

echo "results in $OUT"
