#!/usr/bin/env bash
# Full pipeline with defaults for a 16 GB GPU + 64 GB RAM machine
# (e.g. RTX 3080 Ti laptop) running vicuna-7b-v1.5 in fp16.
#   THREADS=6 CPU_KV=32 GPU_KV=1.0 bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-lmsys/vicuna-7b-v1.5}
THREADS=${THREADS:-$(( $(nproc) / 2 ))}
GPU_KV=${GPU_KV:-1.0}      # GiB of GPU KV, identical in every arm
CPU_KV=${CPU_KV:-32}       # GiB of CPU KV for the fastdecode arm
SESSIONS=${SESSIONS:-4}
OUT=${OUT:-results/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

echo "### 0. unit tests (CPU only)"
python -m pytest tests -q

echo "### 1. CPU attention probe (decides whether a speed-up is possible)"
python scripts/probe_cpu_attn.py --threads "$THREADS" --ctx 512 --batch 64 | tee "$OUT/probe.txt"

echo "### 2. token-level correctness under heavy offload"
python scripts/check_correctness.py --task tokens --model "$MODEL" \
  --gpu-kv-gib 0.25 --cpu-kv-gib 8 --cpu-threads "$THREADS" --limit 128 \
  --outdir "$OUT/correctness" | tee "$OUT/correctness.txt"

echo "### 3. throughput benchmark (ABBA sessions)"
python scripts/bench.py --model "$MODEL" --arms baseline,fastdecode \
  --sessions "$SESSIONS" --gpu-kv-gib "$GPU_KV" --cpu-kv-gib "$CPU_KV" \
  --cpu-threads "$THREADS" --max-num-seqs 128 --num-prompts 128 \
  --input-len 256 --output-len 512 --out "$OUT/bench.jsonl"

echo "### 4. statistics"
python scripts/analyze.py "$OUT/bench.jsonl" | tee "$OUT/analysis.txt"
echo "results in $OUT"
