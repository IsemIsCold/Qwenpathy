#!/usr/bin/env bash
# scripts/exp022_session.sh -- EXP-022, the whole GPU session in one command (2-GPU boot).
#
#   bash /mnt/storage/scripts/exp022_session.sh
#
# 1. bf16 server (same vLLM args as start.sh on 2 GPUs): logprobs on t10_input (600 points),
#    then a 20-sample latency pair (plain vs logprobs, one request at a time).
# 2. FP8 server (= start_fp8.sh): logprobs on t9_input (n100).
# Outputs in /mnt/storage/out/exp022/. Shut the instance down right after it prints DONE.
set -euo pipefail
source activate conda_env

P=${P:-/root/participant}
S=${S:-/mnt/storage/scripts}
OUT=${OUT:-/mnt/storage/out/exp022}
T10=${T10:-/mnt/storage/val/t10_input.jsonl}
T9=${T9:-/mnt/storage/val/t9_input.jsonl}
URL=http://127.0.0.1:8000
mkdir -p "$OUT"
for f in "$T10" "$T9" "$S/emotion_logprobs.py" "$P/model/config.json"; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done
[[ "$(wc -l < "$T10")" -eq 600 ]] || { echo "t10_input.jsonl must have 600 lines" >&2; exit 1; }

PID=""
stop_server() {
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID"; wait "$PID" 2>/dev/null || true
  fi
  PID=""
}
trap stop_server EXIT

start_server() {  # $1 = log name, rest = extra vLLM args
  local log="$OUT/$1.log"; shift
  echo "starting vLLM ($*) -> $log"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm serve "$P/model" --host 127.0.0.1 --port 8000 \
    --served-model-name model --tensor-parallel-size 2 --max-model-len 32768 \
    --gpu-memory-utilization 0.90 "$@" > "$log" 2>&1 &
  PID=$!
  for ((s = 0; s < 600; s++)); do
    kill -0 "$PID" 2>/dev/null || { echo "vLLM died" >&2; tail -n 40 "$log" >&2; exit 1; }
    if python3 -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' "$URL/health" 2>/dev/null; then
      echo "ready after ${s}s"; return 0
    fi
    sleep 1
  done
  echo "vLLM not ready in 600s" >&2; tail -n 40 "$log" >&2; exit 1
}

LP="python3 $S/emotion_logprobs.py --participant $P --base-url $URL/v1"

start_server vllm_bf16
$LP --input "$T10" --out "$OUT/lp_bf16.jsonl" --workers 8
$LP --input "$T9" --out "$OUT/lat_plain.jsonl" --limit 20 --workers 1 --no-logprobs
$LP --input "$T9" --out "$OUT/lat_logprobs.jsonl" --limit 20 --workers 1
stop_server

start_server vllm_fp8 --quantization fp8
$LP --input "$T9" --out "$OUT/lp_fp8.jsonl" --workers 8
stop_server

wc -l "$OUT"/*.jsonl
echo "DONE -- shut the instance down now"
