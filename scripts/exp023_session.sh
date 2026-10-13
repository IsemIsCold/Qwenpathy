#!/usr/bin/env bash
# scripts/exp023_session.sh -- EXP-023: FP8 + anxiety bias, decided and checked in ONE 2-GPU boot.
#
#   bash /mnt/storage/exp023_upload/scripts/exp023_session.sh
#
# 1. FP8 server (= start_fp8.sh args): logprobs on t10_input (600 points = n100 + tune500).
# 2. tune_emotion_bias.py picks FP8's own b on tune500 (top-5 alternatives = what the
#    submission requests), exp023_gate.py applies the Oct 13 rule -> GO / NO-GO.
# 3. NO-GO: stop here, /root/participant untouched (submission #2 stays).
#    GO: back up #2's files to /root/sub2_backup, install the candidate (exp023_install.py),
#    then the check run: plain `bash /root/participant/start.sh t9_input.jsonl ...` (= exactly
#    what gets submitted), validate.py, eval.py on n100, latency of the first 100.
# Outputs in /mnt/storage/out/exp023/. Shut the instance down right after it prints DONE.
set -euo pipefail
source activate conda_env

P=${P:-/root/participant}
U=${U:-/mnt/storage/exp023_upload}      # exp023_upload.tgz unpacked here (repo layout)
V=${V:-/mnt/storage/val}
OUT=${OUT:-/mnt/storage/out/exp023}
BACKUP=${BACKUP:-/root/sub2_backup}
TOPK=${TOPK:-5}
T10=$V/t10_input.jsonl
T9=$V/t9_input.jsonl
G_TUNE=$U/data/val_gold_tune500.jsonl
BF16_REF=${BF16_REF:-$U/outputs/exp022/bf16/tune500_filter_on_b0.jsonl}
URL=http://127.0.0.1:8000
mkdir -p "$OUT"

for f in "$T10" "$T9" "$G_TUNE" "$BF16_REF" "$V/val_gold_n50.jsonl" "$V/val_gold_n50b.jsonl" \
         "$V/val_gold_n100.jsonl" "$P/model/config.json" "$P/start.sh" "$U/participant/run_inference.py" \
         "$U/scripts/emotion_logprobs.py" "$U/scripts/tune_emotion_bias.py" "$U/scripts/exp023_gate.py" \
         "$U/scripts/exp023_install.py" "$U/scripts/eval.py" "$U/scripts/validate.py"; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done
[[ "$(wc -l < "$T10")" -eq 600 ]] || { echo "t10_input.jsonl must have 600 lines" >&2; exit 1; }
NGPU="$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)"
[[ "${NGPU:-0}" -ge 2 ]] || { echo "GPUs detected: ${NGPU:-0} -- shut down and reboot before anything else" >&2; exit 1; }
echo "submission #2 files now: $(sha256sum "$P/run_inference.py" "$P/start.sh" | awk '{print substr($1,1,12)}' | xargs)"

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

echo "== 1/4 FP8 logprobs on t10 (600 points)"
start_server vllm_fp8 --quantization fp8
python3 "$U/scripts/emotion_logprobs.py" --participant "$P" --base-url "$URL/v1" \
  --input "$T10" --out "$OUT/lp_fp8_t10.jsonl" --workers 8
stop_server

echo "== 2/4 tune FP8's b on tune500 (top-$TOPK alternatives) + gate"
python3 "$U/scripts/tune_emotion_bias.py" --lp "$OUT/lp_fp8_t10.jsonl" --tune-gold "$G_TUNE" \
  --report-gold "$G_TUNE" "$V/val_gold_n50.jsonl" "$V/val_gold_n50b.jsonl" "$V/val_gold_n100.jsonl" \
  --top-k "$TOPK" --participant "$U/participant" --out-dir "$OUT/fp8"
set +e
python3 "$U/scripts/exp023_gate.py" --tuner-dir "$OUT/fp8" --gold "$G_TUNE" --bf16-ref "$BF16_REF" --out "$OUT/gate.json"
GATE=$?
set -e
if [[ $GATE -eq 3 ]]; then
  echo "NO-GO: submission #2 stays. /root/participant was not touched."
  echo "DONE -- shut the instance down now"
  exit 0
fi
[[ $GATE -eq 0 ]] || { echo "gate script failed (exit $GATE)" >&2; exit 1; }

echo "== 3/4 install the submission #3 candidate (#2 backed up in $BACKUP)"
B="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["b"])' "$OUT/gate.json")"
python3 "$U/scripts/exp023_install.py" --participant "$P" --backup "$BACKUP" \
  --new-run-inference "$U/participant/run_inference.py" --bias "$B" --top-logprobs "$TOPK" --quantization fp8

echo "== 4/4 check run: bash $P/start.sh t9_input (exactly what gets submitted)"
rm -rf "$OUT/t9"
bash "$P/start.sh" "$T9" "$OUT/t9" > "$OUT/t9_start.log" 2>&1 || { echo "start.sh failed" >&2; tail -n 40 "$OUT/t9_start.log" >&2; exit 1; }
grep -a -E "GPUs detected|emotion bias|PASS" "$OUT/t9_start.log"
python3 "$U/scripts/validate.py" --input "$T9" "$OUT/t9/submission.jsonl" | tail -n 5 \
  || echo "validate.py reported problems -- do NOT submit"
python3 "$U/scripts/eval.py" "$OUT/t9/submission.jsonl" "$V/val_gold_n100.jsonl" --out "$OUT/t9/eval_n100.json" --history= \
  | grep -E "accuracy \(OFFICIAL\)|AVERAGE|BLEU-4" || echo "eval.py failed"
python3 - "$OUT/t9/performance_report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"latency first {r['rounds']}: mean {r['average_latency_ms'] / 1000:.3f} s, median {r['median_latency_ms'] / 1000:.3f} s, "
      f"p95 {r['p95_latency_ms'] / 1000:.3f} s | complete {r['complete']} | parse failures {r['parse_failure_count']}")
EOF
echo "installed files: $(sha256sum "$P/run_inference.py" "$P/start.sh" "$P/configs/inference_config.json" | awk '{print substr($1,1,12)}' | xargs)"
echo "undo (back to submission #2): python3 $U/scripts/exp023_install.py --participant $P --backup $BACKUP --restore"
echo "DONE -- shut the instance down now"
