#!/usr/bin/env bash
# scripts/v4_eval_session.sh -- EXP-030 boot 2 of 2 (FRESH 2-GPU boot after training): merge, score, swap, check.
#
#   bash /mnt/storage/v4_upload/scripts/v4_eval_session.sh 2>&1 | tee /mnt/storage/out/v4_eval_session.log
#
# 1. Merge base + V4 adapter -> /root/merged/v4 (CPU, as for T7; skipped if already merged).
# 2. Score V4 on the 600 val points with #4's code and config (stop after emotion, CPU profile classifier):
#    plain start.sh with VLLM_MODEL_DIR=/root/merged/v4; then the pass rules in scripts/v4_gate.py.
# 3. PASS -> swap the model folders (#4's weights kept in /root/model_t7) and run the check run = plain start.sh
#    on t9, exactly what gets submitted. Any failure there swaps back. FAIL -> #4 stays installed untouched.
# ≈ 35 min ≈ 1.2 card-hours.
set -uo pipefail
if [[ "${V4_TEST:-0}" != 1 ]]; then source activate conda_env; fi
U=${U:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
P=${P:-/root/participant}
V=${V:-/mnt/storage/val}
RUN=${RUN:-/mnt/storage/runs/v4}
OUT=${OUT:-/mnt/storage/out/v4}
BASE=${BASE:-/root/base_qwen3_8b}
MERGED=${MERGED:-/root/merged/v4}
KEEP=${KEEP:-/root/model_t7}
VENV=${VENV:-/opt/train_venv}
PY=${PY:-python3}
EXPECT="e549ef94c12e ff85fc65a978 a7698a02c349"  # #4 = EXP-027 run_inference + config, FP8 start.sh
mkdir -p "$OUT"
say() { echo "[v4-eval $(date +%H:%M:%S)] $*"; }
die() { say "STOP: $*"; echo "DONE -- FAIL ($1)"; exit 1; }

say "preflight"
[[ -f "$RUN/adapter_model.safetensors" && -f "$RUN/adapter_config.json" ]] || die "no finished adapter in $RUN"
[[ -f "$BASE/config.json" ]] || die "no base model at $BASE"
for f in participant/run_inference.py scripts/eval.py scripts/validate.py scripts/v4_gate.py data/t10_input.jsonl \
         data/val_gold_n100.jsonl data/val_gold_tune500.jsonl outputs/exp023/lp_fp8_t10.jsonl; do
  [[ -f "$U/$f" ]] || die "missing $U/$f"
done
[[ -f "$V/t9_input.jsonl" ]] || die "missing $V/t9_input.jsonl"
if [[ "${V4_SKIP_GPU_CHECK:-0}" != 1 ]]; then
  NGPU="$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)"
  [[ "${NGPU:-0}" -ge 2 ]] || die "GPUs detected: ${NGPU:-0} -- shut down and boot again with GPUs"
fi
NOW="$(sha256sum "$P/run_inference.py" "$P/configs/inference_config.json" "$P/start.sh" | awk '{print substr($1,1,12)}' | xargs)"
say "installed: $NOW"
[[ "${V4_EXPECT_SHA:-1}" != 1 || "$NOW" == "$EXPECT" ]] || die "installed files are not #4 ($EXPECT)"
[[ ! -e "$KEEP" ]] || die "$KEEP already exists (an earlier swap?): check by hand"

if [[ -f "$MERGED/config.json" && $(ls "$MERGED"/*.safetensors 2>/dev/null | wc -l) -ge 1 ]]; then
  say "== 1/3 merge: already done ($MERGED)"
else
  FREE_GB=$(df -Pk /root | awk 'NR==2 {printf "%d", $4/1048576}')
  [[ "$FREE_GB" -ge 18 ]] || die "only ${FREE_GB} GB free on /root, the merged model needs ~17 GB"
  say "== 1/3 merge base + adapter -> $MERGED (CPU, ~10-15 min)"
  sed -e "s#^model_name_or_path: .*#model_name_or_path: $BASE#" -e "s#^adapter_name_or_path: .*#adapter_name_or_path: $RUN#" \
      -e "s#^export_dir: .*#export_dir: $MERGED#" "$U/experiments/exp030-v4-decide-first/configs/v4_merge.yaml" > "$OUT/v4_merge.yaml"
  PATH="$VENV/bin:$PATH" llamafactory-cli export "$OUT/v4_merge.yaml" > "$OUT/merge.log" 2>&1 || { tail -n 30 "$OUT/merge.log"; die "merge failed"; }
  [[ -f "$MERGED/config.json" ]] || die "merge wrote no config.json"
fi
say "merged: $(ls "$MERGED"/*.safetensors | wc -l) shards, $(du -sh "$MERGED" | cut -f1)"

say "== 2/3 V4 on the 600 val points (#4's code and config, VLLM_MODEL_DIR=$MERGED)"
rm -rf "$OUT/v4_600"
if ! VLLM_MODEL_DIR="$MERGED" VLLM_SERVED_MODEL_NAME=model bash "$P/start.sh" "$U/data/t10_input.jsonl" "$OUT/v4_600" > "$OUT/v4_600_start.log" 2>&1; then
  tail -n 40 "$OUT/v4_600_start.log"; die "start.sh failed on the 600 points (#4 untouched)"
fi
grep -a -E "GPUs detected|stop after emotion|profile classifier" "$OUT/v4_600_start.log"
"$PY" "$U/scripts/validate.py" --input "$U/data/t10_input.jsonl" "$OUT/v4_600/submission.jsonl" > "$OUT/v4_600/validate.txt" 2>&1 && VALID=1 || VALID=0
"$PY" "$U/scripts/v4_gate.py" --sub "$OUT/v4_600/submission.jsonl" --report "$OUT/v4_600/performance_report.json" \
  --start-log "$OUT/v4_600_start.log" --valid "$VALID" --u "$U"
if [[ $? -ne 0 ]]; then
  say "V4 did not pass its rules: #4 stays installed (untouched). Merged V4 kept in $MERGED for the record."
  echo "DONE -- FAIL (V4 rejected; #4 stays), shut the instance down now"; exit 0
fi

say "== 3/3 swap the model and run the check run (plain start.sh on t9 = what gets submitted)"
mv "$P/model" "$KEEP" && mv "$MERGED" "$P/model" || die "swap failed: check $P/model, $KEEP, $MERGED by hand"
swap_back() { say "swapping back to #4's model"; mv "$P/model" "$MERGED" && mv "$KEEP" "$P/model"; }
rm -rf "$OUT/check_t9"
if ! bash "$P/start.sh" "$V/t9_input.jsonl" "$OUT/check_t9" > "$OUT/check_t9_start.log" 2>&1; then
  tail -n 30 "$OUT/check_t9_start.log"; swap_back; echo "DONE -- FAIL (check run failed; #4 restored), shut down now"; exit 0
fi
"$PY" "$U/scripts/validate.py" --input "$V/t9_input.jsonl" "$OUT/check_t9/submission.jsonl" > "$OUT/check_t9/validate.txt" 2>&1 && CV=1 || CV=0
"$PY" - "$OUT/check_t9" "$CV" <<'EOF'
import json, sys
out, valid = sys.argv[1], sys.argv[2] == "1"
r = json.load(open(f"{out}/performance_report.json"))
ok = valid and r["parse_failure_count"] == 0 and r["complete"]
print(f"  check run: validate {'0 errors' if valid else 'ERRORS'}, parse failures {r['parse_failure_count']}, complete {r['complete']}, "
      f"latency {r['average_latency_ms'] / 1000:.3f} s")
sys.exit(0 if ok else 3)
EOF
if [[ $? -ne 0 ]]; then swap_back; echo "DONE -- FAIL (check run invalid; #4 restored), shut down now"; exit 0; fi
say "installed model: V4 ($(du -sh "$P/model" | cut -f1)); #4's weights kept in $KEEP"
say "undo: mv $P/model $MERGED && mv $KEEP $P/model"
echo "DONE -- PASS: V4 installed. Shut down, then 重新上传 -> 去提交"
