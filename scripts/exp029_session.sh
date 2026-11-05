#!/usr/bin/env bash
# EXP-029 session (one 2-GPU boot on Migu): [EXP-027 check run] -> hidden-state cache -> depth read-out probes.
#
#   bash /mnt/storage/exp029_upload/scripts/exp029_session.sh 2>&1 | tee /mnt/storage/out/exp029_session.log
#
# Resumable: if the boot dies, run the same command again; finished feature rows are kept (/root/exp029_feats).
# Nothing in /root/participant is changed by EXP-029 (the EXP-027 step changes it only through its own script).
# All settings can be overridden with environment variables (defaults below). Tests set EXP029_SKIP_GPU_CHECK=1.
set -uo pipefail

U=${U:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL=${MODEL:-/root/participant/model}
FEATS=${FEATS:-/root/exp029_feats}
OUT=${OUT:-/mnt/storage/out/exp029}
PY=${PY:-/opt/train_venv/bin/python}
SFT_TRAIN=${SFT_TRAIN:-/mnt/storage/sft/train_sharegpt.json}
SFT_DEV=${SFT_DEV:-/mnt/storage/sft/dev_sharegpt.json}
VAL_INPUT=${VAL_INPUT:-$U/data/t10_input.jsonl}
VAL_GOLD=${VAL_GOLD:-"$U/data/val_gold_n100.jsonl $U/data/val_gold_tune500.jsonl"}
LLM_ANSWERS=${LLM_ANSWERS:-$U/outputs/exp023/lp_fp8_t10.jsonl}
N_TRAIN=${N_TRAIN:-6000}
WORLD=${WORLD:-2}
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-75}
PROBE_DEVICE=${PROBE_DEVICE:-cuda}
RUN_EXP027=${RUN_EXP027:-auto}          # auto = run it if its script is on storage; 0 = skip; 1 = required
EXP027=${EXP027:-/mnt/storage/exp027_upload/scripts/exp027_session.sh}
VARIANT=${VARIANT:-exact}
EXTRACT="$U/scripts/exp029_extract.py"
PROBE="$U/scripts/exp029_probe.py"
T0=$(date +%s)

say() { echo "[exp029 $(date +%H:%M:%S)] $*"; }
die() { say "STOP: $*"; say "nothing was submitted; shut the instance down if you are not fixing this now"; exit 1; }
mkdir -p "$OUT"

# ---------------------------------------------------------------- 0. preflight (fails in seconds, not after an hour)
say "preflight"
[[ -f "$EXTRACT" && -f "$PROBE" ]] || die "scripts not found under $U/scripts"
[[ -f "$MODEL/config.json" ]] || die "no model at $MODEL"
for f in "$SFT_TRAIN" "$SFT_DEV" "$VAL_INPUT" "$LLM_ANSWERS" $VAL_GOLD; do [[ -f "$f" ]] || die "missing $f"; done
"$PY" -c "import torch, transformers, numpy; print('python ok: torch', torch.__version__, '| transformers', transformers.__version__)" \
  || die "$PY cannot import torch/transformers/numpy"
"$PY" -c "import accelerate" 2>/dev/null || say "note: accelerate missing -> model is loaded on CPU first (slower start, ~16 GB RAM per worker)"
if [[ "${EXP029_SKIP_GPU_CHECK:-0}" != 1 ]]; then
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found (no-GPU boot?)"
  NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU') || true
  [[ "$NGPU" -ge "$WORLD" ]] || die "need $WORLD GPUs, nvidia-smi sees ${NGPU:-0} (NVML error = broken GPU boot: reboot)"
  "$PY" -c "import torch; assert torch.cuda.device_count() >= $WORLD; [torch.zeros(1, device=f'cuda:{i}') for i in range($WORLD)]" \
    || die "torch cannot use $WORLD GPUs"
fi
FREE_GB=$(df -Pk "$(dirname "$FEATS")" | awk 'NR==2 {printf "%d", $4/1048576}')
say "free disk for features: ${FREE_GB} GB (need ~3 GB for N_TRAIN=6000)"
[[ "$FREE_GB" -ge 4 ]] || die "less than 4 GB free under $(dirname "$FEATS")"

# ---------------------------------------------------------------- 1. EXP-027 check run (submission #4 candidate)
if [[ "$RUN_EXP027" == 1 || ( "$RUN_EXP027" == auto && -f "$EXP027" ) ]]; then
  [[ -f "$EXP027" ]] || die "RUN_EXP027=1 but $EXP027 not found"
  say "EXP-027 check run first (VARIANT=$VARIANT); its log: $OUT/exp027_session.log"
  VARIANT=$VARIANT bash "$EXP027" 2>&1 | tee "$OUT/exp027_session.log"
  say "EXP-027 finished: $(grep -E '^DONE' "$OUT/exp027_session.log" | tail -1)"
  say "(ignore its 'shut the instance down' line: EXP-029 continues in this boot)"
  if [[ "${EXP029_SKIP_GPU_CHECK:-0}" != 1 ]]; then
    for _ in $(seq 1 36); do  # wait up to 3 min for vLLM to release the GPUs
      USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [[ "${USED:-99999}" -lt 2000 ]] && break
      sleep 5
    done
    [[ "${USED:-99999}" -lt 2000 ]] || die "GPUs still hold ${USED} MiB after EXP-027 (leftover vLLM?): check 'nvidia-smi', then rerun with RUN_EXP027=0"
  fi
else
  say "EXP-027 check run skipped (RUN_EXP027=$RUN_EXP027, script ${EXP027})"
fi

# ---------------------------------------------------------------- 2. plan (kept across reruns)
if [[ -f "$FEATS/plan.json" ]]; then
  say "resuming the existing cache in $FEATS ($("$PY" -c "import numpy; d=numpy.load('$FEATS/done.npy'); print(int(d.sum()), '/', len(d), 'rows done')"))"
else
  # shellcheck disable=SC2086
  "$PY" "$EXTRACT" plan --model "$MODEL" --out "$FEATS" --sft-train "$SFT_TRAIN" --sft-dev "$SFT_DEV" \
    --n-train "$N_TRAIN" --val-input "$VAL_INPUT" --val-gold $VAL_GOLD || die "plan failed"
fi
cp "$FEATS/plan.json" "$OUT/plan.json"

# ---------------------------------------------------------------- 3. extract: one worker per GPU
say "extracting on $WORLD worker(s); ETA is printed after 25 and 200 points per worker"
PIDS=(); declare -a LASTLINE=()
for ((r = 0; r < WORLD; r++)); do
  CUDA_VISIBLE_DEVICES=$r "$PY" "$EXTRACT" work --out "$FEATS" --rank "$r" --world "$WORLD" \
    --time-limit-min "$TIME_LIMIT_MIN" > "$OUT/extract_rank$r.log" 2>&1 &
  PIDS+=($!)
done
while :; do
  ALIVE=0
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && ALIVE=1; done
  [[ $ALIVE == 0 ]] && break
  sleep "${EXP029_POLL_S:-60}"
  for ((r = 0; r < WORLD; r++)); do
    L=$(grep -E '^\[rank' "$OUT/extract_rank$r.log" | tail -1)
    if [[ -n "$L" && "$L" != "${LASTLINE[$r]:-}" ]]; then say "$L"; LASTLINE[$r]=$L; fi
  done
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
for ((r = 0; r < WORLD; r++)); do grep -E '^\[rank|Error|error' "$OUT/extract_rank$r.log" | tail -3; done
"$PY" "$EXTRACT" check --out "$FEATS" || die "feature cache incomplete (worker failed=$FAIL): read $OUT/extract_rank*.log; rerun this script to resume"
cp "$FEATS"/extract_summary.json "$FEATS"/worker*.json "$OUT/" 2>/dev/null

# ---------------------------------------------------------------- 4. probes (GPU, CPU fallback)
say "fitting read-outs (device $PROBE_DEVICE)"
rm -f "$OUT/results.json"
if ! "$PY" "$PROBE" --feats "$FEATS" --llm-answers "$LLM_ANSWERS" --out "$OUT" --device "$PROBE_DEVICE" 2>&1 | tee "$OUT/probe.log"; then
  [[ "$PROBE_DEVICE" == cpu ]] && die "probe failed: read $OUT/probe.log"
  say "probe failed on $PROBE_DEVICE -> retrying on CPU (slower, ~30 min)"
  "$PY" "$PROBE" --feats "$FEATS" --llm-answers "$LLM_ANSWERS" --out "$OUT" --device cpu 2>&1 | tee "$OUT/probe_cpu.log" \
    || die "probe failed on CPU too: read $OUT/probe_cpu.log"
fi
[[ -f "$OUT/results.json" ]] || die "probes produced no results.json"

WALL=$(( $(date +%s) - T0 ))
say "report:"; cat "$OUT/report.md"
say "wall time $((WALL / 60)) min -> about $(awk -v s="$WALL" -v g="$WORLD" 'BEGIN {printf "%.2f", s / 3600 * g}') card-hours for this session"
say "results in $OUT (results.json, report.md, layer_curve.csv, alpha.csv, val_predictions.jsonl); features kept in $FEATS"
say "DONE -- shut the instance down now"
