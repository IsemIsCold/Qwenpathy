#!/usr/bin/env bash
# scripts/v4_train_session.sh -- EXP-030 boot 1 of 2: QLoRA training of V4 on 2 GPUs (≈ 2 h 10 min).
#
#   nohup bash /mnt/storage/v4_upload/scripts/v4_train_session.sh > /mnt/storage/out/v4_train_session.log 2>&1 &
#   tail -f /mnt/storage/out/v4_train_session.log          # Ctrl+C stops the tail, NOT the training
#
# Same recipe as T7 (EXP-019) except: decide-first targets, rare-class repeats, dev loss every 250 steps,
# checkpoints every 250 steps. Base = /root/base_qwen3_8b. Nothing in /root/participant is touched.
# If the boot dies: boot again with GPUs and run the same command -> resumes from the last checkpoint.
# When it prints DONE: SHUT DOWN (GPUs break after a training process exits); evaluate in a fresh boot.
set -uo pipefail
U=${U:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA=${DATA:-/mnt/storage/sft_v4}
RUN=${RUN:-/mnt/storage/runs/v4}
OUT=${OUT:-/mnt/storage/out/v4}
BASE=${BASE:-/root/base_qwen3_8b}
VENV=${VENV:-/opt/train_venv}
CFG_SRC=${CFG_SRC:-$U/experiments/exp030-v4-decide-first/configs/v4_train.yaml}
mkdir -p "$OUT"
say() { echo "[v4-train $(date +%H:%M:%S)] $*"; }
die() { say "STOP: $*"; echo "DONE -- FAIL (nothing trained); fix it or shut down"; exit 1; }

say "preflight"
[[ -f "$BASE/config.json" ]] || die "no base model at $BASE"
for f in train_v4_sharegpt.json dev200_sharegpt.json dataset_info.json; do [[ -f "$DATA/$f" ]] || die "missing $DATA/$f (unpack v4_data.tgz)"; done
[[ -x "$VENV/bin/llamafactory-cli" ]] || die "no $VENV/bin/llamafactory-cli"
if [[ ! -x "$VENV/bin/torchrun" ]]; then  # the shim from T2 (conda_env's torchrun cannot import llamafactory)
  printf '#!/usr/bin/env bash\nexec %s/bin/python -m torch.distributed.run "$@"\n' "$VENV" > "$VENV/bin/torchrun" && chmod +x "$VENV/bin/torchrun"
  say "created the torchrun shim in $VENV/bin"
fi
if [[ "${V4_SKIP_GPU_CHECK:-0}" != 1 ]]; then
  NGPU="$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)"
  [[ "${NGPU:-0}" -ge 2 ]] || die "GPUs detected: ${NGPU:-0} (NVML error = broken boot: shut down and boot again with GPUs)"
fi
mkdir -p "$(dirname "$RUN")"
FREE_GB=$(df -Pk "$(dirname "$RUN")" | awk 'NR==2 {printf "%d", $4/1048576}')
[[ "$FREE_GB" -ge 5 ]] || die "only ${FREE_GB} GB free for checkpoints"
N=$("$VENV/bin/python" -c "import json; print(len(json.load(open('$DATA/train_v4_sharegpt.json', encoding='utf-8'))))")
FIRST=$("$VENV/bin/python" -c "import json; print(json.load(open('$DATA/train_v4_sharegpt.json', encoding='utf-8'))[0]['conversations'][1]['value'][:18])")
say "data: $N examples, targets start with: $FIRST"
[[ "$FIRST" == '{"emotion_label":'* ]] || die "targets are not decide-first (wrong data folder?)"
if [[ -f "$RUN/adapter_model.safetensors" && -f "$RUN/all_results.json" ]]; then
  die "$RUN already holds a finished run: move it away first (mv $RUN ${RUN}_old)"
fi
LAST=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[[ -n "$LAST" ]] && say "resuming from $LAST"

sed -e "s#^dataset_dir: .*#dataset_dir: $DATA#" -e "s#^output_dir: .*#output_dir: $RUN#" \
    -e "s#^model_name_or_path: .*#model_name_or_path: $BASE#" "$CFG_SRC" > "$OUT/v4_train.yaml"
grep -E "^(model_name_or_path|dataset|eval_dataset|dataset_dir|output_dir|template|enable_thinking):" "$OUT/v4_train.yaml"

say "training starts; progress lines below every ~2 min (ETA ≈ 2 h 10 min)"
export PATH="$VENV/bin:$PATH" FORCE_TORCHRUN=1
llamafactory-cli train "$OUT/v4_train.yaml" > "$OUT/train.log" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  sleep "${V4_POLL_S:-120}"
  L=$(grep -aoE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" "$OUT/train.log" | tail -1)
  E=$(grep -aoE "'eval_loss': [0-9.]+" "$OUT/train.log" | tail -1)
  [[ -n "$L" ]] && say "step $L ${E:+| last $E}"
done
wait "$PID"; RC=$?
cp "$RUN"/trainer_log.jsonl "$RUN"/all_results.json "$OUT/" 2>/dev/null
if [[ $RC -ne 0 || ! -f "$RUN/adapter_model.safetensors" ]]; then
  tail -n 40 "$OUT/train.log"
  say "training failed (exit $RC). Checkpoints kept in $RUN: boot again and rerun this command to resume."
  echo "DONE -- FAIL, shut the instance down now"; exit 1
fi
"$VENV/bin/python" - "$RUN" <<'EOF'
import json, sys
run = sys.argv[1]
r = json.load(open(f"{run}/all_results.json"))
print(f"  train_loss {r.get('train_loss')}, runtime {r.get('train_runtime', 0) / 60:.0f} min, {r.get('train_samples_per_second')} samples/s")
ev = [json.loads(l) for l in open(f"{run}/trainer_log.jsonl") if '"eval_loss"' in l]
print("  dev loss by step:", ", ".join(f"{e['current_steps']}: {e['eval_loss']:.4f}" for e in ev) or "none logged")
EOF
say "adapter: $(ls "$RUN"/adapter_model.safetensors) ($(du -h "$RUN/adapter_model.safetensors" | cut -f1))"
echo "DONE -- PASS: training finished. SHUT DOWN NOW (GPUs break after training). Next: fresh 2-GPU boot -> v4_eval_session.sh"
