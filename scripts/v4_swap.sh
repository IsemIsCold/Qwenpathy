#!/usr/bin/env bash
# scripts/v4_swap.sh -- install V4 by hand + check run (captain's decision, Nov 10 evening).
# V4 tied #4 on the 600 val points and FAILED the pre-registered emotion rule (476 vs 473); this script only checks
# that plain start.sh produces valid output with V4's weights. Any failure puts #4's weights back.
set -uo pipefail
if [[ "${V4_TEST:-0}" != 1 ]]; then source activate conda_env; fi
P=${P:-/root/participant}; S=${S:-/mnt/storage}; V=${V:-$S/val}; U=${U:-$S/v4_upload}; O=${O:-$S/out/v4}; PY=${PY:-python3}
NEW=${NEW:-$S/merged_v4}; KEEP=${KEEP:-$S/model_t7}
say() { echo "[v4-swap $(date +%H:%M:%S)] $*"; }
stop() { say "$*"; echo "DONE -- FAIL (nothing changed, #4 still installed)"; exit 1; }
[[ -f "$NEW/config.json" ]] || stop "no V4 model at $NEW"
[[ -f "$P/model/config.json" ]] || stop "no model in $P/model"
[[ ! -e "$KEEP" && ! -e "$P/model_v4" ]] || stop "$KEEP or $P/model_v4 already exists"
for f in "$V/t9_input.jsonl" "$V/val_gold_n100.jsonl" "$U/scripts/validate.py" "$U/scripts/eval.py"; do [[ -f "$f" ]] || stop "missing $f"; done
mkdir -p "$O"
say "1/4 copying V4 into $P/model_v4 (a few minutes)"
cp -r "$NEW" "$P/model_v4" || { rm -rf "$P/model_v4"; stop "copy failed"; }
A=$(cd "$NEW" && cat *.safetensors | sha256sum | cut -c1-16); B=$(cd "$P/model_v4" && cat *.safetensors | sha256sum | cut -c1-16)
[[ "$A" == "$B" ]] || { rm -rf "$P/model_v4"; stop "copy differs from the original ($A vs $B)"; }
say "copy verified ($A)"
say "2/4 moving #4's weights to $KEEP, V4 into place"
mv "$P/model" "$KEEP" || stop "could not move #4's weights (check $P/model and $KEEP by hand)"
mv "$P/model_v4" "$P/model"
back() {
  say "putting #4's weights back"
  mv "$P/model" "$P/model_v4_rejected" && cp -r "$KEEP" "$P/model" && rm -rf "$P/model_v4_rejected" && say "#4's weights restored"
}
say "3/4 check run: plain start.sh on t9 (100 points, ~4 min)"
rm -rf "$O/check_t9"
OK=1
if bash "$P/start.sh" "$V/t9_input.jsonl" "$O/check_t9" > "$O/check_t9_start.log" 2>&1; then
  grep -a -E "GPUs detected|stop after emotion|profile classifier" "$O/check_t9_start.log"
  "$PY" "$U/scripts/validate.py" --input "$V/t9_input.jsonl" "$O/check_t9/submission.jsonl" > "$O/check_t9/validate.txt" 2>&1 && VALID=1 || VALID=0
  "$PY" "$U/scripts/eval.py" "$O/check_t9/submission.jsonl" "$V/val_gold_n100.jsonl" --out "$O/check_t9/eval_n100.json" --history= > "$O/check_t9/eval.txt" 2>&1
  "$PY" - "$O/check_t9" "$VALID" "$O/check_t9_start.log" <<'EOF'
import json, re, sys
out, valid, log = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
r = json.load(open(f"{out}/performance_report.json"))
m = json.load(open(f"{out}/eval_n100.json"))
rouge = m["response"]["rouge_l"]
rouge = rouge if isinstance(rouge, (int, float)) else rouge.get("score", 0)
t = re.search(r"stop after emotion: cut (\d+)/(\d+), full-output fallbacks (\d+)", open(log, encoding="utf-8", errors="replace").read())
fb = int(t.group(3)) if t else None
acc = m["emotion"]["accuracy"]
print(f"  V4 check run: emotion {acc:.4f} (#4: 0.8400)  profile exact {m['profile']['exact_match_avg']:.4f} (#4: 0.5267)  "
      f"ROUGE-L {rouge:.4f} (#4: 0.2756)  latency {r['average_latency_ms'] / 1000:.3f} s (#4: 0.923)")
checks = [("validate.py 0 errors", valid), ("parse failures 0", r["parse_failure_count"] == 0), ("complete", bool(r["complete"])),
          ("stop after emotion ran, fallbacks <= 2", fb is not None and fb <= 2), ("emotion n100 >= 0.75 (sanity)", acc >= 0.75)]
for name, ok in checks:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
sys.exit(0 if all(ok for _, ok in checks) else 3)
EOF
  OK=$?
else
  say "start.sh FAILED"; tail -n 30 "$O/check_t9_start.log"
fi
if [[ $OK -ne 0 ]]; then back; echo "DONE -- FAIL (#4's weights are back): do NOT upload"; exit 0; fi
say "4/4 cleanup"
if [[ "${V4_TEST:-0}" != 1 ]]; then rm -rf /root/.cache/* /root/.triton /tmp/* /*.tgz 2>/dev/null; df -h / | tail -1; fi
say "installed: $(sha256sum "$P/run_inference.py" "$P/configs/inference_config.json" "$P/start.sh" | awk '{print substr($1,1,12)}' | xargs), model = V4"
say "undo: mv $P/model $P/model_v4_rejected && cp -r $KEEP $P/model && rm -rf $P/model_v4_rejected"
echo "DONE -- PASS: V4 installed. Check the df line (Used <= 21G), shut down, then upload and submit"
