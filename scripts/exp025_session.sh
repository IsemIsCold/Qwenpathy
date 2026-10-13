#!/usr/bin/env bash
# scripts/exp025_session.sh -- EXP-025: check run of submission #3 (= #2 + profile blend) in ONE 2-GPU boot.
#
#   bash /mnt/storage/exp025_upload/scripts/exp025_session.sh
#
# 1. Install the candidate into /root/participant (backup of #2 in /root/sub2_backup_exp025).
#    Only run_inference.py, profile_clf/ and one config key change; start.sh and the model don't.
# 2. Check run = plain `bash /root/participant/start.sh t9_input.jsonl ...` (exactly what gets submitted).
# 3. validate.py, eval.py on n100, latency of the first 100, and the pass/fail check fixed in advance.
# FAIL -> submission #2's files are restored automatically. Shut the instance down after DONE.
set -euo pipefail
source activate conda_env

P=${P:-/root/participant}
U=${U:-/mnt/storage/exp025_upload}
V=${V:-/mnt/storage/val}
OUT=${OUT:-/mnt/storage/out/exp025}
BACKUP=${BACKUP:-/root/sub2_backup_exp025}
T9=$V/t9_input.jsonl
mkdir -p "$OUT"

for f in "$T9" "$V/val_gold_n100.jsonl" "$P/start.sh" "$P/model/config.json" "$U/participant/run_inference.py" \
         "$U/participant/profile_clf/profile_clf.json" "$U/participant/profile_clf/profile_clf_1.npz" "$U/participant/profile_clf/profile_clf_2.npz" \
         "$U/scripts/exp025_install.py" "$U/scripts/validate.py" "$U/scripts/eval.py"; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done
NGPU="$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)"
[[ "${NGPU:-0}" -ge 2 ]] || { echo "GPUs detected: ${NGPU:-0} -- shut down and reboot before anything else" >&2; exit 1; }
python3 -c "import numpy; print('numpy', numpy.__version__)"
echo "submission #2 files now: $(sha256sum "$P/run_inference.py" "$P/start.sh" "$P/configs/inference_config.json" | awk '{print substr($1,1,12)}' | xargs)"

echo "== 1/3 install the candidate"
python3 "$U/scripts/exp025_install.py" --participant "$P" --backup "$BACKUP" \
  --new-run-inference "$U/participant/run_inference.py" --classifier-dir "$U/participant/profile_clf"

restore() {
  echo "restoring submission #2's files"
  python3 "$U/scripts/exp025_install.py" --participant "$P" --backup "$BACKUP" --restore
}

echo "== 2/3 check run: bash $P/start.sh t9_input (exactly what gets submitted)"
rm -rf "$OUT/t9"
if ! bash "$P/start.sh" "$T9" "$OUT/t9" > "$OUT/t9_start.log" 2>&1; then
  echo "start.sh FAILED" >&2; tail -n 40 "$OUT/t9_start.log" >&2; restore; echo "DONE -- FAIL, shut the instance down now"; exit 1
fi
grep -a -E "GPUs detected|profile classifier|PASS" "$OUT/t9_start.log"

echo "== 3/3 validate + score + check"
python3 "$U/scripts/validate.py" --input "$T9" "$OUT/t9/submission.jsonl" > "$OUT/t9/validate.txt" 2>&1 && VALID=1 || VALID=0
tail -n 3 "$OUT/t9/validate.txt"
python3 "$U/scripts/eval.py" "$OUT/t9/submission.jsonl" "$V/val_gold_n100.jsonl" --out "$OUT/t9/eval_n100.json" --history= \
  | grep -E "accuracy \(OFFICIAL\)|AVERAGE|BLEU-4" || true
set +e
python3 - "$OUT/t9" "$VALID" <<'EOF'
import json, sys
out, valid = sys.argv[1], sys.argv[2] == "1"
r = json.load(open(f"{out}/performance_report.json"))
m = json.load(open(f"{out}/eval_n100.json"))
acc, prof = m["emotion"]["accuracy"], m["profile"]["micro_f1_avg"]
lat = r["average_latency_ms"] / 1000
checks = [  # fixed Oct 13, before the run (T9 = submission #2's own check run: emotion 0.82, latency 1.55 s)
    ("validate.py 0 errors", valid, True),
    ("parse failures 0", r["parse_failure_count"] == 0, r["parse_failure_count"]),
    ("complete (first 100 timed)", r["complete"], r["complete"]),
    ("emotion accuracy n100 >= 0.80", acc >= 0.80, round(acc, 4)),
    ("profile micro-F1 n100 >= 0.4806 (#2)", prof >= 0.4806, round(prof, 4)),
    ("latency mean <= 1.65 s", lat <= 1.65, f"{lat:.3f} s (median {r['median_latency_ms'] / 1000:.3f}, p95 {r['p95_latency_ms'] / 1000:.3f})"),
]
for name, ok, value in checks:
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<40} {value}")
ok = all(c[1] for c in checks)
print("PASS -> keep it installed, shut down, save image, submit #3" if ok else "FAIL")
sys.exit(0 if ok else 3)
EOF
RESULT=$?
set -e
if [[ $RESULT -ne 0 ]]; then
  restore
  echo "DONE -- FAIL (submission #2 restored), shut the instance down now"
  exit 0
fi
echo "installed files: $(sha256sum "$P/run_inference.py" "$P/configs/inference_config.json" "$P/profile_clf/profile_clf_1.npz" "$P/profile_clf/profile_clf_2.npz" | awk '{print substr($1,1,12)}' | xargs)"
echo "undo later if needed: python3 $U/scripts/exp025_install.py --participant $P --backup $BACKUP --restore"
echo "DONE -- PASS, shut the instance down now"
