#!/usr/bin/env bash
# scripts/exp027_session.sh -- EXP-027: check run of submission #4 (= #3 + stop after emotion_label +
# profile from the CPU classifier with exact-match thresholds) in ONE 2-GPU boot.
#
#   bash /mnt/storage/exp027_upload/scripts/exp027_session.sh 2>&1 | tee /mnt/storage/out/exp027_session.log
#
# 1. Run A: plain `bash /root/participant/start.sh t9_input.jsonl ...` with #3 as installed (same-boot baseline).
# 2. Install the candidate (backup of #3 in /root/sub3_backup_exp027; only run_inference.py + the config change).
# 3. Run B: the same plain start.sh = exactly what gets submitted as #4.
# 4. validate.py + eval.py on n100 for both, then the pass rules below. FAIL -> #3's files are restored.
#
# Pass rules (fixed Nov 8, before the run; the Oct 16 draft asked for >= 10% lower latency, but the organisers
# have since confirmed exact-match profile scoring, so the profile gain alone is worth ~ +1.3 points and #4
# only has to be no slower):
#   validate.py 0 errors, 0 parse failures, complete, full-output fallbacks <= 2
#   emotion n100 >= 0.80 and >= run A - 0.02;  ROUGE-L n100 >= run A - 0.005
#   profile per-field exact n100 >= run A;      latency mean <= 1.05 x run A
set -uo pipefail
if [[ "${EXP027_TEST:-0}" != 1 ]]; then source activate conda_env; fi

P=${P:-/root/participant}
U=${U:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
V=${V:-/mnt/storage/val}
OUT=${OUT:-/mnt/storage/out/exp027}
BACKUP=${BACKUP:-/root/sub3_backup_exp027}
VARIANT=${VARIANT:-exact}
PY=${PY:-python3}
T9=$V/t9_input.jsonl
GOLD=$V/val_gold_n100.jsonl
# submission #3 as found on Migu on Nov 8 (run_inference.py, inference_config.json, start.sh)
EXPECT="ec3cfdf5f540 fc4cd8a0b159 a7698a02c349"
mkdir -p "$OUT"
say() { echo "[exp027 $(date +%H:%M:%S)] $*"; }

for f in "$T9" "$GOLD" "$P/start.sh" "$P/model/config.json" "$P/profile_clf/profile_clf.json" \
         "$U/participant/run_inference.py" "$U/scripts/exp027_install.py" "$U/scripts/validate.py" "$U/scripts/eval.py"; do
  [[ -f "$f" ]] || { say "MISSING: $f -- nothing changed"; echo "DONE -- FAIL (nothing changed)"; exit 1; }
done
if [[ "${EXP027_SKIP_GPU_CHECK:-0}" != 1 ]]; then
  NGPU="$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)"
  [[ "${NGPU:-0}" -ge 2 ]] || { say "GPUs detected: ${NGPU:-0} -- reboot with GPUs"; echo "DONE -- FAIL (nothing changed)"; exit 1; }
fi
NOW="$(sha256sum "$P/run_inference.py" "$P/configs/inference_config.json" "$P/start.sh" | awk '{print substr($1,1,12)}' | xargs)"
say "installed files now: $NOW"
if [[ -d "$BACKUP" ]]; then
  say "backup $BACKUP already exists: restoring #3 first so run A is really #3"
  "$PY" "$U/scripts/exp027_install.py" --participant "$P" --backup "$BACKUP" --restore || { echo "DONE -- FAIL (restore failed, check $P by hand)"; exit 1; }
  NOW="$(sha256sum "$P/run_inference.py" "$P/configs/inference_config.json" "$P/start.sh" | awk '{print substr($1,1,12)}' | xargs)"
fi
if [[ "${EXP027_EXPECT_SHA:-1}" == 1 && "$NOW" != "$EXPECT" ]]; then
  say "installed files are not submission #3 ($EXPECT) -- stopping, nothing changed"; echo "DONE -- FAIL (nothing changed)"; exit 1
fi

run_start() {  # $1 = name
  rm -rf "$OUT/$1"
  if ! bash "$P/start.sh" "$T9" "$OUT/$1" > "$OUT/$1_start.log" 2>&1; then
    say "start.sh FAILED ($1)"; tail -n 40 "$OUT/$1_start.log"; return 1
  fi
  grep -a -E "GPUs detected|stop after emotion|profile classifier|^PASS" "$OUT/$1_start.log"
  "$PY" "$U/scripts/validate.py" --input "$T9" "$OUT/$1/submission.jsonl" > "$OUT/$1/validate.txt" 2>&1 \
    && echo 1 > "$OUT/$1/valid" || echo 0 > "$OUT/$1/valid"
  "$PY" "$U/scripts/eval.py" "$OUT/$1/submission.jsonl" "$GOLD" --out "$OUT/$1/eval_n100.json" --history= > "$OUT/$1/eval.txt" 2>&1
}

restore() {
  say "restoring submission #3's files"
  "$PY" "$U/scripts/exp027_install.py" --participant "$P" --backup "$BACKUP" --restore
}

say "== 1/3 run A: #3 as installed (same-boot baseline)"
run_start sub3 || { echo "DONE -- FAIL (nothing changed), shut the instance down now"; exit 1; }

say "== 2/3 install the candidate (variant $VARIANT)"
"$PY" "$U/scripts/exp027_install.py" --participant "$P" --backup "$BACKUP" \
  --new-run-inference "$U/participant/run_inference.py" --variant "$VARIANT" || { echo "DONE -- FAIL (install failed)"; exit 1; }

say "== 3/3 run B: the candidate = what gets submitted as #4"
if ! run_start cand; then restore; echo "DONE -- FAIL (submission #3 restored), shut the instance down now"; exit 0; fi

"$PY" - "$OUT" "$OUT/cand_start.log" <<'EOF'
import json, re, sys
out, log = sys.argv[1], sys.argv[2]
def load(name):
    r = json.load(open(f"{out}/{name}/performance_report.json"))
    m = json.load(open(f"{out}/{name}/eval_n100.json"))
    return {"valid": open(f"{out}/{name}/valid").read().strip() == "1", "parse": r["parse_failure_count"],
            "complete": r["complete"], "lat": r["average_latency_ms"] / 1000, "med": r["median_latency_ms"] / 1000,
            "p95": r["p95_latency_ms"] / 1000, "acc": m["emotion"]["accuracy"], "exact": m["profile"]["exact_match_avg"],
            "f1": m["profile"]["micro_f1_avg"], "rouge": m["response"]["rouge_l"]}
a, b = load("sub3"), load("cand")
text = open(log, encoding="utf-8", errors="replace").read()
m = re.search(r"stop after emotion: cut (\d+)/(\d+), full-output fallbacks (\d+)", text)
fallbacks = int(m.group(3)) if m else None
rouge = lambda x: x["rouge"] if isinstance(x["rouge"], (int, float)) else x["rouge"].get("score", 0)
print(f"  run A (#3):        emotion {a['acc']:.4f}  profile exact {a['exact']:.4f}  micro-F1 {a['f1']:.4f}  ROUGE-L {rouge(a):.4f}  latency {a['lat']:.3f} s (median {a['med']:.3f}, p95 {a['p95']:.3f})")
print(f"  run B (candidate): emotion {b['acc']:.4f}  profile exact {b['exact']:.4f}  micro-F1 {b['f1']:.4f}  ROUGE-L {rouge(b):.4f}  latency {b['lat']:.3f} s (median {b['med']:.3f}, p95 {b['p95']:.3f})")
checks = [
    ("validate.py 0 errors", b["valid"], b["valid"]),
    ("parse failures 0", b["parse"] == 0, b["parse"]),
    ("complete (first 100 timed)", b["complete"], b["complete"]),
    ("stop after emotion ran, fallbacks <= 2", fallbacks is not None and fallbacks <= 2, m.group(0) if m else "line missing"),
    ("emotion n100 >= 0.80 and >= run A - 0.02", b["acc"] >= 0.80 and b["acc"] >= a["acc"] - 0.02, f"{b['acc']:.4f} vs {a['acc']:.4f}"),
    ("ROUGE-L n100 >= run A - 0.005", rouge(b) >= rouge(a) - 0.005, f"{rouge(b):.4f} vs {rouge(a):.4f}"),
    ("profile exact n100 >= run A", b["exact"] >= a["exact"], f"{b['exact']:.4f} vs {a['exact']:.4f}"),
    ("latency mean <= 1.05 x run A", b["lat"] <= 1.05 * a["lat"], f"{b['lat']:.3f} s vs {a['lat']:.3f} s ({b['lat'] / a['lat'] - 1:+.0%})"),
]
for name, ok, value in checks:
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<42} {value}")
json.dump({"run_a": a, "run_b": b, "fallbacks": fallbacks, "pass": all(c[1] for c in checks)}, open(f"{out}/gate.json", "w"), indent=2)
sys.exit(0 if all(c[1] for c in checks) else 3)
EOF
RESULT=$?
if [[ $RESULT -ne 0 ]]; then
  restore
  echo "DONE -- FAIL (submission #3 restored), shut the instance down now"
  exit 0
fi
say "installed files: $(sha256sum "$P/run_inference.py" "$P/configs/inference_config.json" | awk '{print substr($1,1,12)}' | xargs)"
say "undo later if needed: $PY $U/scripts/exp027_install.py --participant $P --backup $BACKUP --restore"
echo "DONE -- PASS: keep it installed, shut down, then 重新上传 -> 去提交 = submission #4"
