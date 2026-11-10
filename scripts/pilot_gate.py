#!/usr/bin/env python3
"""
scripts/pilot_gate.py — P2 pre-flight gate (v2.0).

Run before any GPU experiment to catch misconfigurations that waste card-hours.
Exit code: 0 = PASS, 1 = FAIL.

Usage:
    python scripts/pilot_gate.py \
        --sub-n50 experiments/exp019-qlora-full/n50/submission.jsonl \
        --baseline-n50 experiments/exp005-prompt/v1_7/out_nothink/submission.jsonl \
        --sub-n50b experiments/exp019-qlora-full/n50b/submission.jsonl \
        --baseline-n50b experiments/exp005-prompt/v1_7/out_nothink/submission.jsonl \
        --gold-n50 data/val_gold_n50.jsonl \
        --gold-n50b data/val_gold_n50b.jsonl \
        --anxiety-recall 0.95

Checks (8 total):
  1. Submissions are parseable JSONL
  2. validate.py reports 0 errors, 0 parse failures
  3. No emotion collapse (>80% single label)
  4. Emotion accuracy >= baseline (or within tolerance)
  5. Anxiety recall >= threshold (default 0.95)
  6. Reply length median within +/-30% of baseline
  7. All required files exist (FAIL if missing, not skip)
  8. UTF-8 BOM handled correctly
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval as eval_mod


def load_jsonl_dict(path):
    records = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid is not None:
                records[rid] = rec
    return records


def safe_response_text(rec):
    text = rec.get("response_text")
    if text is None or isinstance(text, (int, float)):
        return ""
    return str(text)


def check_parseable(path, label):
    try:
        records = load_jsonl_dict(path)
        return True, f"{label}: {len(records)} records parseable"
    except Exception as e:
        return False, f"{label}: parse error -- {e}"


def check_validation(sub_path, gold_path):
    try:
        sub = load_jsonl_dict(sub_path)
        gold = load_jsonl_dict(gold_path)

        errors = []
        for rid in sorted(gold.keys()):
            if rid not in sub:
                errors.append(f"{rid}: missing in submission")
                continue
            srec = sub[rid]

            for field in ["emotion_label", "response_text", "user_profile"]:
                if field not in srec:
                    errors.append(f"{rid}: missing field '{field}'")

            emo = srec.get("emotion_label")
            if emo and emo not in eval_mod.EMOTIONS:
                errors.append(f"{rid}: invalid emotion_label '{emo}'")

            prof = srec.get("user_profile", {})
            if isinstance(prof, dict):
                for field in eval_mod.PROFILE_FIELDS:
                    val = prof.get(field)
                    if val is None:
                        errors.append(f"{rid}: profile missing '{field}'")
                    elif not isinstance(val, list):
                        errors.append(f"{rid}: profile '{field}' is not a list")
                    else:
                        valid_tags = eval_mod.PROFILE_VOCAB.get(field, set())
                        for tag in val:
                            if tag not in valid_tags:
                                errors.append(f"{rid}: invalid profile tag '{tag}' in {field}")

        if errors:
            return False, f"validation: {len(errors)} errors (first: {errors[0]})"
        return True, f"validation: 0 errors, {len(sub)} records"
    except Exception as e:
        return False, f"validation: exception -- {e}"


def check_emotion_collapse(sub_path):
    sub = load_jsonl_dict(sub_path)
    labels = [r.get("emotion_label") for r in sub.values() if r.get("emotion_label")]
    if not labels:
        return False, "emotion collapse: no labels found"
    c = Counter(labels)
    most_common = c.most_common(1)[0]
    pct = most_common[1] / len(labels) * 100
    ok = pct <= 80
    return ok, f"emotion collapse: {most_common[0]} = {pct:.1f}% (threshold 80%)"


def check_accuracy_vs_baseline(sub_path, baseline_path, gold_path, tolerance=0.0):
    sub = load_jsonl_dict(sub_path)
    baseline = load_jsonl_dict(baseline_path)
    gold = load_jsonl_dict(gold_path)

    ids = sorted(gold.keys())
    sub_pairs = [(sub.get(rid, {}).get("emotion_label"), gold[rid].get("emotion_label")) for rid in ids]
    base_pairs = [(baseline.get(rid, {}).get("emotion_label"), gold[rid].get("emotion_label")) for rid in ids]

    sub_acc = eval_mod.score_emotion(sub_pairs)["accuracy"]
    base_acc = eval_mod.score_emotion(base_pairs)["accuracy"]

    ok = sub_acc >= (base_acc - tolerance - 1e-9)
    return ok, f"accuracy: {sub_acc:.4f} vs baseline {base_acc:.4f} (tolerance {tolerance:.4f})"


def check_anxiety_recall(sub_path, gold_path, threshold):
    sub = load_jsonl_dict(sub_path)
    gold = load_jsonl_dict(gold_path)

    ids = sorted(gold.keys())
    pairs = [(sub.get(rid, {}).get("emotion_label"), gold[rid].get("emotion_label")) for rid in ids]
    result = eval_mod.score_emotion(pairs)
    anxiety_r = result["per_class"].get("anxiety", {}).get("recall", 0.0)
    ok = anxiety_r >= (threshold - 1e-9)
    return ok, f"anxiety recall: {anxiety_r:.4f} (threshold {threshold:.4f})"


def check_length_median(sub_path, baseline_path, tolerance=0.30):
    sub = load_jsonl_dict(sub_path)
    baseline = load_jsonl_dict(baseline_path)

    sub_lens = [len(safe_response_text(r)) for r in sub.values()]
    base_lens = [len(safe_response_text(r)) for r in baseline.values()]

    if not sub_lens or not base_lens:
        return False, "length: no texts to compare"

    sub_med = statistics.median(sub_lens)
    base_med = statistics.median(base_lens)

    if base_med == 0:
        return False, "length: baseline median is 0"

    ratio = abs(sub_med - base_med) / base_med
    ok = ratio <= tolerance
    return ok, f"length median: {sub_med:.0f} vs baseline {base_med:.0f} (ratio {ratio:.2f}, tolerance {tolerance:.2f})"


def check_files_exist(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        return False, f"missing files: {', '.join(missing)}"
    return True, f"all {len(paths)} files exist"


def check_utf8_bom(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = [l for l in f if l.strip()]
        return True, f"UTF-8 OK: {len(lines)} lines"
    except Exception as e:
        return False, f"UTF-8 error: {e}"


def main():
    ap = argparse.ArgumentParser(description="P2 pre-flight gate")
    ap.add_argument("--sub-n50", required=True)
    ap.add_argument("--baseline-n50", required=True)
    ap.add_argument("--sub-n50b", required=True)
    ap.add_argument("--baseline-n50b", required=True)
    ap.add_argument("--gold-n50", required=True)
    ap.add_argument("--gold-n50b", required=True)
    ap.add_argument("--anxiety-recall", type=float, default=0.95)
    ap.add_argument("--acc-tolerance", type=float, default=0.0)
    ap.add_argument("--length-tolerance", type=float, default=0.30)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    checks = []

    all_paths = [args.sub_n50, args.baseline_n50, args.sub_n50b, args.baseline_n50b,
                 args.gold_n50, args.gold_n50b]
    ok, msg = check_files_exist(all_paths)
    checks.append(("files_exist", ok, msg))

    for label, path in [("sub-n50", args.sub_n50), ("sub-n50b", args.sub_n50b),
                         ("baseline-n50", args.baseline_n50), ("baseline-n50b", args.baseline_n50b)]:
        ok, msg = check_parseable(path, label)
        checks.append((f"parse_{label}", ok, msg))

    ok, msg = check_validation(args.sub_n50, args.gold_n50)
    checks.append(("validate_n50", ok, msg))
    ok, msg = check_validation(args.sub_n50b, args.gold_n50b)
    checks.append(("validate_n50b", ok, msg))

    ok, msg = check_emotion_collapse(args.sub_n50)
    checks.append(("collapse_n50", ok, msg))
    ok, msg = check_emotion_collapse(args.sub_n50b)
    checks.append(("collapse_n50b", ok, msg))

    ok, msg = check_accuracy_vs_baseline(args.sub_n50, args.baseline_n50, args.gold_n50, args.acc_tolerance)
    checks.append(("acc_n50", ok, msg))
    ok, msg = check_accuracy_vs_baseline(args.sub_n50b, args.baseline_n50b, args.gold_n50b, args.acc_tolerance)
    checks.append(("acc_n50b", ok, msg))

    ok, msg = check_anxiety_recall(args.sub_n50, args.gold_n50, args.anxiety_recall)
    checks.append(("anxiety_n50", ok, msg))
    ok, msg = check_anxiety_recall(args.sub_n50b, args.gold_n50b, args.anxiety_recall)
    checks.append(("anxiety_n50b", ok, msg))

    ok, msg = check_length_median(args.sub_n50, args.baseline_n50, args.length_tolerance)
    checks.append(("length_n50", ok, msg))
    ok, msg = check_length_median(args.sub_n50b, args.baseline_n50b, args.length_tolerance)
    checks.append(("length_n50b", ok, msg))

    for label, path in [("sub-n50", args.sub_n50), ("sub-n50b", args.sub_n50b)]:
        ok, msg = check_utf8_bom(path)
        checks.append((f"utf8_{label}", ok, msg))

    all_ok = all(ok for _, ok, _ in checks)

    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"PILOT GATE v2.0 -- {'PASS' if all_ok else 'FAIL'}")
        print(f"{'='*60}")
        for name, ok, msg in checks:
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {name:<20} {msg}")
        print(f"{'='*60}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
