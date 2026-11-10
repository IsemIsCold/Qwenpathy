#!/usr/bin/env python3
"""
scripts/v4_gate.py -- EXP-030 pass rules: V4 (candidate) vs #4 (control) on the 600 val points.

Control = #4's own predictions on the same 600 points, rebuilt from the saved FP8 answers
(outputs/exp023/lp_fp8_t10.jsonl) with the EXP-027 parser = what #4 outputs (same weights, same parse).
The profile is not compared: in both, it comes from the same CPU classifier.

Rules (fixed Nov 9, before training):
  1. validate.py 0 errors, 0 parse failures, complete
  2. full-output fallbacks <= 6 (1% of 600)
  3. emotion: correct >= control + 12 (+2 points of accuracy) AND the 95% bootstrap CI of the difference
     (resampling conversations) is above 0
  4. reply ROUGE-L (eval.py, char, macro) >= control - 0.005
  5. latency mean of the first 100 <= 1.10 x 0.923 s (#4's own check run, Nov 8; 10% margin across boots)

    python v4_gate.py --sub OUT/v4_600/submission.jsonl --report OUT/v4_600/performance_report.json \
        --start-log OUT/v4_600_start.log --valid 1 --u /mnt/storage/v4_upload
Exit 0 = PASS, 3 = FAIL. Writes gate.json next to --sub.
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LAT_4 = 0.923


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--start-log", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--u", required=True, help="unpacked upload: participant/run_inference.py, scripts/eval.py, data/, outputs/")
    args = ap.parse_args()
    U = Path(args.u)
    ri = load("ri", U / "participant/run_inference.py")
    ev = load("ev", U / "scripts/eval.py")
    inputs = read_jsonl(U / "data/t10_input.jsonl")
    ids = [r["id"] for r in inputs]
    conv = np.array([r["conversation_id"] for r in inputs])
    gold = {r["id"]: r for p in ("data/val_gold_n100.jsonl", "data/val_gold_tune500.jsonl") for r in read_jsonl(U / p)}
    raw4 = {r["id"]: r["content"] for r in read_jsonl(U / "outputs/exp023/lp_fp8_t10.jsonl")}
    ctrl = {i: ri.parse_prediction(i, raw4[i])[0] for i in ids}
    cand = {r["id"]: r for r in read_jsonl(args.sub)}
    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    log = Path(args.start_log).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"stop after emotion: cut (\d+)/(\d+), full-output fallbacks (\d+)", log)
    fallbacks = int(m.group(3)) if m else None

    missing = [i for i in ids if i not in cand]
    if missing:
        print(f"  FAIL candidate misses {len(missing)} of the 600 ids")
        sys.exit(3)
    g = np.array([gold[i]["emotion_label"] for i in ids])
    a_ok = np.array([cand[i]["emotion_label"] for i in ids]) == g
    b_ok = np.array([ctrl[i]["emotion_label"] for i in ids]) == g
    rng = np.random.default_rng(0)
    uniq = np.unique(conv)
    rows = {c: np.flatnonzero(conv == c) for c in uniq}
    diffs = []
    for _ in range(2000):
        pick = np.concatenate([rows[c] for c in rng.choice(uniq, size=len(uniq))])
        diffs.append(a_ok[pick].mean() - b_ok[pick].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    rouge_a = ev.rouge_l_macro([cand[i]["response_text"] for i in ids], [gold[i]["response_text"] for i in ids])
    rouge_b = ev.rouge_l_macro([ctrl[i]["response_text"] for i in ids], [gold[i]["response_text"] for i in ids])
    lat = rep["average_latency_ms"] / 1000
    print(f"  emotion: V4 {a_ok.sum()}/600 = {a_ok.mean():.4f} vs #4 {b_ok.sum()}/600 = {b_ok.mean():.4f}  "
          f"(V4 only right {only_a}, #4 only right {only_b}; diff CI95 {lo:+.3f}…{hi:+.3f})")
    print(f"  ROUGE-L: V4 {rouge_a:.4f} vs #4 {rouge_b:.4f} | latency first 100: {lat:.3f} s (median {rep['median_latency_ms'] / 1000:.3f}) "
          f"| fallbacks {fallbacks} | parse failures {rep['parse_failure_count']}")
    sup = Counter(g)
    print("  recall by class (support): " + ", ".join(
        f"{c} {np.mean(a_ok[g == c]):.2f} vs {np.mean(b_ok[g == c]):.2f} ({sup[c]})" for c, _ in sup.most_common(8)))
    pred = Counter(cand[i]["emotion_label"] for i in ids)
    print(f"  V4 predicts anxiety on {pred['anxiety']}/600 (#4: {sum(ctrl[i]['emotion_label'] == 'anxiety' for i in ids)}; gold {sup['anxiety']})")
    checks = [
        ("validate.py 0 errors", args.valid == "1", args.valid == "1"),
        ("parse failures 0, complete", rep["parse_failure_count"] == 0 and rep["complete"], f"{rep['parse_failure_count']}, {rep['complete']}"),
        ("full-output fallbacks <= 6", fallbacks is not None and fallbacks <= 6, fallbacks),
        ("emotion >= #4 + 12 correct", a_ok.sum() >= b_ok.sum() + 12, f"{a_ok.sum()} vs {b_ok.sum()} + 12"),
        ("emotion diff CI95 above 0", lo > 0, f"{lo:+.4f}"),
        ("ROUGE-L >= #4 - 0.005", rouge_a >= rouge_b - 0.005, f"{rouge_a:.4f} vs {rouge_b:.4f}"),
        ("latency <= 1.10 x 0.923 s", lat <= 1.10 * LAT_4, f"{lat:.3f} s"),
    ]
    for name, ok, value in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<34} {value}")
    ok = all(c[1] for c in checks)
    Path(args.sub).with_name("gate.json").write_text(json.dumps({
        "pass": ok, "v4_correct": int(a_ok.sum()), "ctrl_correct": int(b_ok.sum()), "ci95": [float(lo), float(hi)],
        "rouge_v4": rouge_a, "rouge_ctrl": rouge_b, "latency_s": lat, "fallbacks": fallbacks}, indent=2), encoding="utf-8")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
