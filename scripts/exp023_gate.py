#!/usr/bin/env python3
"""
scripts/exp023_gate.py -- EXP-023 go/no-go for submission #3 (FP8 + anxiety bias).

Rule, fixed on Oct 13 BEFORE any FP8 tune500 number was seen. On tune500, FP8 with
its own tuned b (filter on) must reach all three:
  1. emotion accuracy  >= 0.782   (bf16 submission #2, no bias, EXP-022)
  2. profile micro-F1  >= 0.4461  (bf16 0.4561 minus 1 point)
  3. ROUGE-L           >= bf16 ROUGE-L on the same 500 points minus 0.01
n50 / n50b / n100 are printed for information only.

    python3 exp023_gate.py --tuner-dir OUT/fp8 --gold val_gold_tune500.jsonl \
        --bf16-ref outputs/exp022/bf16/tune500_filter_on_b0.jsonl --out OUT/gate.json

Exit code: 0 = GO, 3 = NO-GO, anything else = error.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_eval():
    sys.path.insert(0, str(ROOT / "scripts"))  # eval.py imports validate.py
    spec = importlib.util.spec_from_file_location("eval_mod", ROOT / "scripts" / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def reply_scores(ev, sub_path, gold_rows):
    sub = {r["id"]: r for r in read_jsonl(sub_path)}
    missing = [g["id"] for g in gold_rows if g["id"] not in sub]
    if missing:
        sys.exit(f"{sub_path}: {len(missing)} gold ids missing (first {missing[0]})")
    pred = [sub[g["id"]].get("response_text") or "" for g in gold_rows]
    gold = [g.get("response_text") or "" for g in gold_rows]
    return ev.rouge_l_macro(pred, gold), ev.bleu4_corpus(pred, gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuner-dir", required=True, help="--out-dir of tune_emotion_bias.py (summary.json + files)")
    ap.add_argument("--gold", required=True, help="val_gold_tune500.jsonl")
    ap.add_argument("--bf16-ref", required=True, help="bf16 tune500 submission file (EXP-022, filter on, b 0)")
    ap.add_argument("--out", required=True, help="gate.json")
    ap.add_argument("--min-acc", type=float, default=0.782)
    ap.add_argument("--min-profile", type=float, default=0.4461)
    ap.add_argument("--max-rouge-drop", type=float, default=0.01)
    args = ap.parse_args()

    tdir = Path(args.tuner_dir)
    summary = json.loads((tdir / "summary.json").read_text(encoding="utf-8"))
    b = summary["b_chosen"]
    variant = f"filter_on_b{b:g}"
    slices = summary["slices"]
    if "tune500" not in slices:
        sys.exit("summary.json has no tune500 slice (run the tuner with tune500 in --report-gold)")
    t = slices["tune500"][variant]

    ev = load_eval()
    gold_rows = read_jsonl(args.gold)
    rouge, bleu = reply_scores(ev, tdir / f"tune500_{variant}.jsonl", gold_rows)
    rouge_ref, bleu_ref = reply_scores(ev, args.bf16_ref, gold_rows)

    checks = [
        ("emotion accuracy", t["emotion_acc"], args.min_acc),
        ("profile micro-F1", t["profile_micro_f1_avg"], args.min_profile),
        ("ROUGE-L", rouge, rouge_ref - args.max_rouge_drop),
    ]
    go = all(value >= bar for _, value, bar in checks)

    print(f"\nEXP-023 gate on tune500 (500 points): FP8, b = {b:g} (tuned on tune500), "
          f"top-{summary.get('top_k') or 'all'} alternatives, tag filter on")
    print(f"  {'metric':<18}{'FP8':>9}{'bar':>9}")
    for name, value, bar in checks:
        print(f"  {name:<18}{value:>9.4f}{bar:>9.4f}   {'ok' if value >= bar else 'FAIL'}")
    print(f"  {'BLEU-4 (info)':<18}{bleu:>9.4f}{bleu_ref:>9.4f}   (bf16)")
    print(f"  {'ROUGE-L bf16':<18}{rouge_ref:>9.4f}")
    print("\n  other slices (info only; b was not picked on them):")
    for name, rows in slices.items():
        if name == "tune500":
            continue
        base, tuned = rows["filter_on_b0"], rows[variant]
        print(f"  {name:<8} emotion {base['emotion_acc']:.4f} -> {tuned['emotion_acc']:.4f} "
              f"({tuned['changed_labels']} switched)   profile F1 {tuned['profile_micro_f1_avg']:.4f}")
    print(f"\n{'GO' if go else 'NO-GO'}")

    Path(args.out).write_text(json.dumps({
        "go": go, "b": b, "top_k": summary.get("top_k"),
        "tune500": {"emotion_acc": t["emotion_acc"], "profile_micro_f1_avg": t["profile_micro_f1_avg"],
                    "rouge_l": rouge, "bleu4": bleu, "rouge_l_bf16": rouge_ref, "bleu4_bf16": bleu_ref},
        "bars": {name: bar for name, _, bar in checks},
    }, indent=2), encoding="utf-8")
    sys.exit(0 if go else 3)


if __name__ == "__main__":
    main()
