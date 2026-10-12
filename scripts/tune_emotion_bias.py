#!/usr/bin/env python3
"""
scripts/tune_emotion_bias.py -- EXP-022: offline anxiety-prior correction + tag-filter test.

Reads the file written by scripts/emotion_logprobs.py (raw answer + top-20 logprobs at
the first token of the emotion label) and the gold files. No GPU, no server.

1. Parses every raw answer twice with participant/run_inference.py's own parser:
   tag filter ON (= submission #2) and OFF (OVERPREDICTED_TAGS emptied).
2. Anxiety correction: the model says "anxiety" too often (42% predicted vs 33% gold).
   When the parsed label is anxiety, switch to the best other label if
       logprob(other) > logprob(anxiety) - b
   b = 0 changes nothing. b is picked on the tuning set only, then checked on the
   report slices (which were not used to pick it).
3. Prints emotion accuracy and profile micro-F1 (scripts/eval.py's own scorers) and
   writes one submission file per variant and slice, for a full scripts/eval.py run.

Usage (laptop, repo root, venv):
    python scripts/tune_emotion_bias.py --lp outputs/exp022/lp_bf16.jsonl \
        --tune-gold data/val_gold_tune500.jsonl \
        --report-gold data/val_gold_n50.jsonl data/val_gold_n50b.jsonl data/val_gold_n100.jsonl \
        --out-dir outputs/exp022/bf16
    # FP8 (no tuning set): reuse the b chosen on bf16
    python scripts/tune_emotion_bias.py --lp outputs/exp022/lp_fp8.jsonl --bias 1.25 \
        --report-gold data/val_gold_n50.jsonl data/val_gold_n50b.jsonl data/val_gold_n100.jsonl \
        --out-dir outputs/exp022/fp8
"""

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# First token of each label right after ' "' (Qwen3 tokenizer, checked Oct 12).
# Exact match only: "an" is anxiety, "anger" is its own token.
FIRST_TOKEN = {
    "joy": "joy", "gr": "gratitude", "rel": "relaxed", "care": "care", "pr": "pride",
    "neutral": "neutral", "sur": "surprise", "mixed": "mixed", "sad": "sadness",
    "lon": "loneliness", "an": "anxiety", "anger": "anger", "f": "fear", "dis": "disgust",
    "sh": "shame", "help": "helplessness",
}
GRID = [round(0.05 * i, 2) for i in range(0, 101)]  # b = 0.00 .. 5.00


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_scores(label_top):
    """{label: logprob} from the top-k list; several tokens for one label are summed in prob space."""
    probs = {}
    for alt in label_top or []:
        label = FIRST_TOKEN.get(alt["token"].strip())
        if label is not None:
            probs[label] = probs.get(label, 0.0) + math.exp(alt["logprob"])
    return {k: math.log(v) for k, v in probs.items()}


def corrected(label, scores, b):
    """Apply the anxiety correction to one parsed label."""
    if b <= 0 or label != "anxiety" or "anxiety" not in scores:
        return label
    others = {k: v for k, v in scores.items() if k != "anxiety"}
    if not others:
        return label
    best = max(others, key=others.get)
    return best if others[best] > scores["anxiety"] - b else label


def accuracy(rows, gold, b):
    ids = [r["id"] for r in rows if r["id"] in gold]
    hits = sum(corrected(r["label"], r["scores"], b) == gold[r["id"]]["emotion_label"] for r in rows if r["id"] in gold)
    return hits / len(ids) if ids else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp", required=True, help="output of scripts/emotion_logprobs.py")
    ap.add_argument("--tune-gold", default=None, help="gold of the tuning set (picks b)")
    ap.add_argument("--report-gold", nargs="+", required=True, help="gold slices to report on")
    ap.add_argument("--bias", type=float, default=None, help="use this b instead of tuning")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--participant", default=str(ROOT / "participant"))
    args = ap.parse_args()
    if args.bias is None and not args.tune_gold:
        sys.exit("give --tune-gold (to pick b) or --bias")

    ri = load_module("run_inference", Path(args.participant) / "run_inference.py")
    sys.path.insert(0, str(ROOT / "scripts"))  # eval.py imports validate.py
    ev = load_module("eval_mod", ROOT / "scripts" / "eval.py")

    lp = read_jsonl(args.lp)
    filter_on = ri.OVERPREDICTED_TAGS
    rows = []
    for rec in lp:
        ri.OVERPREDICTED_TAGS = filter_on
        pred_on, failed = ri.parse_prediction(rec["id"], rec["content"])
        ri.OVERPREDICTED_TAGS = {}
        pred_off, _ = ri.parse_prediction(rec["id"], rec["content"])
        rows.append({"id": rec["id"], "on": pred_on, "off": pred_off, "failed": failed,
                     "label": pred_on["emotion_label"], "scores": label_scores(rec.get("label_top"))})
    ri.OVERPREDICTED_TAGS = filter_on

    # sanity: with greedy decoding the top token should be the parsed label
    n_lp = sum(bool(r["scores"]) for r in rows)
    agree = sum(bool(r["scores"]) and max(r["scores"], key=r["scores"].get) == r["label"] for r in rows)
    print(f"records: {len(rows)}, with label logprobs: {n_lp}, parse failures: {sum(r['failed'] for r in rows)}")
    print(f"top logprob label == parsed label: {agree}/{n_lp}  (should be ~all)")

    # ---- pick b
    if args.bias is not None:
        b_star = args.bias
        print(f"\nb fixed by --bias: {b_star}")
    else:
        tune = {g["id"]: g for g in read_jsonl(args.tune_gold)}
        tune_rows = [r for r in rows if r["id"] in tune]
        if len(tune_rows) != len(tune):
            sys.exit(f"tuning gold has {len(tune)} ids, only {len(tune_rows)} found in --lp")
        curve = [(b, accuracy(tune_rows, tune, b)) for b in GRID]
        best = max(acc for _, acc in curve)
        b_star = min(b for b, acc in curve if acc == best)  # smallest b reaching the best
        print(f"\ntuning set ({len(tune)} points): accuracy vs b")
        for b, acc in curve:
            if b in (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5) or b == b_star:
                print(f"  b={b:4.2f}  {acc:.4f}" + ("   <- chosen" if b == b_star else ""))
        anx_tune = sum(r["label"] == "anxiety" for r in tune_rows) / len(tune_rows)
        anx_new = sum(corrected(r["label"], r["scores"], b_star) == "anxiety" for r in tune_rows) / len(tune_rows)
        anx_gold = sum(g["emotion_label"] == "anxiety" for g in tune.values()) / len(tune)
        print(f"  anxiety predicted: {anx_tune:.1%} -> {anx_new:.1%} (gold {anx_gold:.1%})")

    # ---- report slices + submission files
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [("filter_on_b0", "on", 0.0), ("filter_off_b0", "off", 0.0),
                (f"filter_on_b{b_star:g}", "on", b_star), (f"filter_off_b{b_star:g}", "off", b_star)]
    summary = {"lp": args.lp, "b_chosen": b_star, "slices": {}}
    print(f"\n{'slice':<10} {'variant':<18} {'emotion acc':>11} {'profile F1':>10} {'changed':>8}")
    for gpath in args.report_gold:
        gold_rows = read_jsonl(gpath)
        gold = {g["id"]: g for g in gold_rows}
        by_id = {r["id"]: r for r in rows}
        missing = [g for g in gold if g not in by_id]
        if missing:
            sys.exit(f"{gpath}: {len(missing)} ids not in --lp (first {missing[0]})")
        name = Path(gpath).stem.replace("val_gold_", "")
        summary["slices"][name] = {}
        for vname, side, b in variants:
            emo_pairs, prof_pairs, changed = [], {f: [] for f in ev.PROFILE_FIELDS}, 0
            with open(out_dir / f"{name}_{vname}.jsonl", "w", encoding="utf-8") as f:
                for g in gold_rows:
                    r = by_id[g["id"]]
                    pred = dict(r[side])
                    pred["emotion_label"] = corrected(r["label"], r["scores"], b)
                    changed += pred["emotion_label"] != r["label"]
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                    emo_pairs.append((pred["emotion_label"], g["emotion_label"]))
                    for field in ev.PROFILE_FIELDS:
                        prof_pairs[field].append((ev.to_label_set(pred["user_profile"].get(field)),
                                                  ev.to_label_set((g.get("user_profile") or {}).get(field))))
            emo = ev.score_emotion(emo_pairs)
            prof = ev.score_profile(prof_pairs)
            summary["slices"][name][vname] = {"emotion_acc": emo["accuracy"], "macro_f1": emo["macro_f1"],
                                              "profile_micro_f1_avg": prof["micro_f1_avg"],
                                              "profile_exact_avg": prof["exact_match_avg"],
                                              "interests_micro_f1": prof["interests"]["micro_f1"],
                                              "changed_labels": changed}
            print(f"{name:<10} {vname:<18} {emo['accuracy']:>11.4f} {prof['micro_f1_avg']:>10.4f} {changed:>8}")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsubmission files + summary.json -> {out_dir}")
    print("full report for one variant: python scripts/eval.py <file> <gold> --history ''")


if __name__ == "__main__":
    main()
