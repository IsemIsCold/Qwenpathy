#!/usr/bin/env python3
"""
scripts/eval_600.py -- Y5: final evaluation report on the 600 val points, from answers we already saved.

No GPU, no model: it re-parses the raw model answers saved by EXP-022 / EXP-023 with submission #3's parser
(experiments/exp027-early-stop/run_inference_sub3.py, the exact submitted file) and scores them with
scripts/eval.py's own functions.

Systems (all on the same 600 points = data/t10_input.jsonl = n100 + tune500):
  #2        full QLoRA model, bf16, V1.8 parser, profile = the model's own tags   (outputs/exp022/lp_bf16.jsonl)
  #3        same weights in FP8, profile = LLM tags blended with the CPU classifier (EXP-025)
                                                                                  (outputs/exp023/lp_fp8_t10.jsonl)
  #4 cand.  #3's answers, label read leniently (= EXP-027's parser: 2 broken-JSON answers keep their label),
            profile = CPU classifier alone with the exact-match thresholds (0.45 / 0.30 / 0.35)
#4 stops generating after emotion_label; that changes latency, not reply or label, so the saved FP8 answers
give its scores. All numbers are IN-SAMPLE for the profile thresholds of #3/#4 (they were fitted on these 600).

    python scripts/eval_600.py                    # writes docs/eval-600.md (+ outputs/eval_600.json)
    python scripts/eval_600.py --check            # exit 1 if the numbers differ from the ones in LOG.md

Deterministic: a second run leaves `git diff` clean. Needs numpy only for the profile classifier.
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABEL_RE = re.compile(r'"emotion_label"\s*:\s*"([a-z_]+)"')
EXACT_THRESHOLDS = {"personality_traits": 0.45, "interests": 0.30, "style": 0.35}  # EXP-027 `exact` variant
# numbers already in LOG.md / the experiment READMEs that this report must reproduce
EXPECTED = {"#2": {"correct": 474}, "#3": {"correct": 472}, "#4 cand.": {"correct": 473}}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_systems(ri, args):
    inputs = {r["id"]: r for r in read_jsonl(args.input)}
    ids = [r["id"] for r in read_jsonl(args.input)]
    raw = {}
    for name, path in (("bf16", args.bf16), ("fp8", args.fp8)):
        rows = {r["id"]: r for r in read_jsonl(path)}
        missing = [i for i in ids if i not in rows]
        if missing:
            sys.exit(f"{path}: {len(missing)} of the 600 ids missing (first {missing[0]})")
        raw[name] = rows
    clf = None
    if Path(args.profile_clf, "profile_clf.json").exists():
        clf = ri.ProfileClassifier(Path(args.profile_clf))
    systems = {"#2": {}, "#3": {}, "#4 cand.": {}}
    for i in ids:
        p2, _ = ri.parse_prediction(i, raw["bf16"][i]["content"])
        systems["#2"][i] = p2
        p3, _ = ri.parse_prediction(i, raw["fp8"][i]["content"])
        p4 = json.loads(json.dumps(p3))
        if clf is not None:
            p3["user_profile"] = clf.apply(p3["user_profile"], inputs[i])
            probs = clf.probabilities(inputs[i])
            p4["user_profile"] = {f: [t for j, (ff, t) in enumerate(clf.tags) if ff == f and probs[j] >= EXACT_THRESHOLDS[f]]
                                  for f in ri.PROFILE_LABELS}
        m = LABEL_RE.search(raw["fp8"][i]["content"])
        if m and m.group(1) in ri.EMOTIONS:
            p4["emotion_label"] = m.group(1)
        systems["#3"][i] = p3
        systems["#4 cand."][i] = p4
    return ids, systems, clf is not None


def score(ev, ids, sub, gold):
    emo = ev.score_emotion([(sub[i]["emotion_label"], gold[i]["emotion_label"]) for i in ids])
    pairs = {f: [(ev.to_label_set(sub[i]["user_profile"].get(f)), ev.to_label_set((gold[i].get("user_profile") or {}).get(f)))
                 for i in ids] for f in ev.PROFILE_FIELDS}
    prof = ev.score_profile(pairs)
    whole = sum(all(pairs[f][k][0] == pairs[f][k][1] for f in ev.PROFILE_FIELDS) for k in range(len(ids))) / len(ids)
    pred_t = [sub[i]["response_text"] or "" for i in ids]
    gold_t = [gold[i]["response_text"] or "" for i in ids]
    return {
        "correct": sum(sub[i]["emotion_label"] == gold[i]["emotion_label"] for i in ids),
        "emotion": emo, "profile": prof, "whole_profile_exact": round(whole, 4),
        "rouge_l": ev.rouge_l_macro(pred_t, gold_t), "bleu4": ev.bleu4_corpus(pred_t, gold_t),
        "reply_median_chars": ev.length_stats(pred_t).get("median"),
    }


def fmt(x, nd=3):
    if isinstance(x, dict):  # eval.py returns some metrics as {"score": ...} dicts
        for k in ("score", "f1", "value", "mean"):
            if k in x:
                return fmt(x[k], nd)
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def report(ids, res, gold, ev, have_clf, systems):
    n = len(ids)
    E = ev.EMOTIONS
    sup = Counter(gold[i]["emotion_label"] for i in ids)
    L = ["# Evaluation on the 600 val points (Y5)", "",
         "Generated by `python scripts/eval_600.py` from saved model answers (no GPU). Points: `data/t10_input.jsonl` "
         "(= `n100` + `tune500`, 600 points from 488 conversations). Gold: `data/val_gold_n100.jsonl` + "
         "`data/val_gold_tune500.jsonl`. Scorer: `scripts/eval.py`'s own functions.", "",
         "- **#2** = full QLoRA model in bf16 (EXP-019), V1.8 parser, profile = the model's own tags.",
         "- **#3** = the same weights served in FP8 (EXP-020/023), profile = model tags blended with the CPU classifier (EXP-025).",
         "- **#4 cand.** = #3's answers with the lenient label parse and the profile from the CPU classifier alone, exact-match "
         "thresholds 0.45 / 0.30 / 0.35 (EXP-027). Stopping after `emotion_label` changes latency, not the reply or the label.",
         "- Profile thresholds of #3 and #4 were fitted on these same 600 points → their profile numbers are in-sample "
         "(5-fold CV estimate for #4: 0.647 per-field exact, EXP-027).", ""]
    if not have_clf:
        L += ["> **participant/profile_clf/ not found: #3 / #4 profile columns show the model's own tags only.**", ""]
    names = list(res)
    L += ["## Summary", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [
        ("Emotion accuracy", lambda r: f"{r['correct']}/{n} = {r['correct'] / n:.3f}"),
        ("Emotion macro-F1", lambda r: fmt(r["emotion"]["macro_f1"])),
        ("Profile exact match, per-field mean (scored)", lambda r: fmt(r["profile"]["exact_match_avg"])),
        ("Profile exact: personality / interests / style", lambda r: " / ".join(fmt(r["profile"][f]["exact_match"]) for f in ev.PROFILE_FIELDS)),
        ("Whole profile exact (all 3 fields)", lambda r: fmt(r["whole_profile_exact"])),
        ("Profile micro-F1 (mean of fields)", lambda r: fmt(r["profile"]["micro_f1_avg"])),
        ("Reply ROUGE-L (char, macro)", lambda r: fmt(r["rouge_l"])),
        ("Reply BLEU-4 (char, corpus)", lambda r: fmt(r["bleu4"])),
        ("Reply median length (chars)", lambda r: fmt(r["reply_median_chars"], 0)),
    ]
    for label, f in rows:
        L.append(f"| {label} | " + " | ".join(f(res[s]) for s in names) + " |")
    anx = sup["anxiety"] / n
    empty = sum(all(not (gold[i].get("user_profile") or {}).get(f) for f in ev.PROFILE_FIELDS) for i in ids) / n
    L += ["", f"Baselines: always `anxiety` = {sup['anxiety']}/{n} = {anx:.3f}; all-empty profile = whole-profile exact {empty:.3f}.", ""]

    L += ["## Emotion recall per class", "", "| gold label | support | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for c in sorted(E, key=lambda c: -sup[c]):
        if sup[c] == 0:
            continue
        L.append(f"| {c} | {sup[c]} | " + " | ".join(fmt(res[s]["emotion"]["per_class"][c]["recall"], 2) for s in names) + " |")

    main = "#4 cand."
    pairs = [(gold[i]["emotion_label"], systems[main][i]["emotion_label"]) for i in ids]
    conf = Counter(pairs)
    L += ["", f"## Top-10 confusions ({main}, gold → predicted)", "", "| gold → predicted | count | share of that gold class |", "|---|---|---|"]
    for (g, p), k in sorted(((gp, k) for gp, k in conf.items() if gp[0] != gp[1]), key=lambda x: (-x[1], x[0]))[:10]:
        L.append(f"| {g} → {p} | {k} | {k / sup[g]:.0%} |")
    errors = sum(k for (g, p), k in conf.items() if g != p)
    to_anx = sum(k for (g, p), k in conf.items() if g != p and p == "anxiety")
    L += ["", f"{errors} errors; {to_anx} of them ({to_anx / errors:.0%}) are predicted `anxiety`.", ""]

    cols = [c for c in sorted(E, key=lambda c: -sup[c]) if sup[c] or any(p == c for _, p in pairs)]
    short = {c: c[:5] for c in cols}
    L += [f"## Confusion matrix ({main}; rows = gold, columns = predicted)", "",
          "| gold \\ pred | " + " | ".join(short[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for g in cols:
        if sup[g] == 0:
            continue
        L.append(f"| {g} | " + " | ".join(str(conf[(g, p)]) if conf[(g, p)] else "·" for p in cols) + " |")
    L += ["", "Column names are the first 5 letters of the label.", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(REPO / "data/t10_input.jsonl"))
    ap.add_argument("--gold", nargs="+", default=[str(REPO / "data/val_gold_n100.jsonl"), str(REPO / "data/val_gold_tune500.jsonl")])
    ap.add_argument("--bf16", default=str(REPO / "outputs/exp022/lp_bf16.jsonl"))
    ap.add_argument("--fp8", default=str(REPO / "outputs/exp023/lp_fp8_t10.jsonl"))
    ap.add_argument("--profile-clf", default=str(REPO / "participant/profile_clf"))
    ap.add_argument("--out", default=str(REPO / "docs/eval-600.md"))
    ap.add_argument("--json", default=str(REPO / "outputs/eval_600.json"))
    ap.add_argument("--check", action="store_true", help="exit 1 unless the emotion counts match LOG.md")
    args = ap.parse_args()
    # #2/#3 are scored with #3's own parser (frozen copy); #4 with the current one (EXP-027 lenient parse)
    ri = load("ri", REPO / "experiments/exp027-early-stop/run_inference_sub3.py")
    ev = load("ev", REPO / "scripts/eval.py")
    gold = {}
    for g in args.gold:
        for r in read_jsonl(g):
            gold[r["id"]] = r
    ids, systems, have_clf = build_systems(ri, args)
    if any(i not in gold for i in ids):
        sys.exit("gold missing for some of the 600 ids")
    res = {name: score(ev, ids, sub, gold) for name, sub in systems.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report(ids, res, gold, ev, have_clf, systems), encoding="utf-8", newline="\n")
    slim = {s: {"correct": r["correct"], "accuracy": r["emotion"]["accuracy"], "macro_f1": r["emotion"]["macro_f1"],
                "profile_exact_avg": r["profile"]["exact_match_avg"], "whole_profile_exact": r["whole_profile_exact"],
                "profile_micro_f1_avg": r["profile"]["micro_f1_avg"], "rouge_l": r["rouge_l"], "bleu4": r["bleu4"]} for s, r in res.items()}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for s, r in slim.items():
        print(f"{s:<9} emotion {r['correct']}/600 = {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}  "
              f"profile exact {r['profile_exact_avg']:.4f} (whole {r['whole_profile_exact']:.4f})  ROUGE-L {fmt(r['rouge_l'], 4)}")
    print(f"-> {args.out}, {args.json}" + ("" if have_clf else "  (no profile classifier: #3/#4 profile = model tags)"))
    if args.check:
        bad = [f"{s}: {r['correct']} != {EXPECTED[s]['correct']}" for s, r in slim.items() if r["correct"] != EXPECTED[s]["correct"]]
        if have_clf and abs(slim["#4 cand."]["profile_exact_avg"] - 0.6522) > 1e-4:  # EXP-027 `exact`, in-sample
            bad.append(f"#4 profile exact {slim['#4 cand.']['profile_exact_avg']} != 0.6522")
        if bad:
            sys.exit("CHECK FAILED: " + "; ".join(bad))
        print("check: emotion counts match LOG.md (474 / 472 / 473)" + (", #4 profile exact 0.6522 (EXP-027)" if have_clf else ""))


if __name__ == "__main__":
    main()
