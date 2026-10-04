#!/usr/bin/env python3
"""
scripts/eval.py -- B4: score a submission against gold labels.

Usage:
    python scripts/eval.py outputs/submission.jsonl data/val_gold.jsonl \
        [--out outputs/submission.jsonl.metrics.json] [--history docs/score-history.md]

Scope (per docs/eval-design.md, metric tiers):
  DECISION     emotion accuracy (official), macro-F1 (diagnostic-decision),
               profile micro-F1 (averaged across the 3 fields)
  DIAGNOSTIC   per-class P/R/F1, confusion matrix, per-label profile P/R/F1,
               exact-match, reply length stats, Chinese-char ratio
  Baselines    always-anxiety accuracy, empty-profile exact-match

Emotion metric = ACCURACY (official, 25 pts -- source: official competition
rules page, screenshot Oct 4 via WeChat translate: 70 auto = 25 emotion Acc +
20 BLEU-4/ROUGE-L/BERTScore + 15 profile avg + 10 latency; + 30 expert review).
Macro-F1 is tracked because the official metric rewards majority-class
collapse; per-class tables show where.
Profile = multi-label, scored per sub-field: micro-F1 primary, macro-F1 also
reported. Empty gold profile vs empty prediction counts as an exact match.
All metrics are pure-stdlib (no sklearn) so the script runs identically on
Windows / Linux / the offline eval box.

This is the CORE scorer (build-order step 2 of docs/eval-design.md): emotion,
profile, baselines, length stats. Response-quality metrics (BLEU-4 / ROUGE-L /
BERTScore), latency ingestion, and the statistics layer (bootstrap CI, paired
comparison) are later layers and intentionally not implemented here.

Gate (critique fix): eval.py refuses to score a submission that fails the same
record-level checks as validate.py, unless --force is given. Run order:
    python scripts/validate.py <submission> --input <val_input>
    python scripts/eval.py <submission> <gold>

Exit code 0 = scored, 1 = refused / misalignment, 2 = unreadable file.
"""

import argparse
import json
import os
import statistics
import sys
import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate as gate  # reuse the gate's record checks; no duplicated logic

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
]
EMOTION_SET = frozenset(EMOTIONS)

PROFILE_VOCAB = {
    "personality_traits": [
        "sensitive", "conservative", "high_conscientiousness", "assertive",
        "agreeable", "introverted", "emotionally_stable", "casual", "open",
        "extroverted",
    ],
    "interests": [
        "career_development", "pets", "study_exam", "sports_fitness", "games",
        "reading_writing", "art_design", "social", "travel_outdoor", "music",
        "film_animation", "programming_technology",
    ],
    "style": [
        "rational", "direct", "detailed", "brief", "indirect",
        "high_emotional_expression", "colloquial", "formal", "humorous",
        "emoji_user", "low_emotional_expression",
    ],
}
PROFILE_FIELDS = list(PROFILE_VOCAB.keys())


# ---------------------------------------------------------------- loading

def load_jsonl(path):
    records = {}
    order = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"ERROR: {path} line {line_no}: invalid JSON: {exc}", file=sys.stderr)
                sys.exit(1)
            if not isinstance(obj, dict) or "id" not in obj:
                print(f"ERROR: {path} line {line_no}: not an object with an id", file=sys.stderr)
                sys.exit(1)
            rid = obj["id"]
            if not isinstance(rid, str):
                print(f"ERROR: {path} line {line_no}: 'id' must be a string", file=sys.stderr)
                sys.exit(1)
            if rid in records:
                print(f"ERROR: {path} line {line_no}: duplicate id {rid!r}", file=sys.stderr)
                sys.exit(1)
            records[rid] = obj
            order.append(rid)
    return records, order


# ---------------------------------------------------------------- emotion

def score_emotion(pairs):
    """pairs = [(pred_label, gold_label)] ; invalid pred labels count as wrong."""
    n = len(pairs)
    correct = sum(1 for p, g in pairs if p == g)
    per_class = {}
    raw_f1 = {}
    for cls in EMOTIONS:
        tp = sum(1 for p, g in pairs if p == cls and g == cls)
        fp = sum(1 for p, g in pairs if p == cls and g != cls)
        fn = sum(1 for p, g in pairs if p != cls and g == cls)
        support = sum(1 for _, g in pairs if g == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        raw_f1[cls] = f1
        per_class[cls] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
        }
    # macro/weighted computed from RAW f1, not the rounded display values
    macro_f1 = sum(raw_f1[c] for c in EMOTIONS) / len(EMOTIONS)
    weighted_f1 = (sum(raw_f1[c] * per_class[c]["support"] for c in EMOTIONS) / n) if n else 0.0

    confusion = {g: Counter() for g in EMOTIONS}
    for p, g in pairs:
        confusion[g][p] += 1
    confusion = {g: dict(sorted(cnt.items(), key=lambda kv: -kv[1])) for g, cnt in confusion.items()}

    n_invalid = sum(1 for p, _ in pairs if p not in EMOTION_SET)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "n_invalid_predictions": n_invalid,
        "per_class": per_class,
        "confusion_top": confusion,
    }


# ---------------------------------------------------------------- profile

def score_profile_field(pairs, vocab):
    """pairs = [(pred_set, gold_set)] for one sub-field."""
    tp = fp = fn = 0
    exact = 0
    label_stats = {lab: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lab in vocab}
    n_empty_gold = 0
    for pred, gold in pairs:
        if not gold:
            n_empty_gold += 1
        if pred == gold:
            exact += 1
        for lab in vocab:
            p, g = lab in pred, lab in gold
            if g:
                label_stats[lab]["support"] += 1
            if p and g:
                tp += 1; label_stats[lab]["tp"] += 1
            elif p and not g:
                fp += 1; label_stats[lab]["fp"] += 1
            elif not p and g:
                fn += 1; label_stats[lab]["fn"] += 1
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    per_label = {}
    f1s = []
    for lab in vocab:
        s = label_stats[lab]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        # macro averages only labels with gold support: a label that never
        # appears in gold (e.g. train-only style labels) must not drag the
        # macro down -- a perfect submission must score macro-F1 1.0
        if s["support"] > 0:
            f1s.append(f1)
        per_label[lab] = {
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "support": s["support"],
        }
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "n": len(pairs),
        "n_empty_gold": n_empty_gold,
        "empty_gold_rate": round(n_empty_gold / len(pairs), 4) if pairs else 0.0,
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match": round(exact / len(pairs), 4) if pairs else 0.0,
        "per_label": per_label,
    }


def to_label_set(value):
    if not isinstance(value, list):
        return frozenset()
    return frozenset(v for v in value if isinstance(v, str))


def score_profile(pairs_by_field):
    out = {}
    for field in PROFILE_FIELDS:
        out[field] = score_profile_field(pairs_by_field[field], PROFILE_VOCAB[field])
    out["micro_f1_avg"] = round(sum(out[f]["micro_f1"] for f in PROFILE_FIELDS) / len(PROFILE_FIELDS), 4)
    out["macro_f1_avg"] = round(sum(out[f]["macro_f1"] for f in PROFILE_FIELDS) / len(PROFILE_FIELDS), 4)
    out["exact_match_avg"] = round(sum(out[f]["exact_match"] for f in PROFILE_FIELDS) / len(PROFILE_FIELDS), 4)
    return out


# ---------------------------------------------------------------- response

def chinese_char_ratio(text):
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cjk / len(text)


def mean_chinese_ratio(texts):
    """Mean of per-sample CJK ratios -- unweighted by response length."""
    ratios = [chinese_char_ratio(t) for t in texts if t]
    return round(statistics.mean(ratios), 4) if ratios else 0.0


def length_stats(texts):
    lens = [len(t) for t in texts]
    if not lens:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.mean(lens), 1),
        "median": round(statistics.median(lens), 1),
        "min": min(lens), "max": max(lens),
    }


# ---------------------------------------------------------------- report

def print_report(metrics):
    e = metrics["emotion"]
    print("=" * 64)
    print("EMOTION")
    print("=" * 64)
    print(f"accuracy (OFFICIAL)   {e['accuracy']:.4f}")
    print(f"macro-F1              {e['macro_f1']:.4f}   (diagnostic: majority-class guard)")
    print(f"weighted-F1           {e['weighted_f1']:.4f}")
    if e["n_invalid_predictions"]:
        print(f"!! {e['n_invalid_predictions']} predictions had invalid emotion labels (counted as wrong)")
    print(f"{'class':<16}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}")
    for cls in EMOTIONS:
        s = e["per_class"][cls]
        flag = "  <-- rare" if 0 < s["support"] < 30 else ""
        print(f"{cls:<16}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}{s['support']:>9}{flag}")
    print()
    print("top confusion pairs (gold -> pred):")
    pairs = []
    for g, preds in e["confusion_top"].items():
        for p, c in preds.items():
            if p != g and c > 0:
                pairs.append((c, g, p))
    for c, g, p in sorted(pairs, reverse=True)[:5]:
        print(f"  {g:<14} -> {p:<14} x{c}")

    print()
    print("=" * 64)
    print("USER PROFILE (multi-label, per sub-field)")
    print("=" * 64)
    pr = metrics["profile"]
    for field in PROFILE_FIELDS:
        s = pr[field]
        print(f"{field:<22} micro-F1 {s['micro_f1']:.4f}  macro-F1 {s['macro_f1']:.4f}  "
              f"exact-match {s['exact_match']:.4f}  (empty gold {s['n_empty_gold']}/{s['n']})")
    print(f"{'AVERAGE':<22} micro-F1 {pr['micro_f1_avg']:.4f}  macro-F1 {pr['macro_f1_avg']:.4f}  "
          f"exact-match {pr['exact_match_avg']:.4f}")

    print()
    print("=" * 64)
    print("BASELINES (what a useless model would get)")
    print("=" * 64)
    b = metrics["baselines"]
    print(f"always-anxiety accuracy   {b['always_anxiety_accuracy']:.4f}")
    print(f"empty-profile exact-match {b['empty_profile_exact_match_avg']:.4f} "
          f"(F1 = 0.0 by definition; exact-match is the informative number)")
    print(f"delta vs always-anxiety   {metrics['emotion']['accuracy'] - b['always_anxiety_accuracy']:+.4f}")

    print()
    print("=" * 64)
    print("REPLY LENGTH (chars)")
    print("=" * 64)
    r = metrics["response"]
    print(f"pred : mean {r['length_pred']['mean']:>7}  median {r['length_pred']['median']:>7}  max {r['length_pred']['max']}")
    print(f"gold : mean {r['length_gold']['mean']:>7}  median {r['length_gold']['median']:>7}  max {r['length_gold']['max']}")
    print(f"pred Chinese-char ratio {r['chinese_char_ratio_pred']:.3f}  "
          f"(< 0.70 suggests non-Chinese output drift; gold mean 0.855, 5th pct 0.745)")


def append_history(path, metrics, submission_path):
    header = ("| date | submission | emotion acc | macro-F1 | profile micro-F1 (avg) | always-anxiety baseline | delta |\n"
              "|---|---|---|---|---|---|---|\n")
    try:
        with open(path, "r", encoding="utf-8") as f:
            exists = bool(f.read().strip())
    except OSError:
        exists = False
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    e = metrics["emotion"]["accuracy"]
    m = metrics["emotion"]["macro_f1"]
    pf = metrics["profile"]["micro_f1_avg"]
    b = metrics["baselines"]["always_anxiety_accuracy"]
    row = f"| {date} | {submission_path} | {e:.4f} | {m:.4f} | {pf:.4f} | {b:.4f} | {e - b:+.4f} |\n"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(header)
        f.write(row)


# ---------------------------------------------------------------- main

def precheck_or_die(submission_path, force):
    """Refuse to score submissions that fail the gate's record-level checks.

    Returns the already-parsed (records_dict, order) so main() does not parse
    the submission file a second time. Returns None when force=True (caller
    must then load the file itself, at its own risk).
    """
    if force:
        return None
    counters = gate.Counters()
    records = gate.load_records(submission_path, counters, "submission")
    for line_no, rec in records:
        gate.check_record(rec, line_no, counters)
    # only string ids enter set operations; non-string ids are already flagged
    # as structural violations by check_record
    str_ids = [rec.get("id") for _, rec in records if isinstance(rec.get("id"), str)]
    n_dup = len(str_ids) - len(set(str_ids))
    hard = (counters.json_errors or counters.structural_errors
            or counters.bad_labels or counters.mem_errors or n_dup)
    if hard:
        print("ERROR: submission failed pre-score validation; refusing to score:", file=sys.stderr)
        print("       " + counters.summary(), file=sys.stderr)
        print("       Run first: python scripts/validate.py <submission> --input <val_input>", file=sys.stderr)
        print("       Override (not recommended): eval.py --force", file=sys.stderr)
        sys.exit(1)
    # success path: every id is a string and there are no duplicates, so the
    # dict construction below cannot collide or crash
    return ({rec["id"]: rec for _, rec in records},
            [rec["id"] for _, rec in records])


def main():
    ap = argparse.ArgumentParser(description="Score a submission against gold labels.")
    ap.add_argument("submission", help="Path to submission.jsonl")
    ap.add_argument("gold", help="Path to val_gold.jsonl")
    ap.add_argument("--out", default=None, help="Write metrics JSON here (default: <submission>.metrics.json)")
    ap.add_argument("--history", default="docs/score-history.md", help="Append row to this file (use --history '' to skip)")
    ap.add_argument("--force", action="store_true",
                    help="Score even if the submission fails record-level validation (not recommended)")
    args = ap.parse_args()

    parsed = precheck_or_die(args.submission, args.force)
    if parsed is not None:
        sub, _unused = parsed  # already parsed by the gate -- no second read
    else:
        sub, _unused = load_jsonl(args.submission)  # --force path
    gold, gold_order = load_jsonl(args.gold)

    if set(sub.keys()) != set(gold.keys()):
        print("ERROR: submission and gold ids do not match exactly.", file=sys.stderr)
        print("       Run: python scripts/validate.py <submission> --input <val_input>", file=sys.stderr)
        sys.exit(1)

    ids = gold_order  # score in gold order (independent of submission file order)

    emotion_pairs = []
    profile_pairs = {f: [] for f in PROFILE_FIELDS}
    pred_texts, gold_texts = [], []
    for rid in ids:
        s, g = sub[rid], gold[rid]
        emotion_pairs.append((s.get("emotion_label"), g.get("emotion_label")))
        for field in PROFILE_FIELDS:
            sp = s.get("user_profile", {}) if isinstance(s.get("user_profile"), dict) else {}
            gp = g.get("user_profile", {}) if isinstance(g.get("user_profile"), dict) else {}
            profile_pairs[field].append((to_label_set(sp.get(field)), to_label_set(gp.get(field))))
        pred_texts.append(s.get("response_text") or "")
        gold_texts.append(g.get("response_text") or "")

    metrics = {
        "submission": args.submission,
        "gold": args.gold,
        "n_scored": len(ids),
        "emotion": score_emotion(emotion_pairs),
        "profile": score_profile(profile_pairs),
        "response": {
            "length_pred": length_stats(pred_texts),
            "length_gold": length_stats(gold_texts),
            "chinese_char_ratio_pred": mean_chinese_ratio(pred_texts),
        },
    }

    anxiety_acc = sum(1 for _, g in emotion_pairs if g == "anxiety") / len(emotion_pairs)

    def gold_field_set(rid, field):
        prof = gold[rid].get("user_profile")
        return to_label_set(prof.get(field)) if isinstance(prof, dict) else frozenset()

    metrics["baselines"] = {
        "always_anxiety_accuracy": round(anxiety_acc, 4),
        "empty_profile_exact_match_avg": round(
            sum(
                score_profile_field(
                    [(frozenset(), gold_field_set(rid, f)) for rid in ids],
                    PROFILE_VOCAB[f],
                )["exact_match"]
                for f in PROFILE_FIELDS
            ) / len(PROFILE_FIELDS), 4),
    }

    print_report(metrics)

    out_path = args.out or (args.submission + ".metrics.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print()
    print(f"metrics JSON -> {out_path}")

    if args.history:
        append_history(args.history, metrics, args.submission)
        print(f"history row  -> {args.history}")

    print()


if __name__ == "__main__":
    main()
