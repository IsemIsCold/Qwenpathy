#!/usr/bin/env python3
"""
scripts/eval.py -- B4: score a submission against gold labels.

Usage:
    python scripts/eval.py outputs/submission.jsonl data/val_gold.jsonl \
        [--out outputs/submission.jsonl.metrics.json] [--history docs/score-history.md]

Scope (per docs/eval-design.md, metric tiers):
  DECISION     emotion accuracy (official), macro-F1 (diagnostic-decision),
               profile micro-F1 (averaged across the 3 fields), ROUGE-L
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
profile, baselines, length stats, and the response-quality mirror (char-level
BLEU-4 corpus + ROUGE-L macro + BERTScore-with-local-weights, which skips
gracefully when torch/weights are absent). These approximate the official
20-pt component; composite weights unknown. Latency ingestion and the
statistics layer (bootstrap CI, paired comparison) remain later layers.

Gate (critique fix): eval.py refuses to score a submission that fails the same
record-level checks as validate.py, unless --force is given. Run order:
    python scripts/validate.py <submission> --input <val_input>
    python scripts/eval.py <submission> <gold>

Exit code 0 = scored, 1 = refused / misalignment, 2 = unreadable file.
"""

import argparse
import json
import math
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
    with open(path, "r", encoding="utf-8-sig") as f:
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
    confusion = {g: Counter() for g in EMOTIONS}
    pred_count = Counter()
    correct = n_invalid = 0
    for p, g in pairs:  # single pass; every count below derives from these tallies
        if g in confusion:
            confusion[g][p] += 1
        pred_count[p] += 1
        if p == g:
            correct += 1
        if p not in EMOTION_SET:
            n_invalid += 1

    per_class = {}
    raw_f1 = {}
    for cls in EMOTIONS:
        tp = confusion[cls][cls]
        support = sum(confusion[cls].values())
        fp = pred_count[cls] - tp
        fn = support - tp
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
    confusion_top = {g: dict(sorted(cnt.items(), key=lambda kv: -kv[1])) for g, cnt in confusion.items()}

    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "n_invalid_predictions": n_invalid,
        "per_class": per_class,
        "confusion_top": confusion_top,
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


# ---- response-quality metrics: char-level (Chinese standard), stdlib only.
# These MIRROR the official 20-pt response-quality component (BLEU-4, ROUGE-L)
# as closely as possible without knowing the official composite weights.
# Tokenization = characters (jieba-free: no segmentation noise, no dependency).
# BLEU-4 = corpus-level with standard brevity penalty, no smoothing (a zero
# n-gram precision means 0 -- informative for degenerate output).
# ROUGE-L = per-sample LCS-based F1, macro-averaged over samples.

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu4_corpus(pred_texts, gold_texts):
    pred_tokens = [list(t) for t in pred_texts]
    gold_tokens = [list(t) for t in gold_texts]
    pred_len = sum(len(t) for t in pred_tokens)
    gold_len = sum(len(t) for t in gold_tokens)
    if pred_len == 0 or gold_len == 0:
        return 0.0
    logs = []
    for n in (1, 2, 3, 4):
        overlap = 0
        total = 0
        for p, g in zip(pred_tokens, gold_tokens):
            pn = _ngrams(p, n)
            gn = _ngrams(g, n)
            overlap += sum(min(c, gn.get(ng, 0)) for ng, c in pn.items())
            total += max(len(p) - n + 1, 0)
        if total == 0 or overlap == 0:
            return 0.0
        logs.append(math.log(overlap / total))
    bp = 1.0 if pred_len > gold_len else math.exp(1 - gold_len / pred_len)
    return round(bp * math.exp(sum(logs) / 4), 4)


def _lcs_len(a, b):
    """Exact LCS length, bit-parallel (Allison & Dix 1986; Hyyro 2004).

    One Python big-int holds a whole DP row, so the cost is O(len(a)) big-int
    operations instead of O(len(a) * len(b)) Python steps. Same result as the
    textbook DP (checked against it in tests/run_tests.py, E19).
    """
    if not a or not b:
        return 0
    masks = {}
    for i, ch in enumerate(b):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    full = (1 << len(b)) - 1
    v = full
    for ch in a:
        u = v & masks.get(ch, 0)
        v = ((v + u) | (v - u)) & full
    return len(b) - bin(v).count("1")


ROUGE_L_MAX_CHARS = 2048  # per-side cap; diagnostic metric, keeps LCS quadratic time bounded


def rouge_l_macro(pred_texts, gold_texts):
    """Inputs longer than ROUGE_L_MAX_CHARS chars are truncated per side, so a
    degenerate 20k-char output cannot dominate the run time. Chat replies
    beyond 2048 chars are degenerate anyway; gold max is 1809."""
    f1s = []
    for p, g in zip(pred_texts, gold_texts):
        p = p[:ROUGE_L_MAX_CHARS]
        g = g[:ROUGE_L_MAX_CHARS]
        if not p or not g:
            f1s.append(0.0)
            continue
        L = _lcs_len(p, g)
        r = L / len(g)
        pr = L / len(p)
        f1s.append((2 * r * pr / (r + pr)) if (r + pr) else 0.0)
    return round(sum(f1s) / len(f1s), 4) if f1s else 0.0


def _greedy_cosine_f1(pred_emb, gold_emb, drop_special=True):
    """BERTScore greedy matching (per the Zhang et al. paper) on per-token
    embeddings: precision = mean over pred tokens of max sim to any gold
    token; recall = vice versa. Inputs: np.ndarray (n_tokens, dim).

    drop_special=True gives the first and last token ([CLS]/[SEP] for
    bert-style tokenizers) weight 0 in the averages, exactly like bert-score
    (they still count as match candidates). Counting them inflates scores
    (they match almost perfectly) and favours short replies; a text with no
    real tokens scores 0. Callers with synthetic embeddings pass
    drop_special=False.
    """
    import numpy as np
    p = pred_emb / (np.linalg.norm(pred_emb, axis=1, keepdims=True) + 1e-8)
    g = gold_emb / (np.linalg.norm(gold_emb, axis=1, keepdims=True) + 1e-8)
    sim = p @ g.T
    best_for_pred = sim.max(axis=1)  # each pred token's best match among ALL gold tokens
    best_for_gold = sim.max(axis=0)
    if drop_special:
        if sim.shape[0] <= 2 or sim.shape[1] <= 2:
            # only [CLS][SEP] left: an empty text matches nothing. Without this
            # a blank reply scores near 1.0 (its specials match gold's specials)
            return 0.0, 0.0, 0.0
        # like bert-score: [CLS]/[SEP] stay as match candidates but get weight 0
        # in the average, so drop them only after taking the max
        best_for_pred = best_for_pred[1:-1]
        best_for_gold = best_for_gold[1:-1]
    precision = float(best_for_pred.mean())
    recall = float(best_for_gold.mean())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def bertscore_macro(pred_texts, gold_texts, model_name="bert-base-chinese",
                    layer=8, batch_size=32):
    """Macro-averaged BERTScore (precision/recall/F1) or None if unavailable.

    Mirrors the official bert-score defaults (as verified against the library
    source, Oct 4 team review): per-token embeddings from an INTERMEDIATE
    layer -- bert-score's model2layers uses layer 8 for bert-base-chinese,
    not the last layer (last-layer similarities bunch together, compressing
    real differences); greedy max matching with [CLS]/[SEP] given weight 0
    (drop_special=True in _greedy_cosine_f1); uniform weights (no idf);
    truncation at 512 tokens (gold replies are short so this bites only P99
    outliers). Weights must be present LOCALLY (the eval box is offline and
    HF is blocked; download from ModelScope, pass the folder via --bert-model
    or BERTSCORE_MODEL; override the layer via --bert-layer if your bert-score
    version disagrees). Never raises: any failure returns None so the scorer
    always runs, and the reason is printed to stderr.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None  # torch/transformers not installed: expected on laptops
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:  # weights not cached locally: expected, but say so
        print(f"BERTScore: cannot load {model_name!r} ({type(exc).__name__})", file=sys.stderr)
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        n = len(pred_texts)
        # batch pairs of similar length together: far less padding per batch
        order = sorted(range(n), key=lambda k: max(len(pred_texts[k]), len(gold_texts[k])))
        ps, rs, f1s = [0.0] * n, [0.0] * n, [0.0] * n
        with torch.no_grad():
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                pb = [pred_texts[k] for k in idx]
                gb = [gold_texts[k] for k in idx]
                enc = tokenizer(pb + gb, return_tensors="pt", truncation=True,
                                max_length=512, padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states[layer].float().cpu().numpy()
                lens = enc["attention_mask"].sum(dim=1).tolist()  # right padding (BERT default)
                B = len(idx)
                for j, k in enumerate(idx):
                    ps[k], rs[k], f1s[k] = _greedy_cosine_f1(
                        hs[j, :lens[j]], hs[B + j, :lens[B + j]], drop_special=True)
        if not n:
            return None
        return {"precision": round(sum(ps) / n, 4),
                "recall": round(sum(rs) / n, 4),
                "f1": round(sum(f1s) / n, 4)}
    except Exception as exc:  # a real bug (bad layer, OOM...): never crash, but never hide it
        print(f"BERTScore: failed during scoring ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None


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
    print(f"BLEU-4 (char-level, corpus) {r['bleu4']:.4f}   "
          f"ROUGE-L (char-level, macro) {r['rouge_l']:.4f}")
    bs = r.get("bertscore")
    if bs:
        print(f"BERTScore (macro)  P {bs['precision']:.4f}  R {bs['recall']:.4f}  F1 {bs['f1']:.4f}")
    else:
        print("BERTScore: skipped -- torch/transformers or local weights unavailable "
              "(download bert-base-chinese from ModelScope, pass --bert-model <folder>)")


HISTORY_HEADER = ("| date | submission | emotion acc | macro-F1 | profile micro-F1 (avg) | "
                  "always-anxiety baseline | delta | bleu4 | rouge_l | bertscore |\n"
                  "|---|---|---|---|---|---|---|---|---|---|\n")


def append_history(path, metrics, submission_path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            exists = bool(f.read().strip())
    except OSError:
        exists = False
    if exists:
        with open(path, "r", encoding="utf-8") as f:
            old_header = f.readline()
        if "bleu4" not in old_header:
            # header format change: archive the old table instead of
            # misaligning it (registry rule: never lose an experiment)
            version = 1
            while os.path.exists(f"{path}.v{version}.md"):
                version += 1
            os.replace(path, f"{path}.v{version}.md")
            exists = False
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    e = metrics["emotion"]["accuracy"]
    m = metrics["emotion"]["macro_f1"]
    pf = metrics["profile"]["micro_f1_avg"]
    b = metrics["baselines"]["always_anxiety_accuracy"]
    bleu = metrics["response"]["bleu4"]
    rouge = metrics["response"]["rouge_l"]
    bs = metrics["response"].get("bertscore")
    bs_f1 = f"{bs['f1']:.4f}" if bs else "-"
    row = (f"| {date} | {submission_path} | {e:.4f} | {m:.4f} | {pf:.4f} | {b:.4f} "
           f"| {e - b:+.4f} | {bleu:.4f} | {rouge:.4f} | {bs_f1} |\n")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(HISTORY_HEADER)
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
    ap.add_argument("--bert-layer", type=int, default=None,
                    help="Hidden layer for BERTScore embeddings (default: env BERTSCORE_LAYER or 8; "
                         "bert-score's model2layers uses 8 for bert-base-chinese -- NOT the last layer)")
    ap.add_argument("--bert-model", default=None,
                    help="Local path/name for BERTScore weights (default: env BERTSCORE_MODEL or "
                         "bert-base-chinese; must be cached locally -- box is offline)")
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

    if not gold_order:
        print(f"ERROR: {args.gold} has no records", file=sys.stderr)
        sys.exit(2)
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
            "bleu4": bleu4_corpus(pred_texts, gold_texts),
            "rouge_l": rouge_l_macro(pred_texts, gold_texts),
            "bertscore": bertscore_macro(
                pred_texts, gold_texts,
                model_name=args.bert_model or os.environ.get("BERTSCORE_MODEL", "bert-base-chinese"),
                layer=args.bert_layer if args.bert_layer is not None
                      else int(os.environ.get("BERTSCORE_LAYER", "8"))),
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
