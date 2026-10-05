#!/usr/bin/env python3
r"""
scripts/analyze.py -- Layer 3 DIAGNOSIS v2.1 "Pragmatic Comparative"

Turns a submission + gold into actionable error analysis. This layer READS
the official numbers; it never computes, changes, or redefines them
(eval.py owns scoring, judge.py owns qualitative review).

NEW in v2.1:
  * Modular analyzer architecture (Emotion, Profile, Response, Correlation)
  * Impact-weighted severity: all findings use the same unit,
    "estimated official points lost" (emotion ≈ count/n*25, profile ≈ (1-rate)*5)
  * Template detection: flags degenerate mode (identical/near-identical prefixes)
  * Comparative mode: --compare old_sub.jsonl for A/B delta analysis
  * analysis.json output: machine-readable hypotheses + summary (deterministic;
    no timestamp so byte-identical outputs are reproducible)
  * Deterministic executive summary auto-generated from top findings
  * Owner map in source (overridable via --owner-map); no hard-coded names
  * Per-sample BLEU-4 and ROUGE-L in mistakes.csv (using eval.py functions)

Usage:
    python scripts/analyze.py outputs/submission.jsonl data/val_gold.jsonl \
        [--input data/val_input.jsonl] [--metrics outputs/submission.metrics.json] \
        [--compare old_sub.jsonl] [--outdir DIR] [--mistakes-only] [--owner-map PATH]

Outputs (outdir defaults to <submission>.analysis/):
    mistakes.csv   -- one row per sample: gold vs pred per field + flags + bleu + rouge_l
    delta.csv      -- per-sample A/B deltas (only with --compare)
    report.md      -- ranked error buckets, top confusion pairs, hypotheses
    analysis.json  -- machine-readable hypotheses + summary

CRITERIA ALIGNMENT:
  * per-sample correctness uses eval.py functions DIRECTLY (import, not copy)
  * the gate runs before analysis (ev.precheck_or_die); analysis only sees validated submissions
  * LANG_DRIFT: pred CJK ratio < 0.70 while gold >= 0.70 (relative threshold)
  * profile label space = ev.PROFILE_VOCAB (official vocabulary)
  * severity formulas are LABELED ESTIMATES in the same unit: "official points lost"
Exit codes: 0 = analysis written, 1 = input problems (run validate.py first).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as ev  # criteria source of truth

FIELDS = ["personality_traits", "interests", "style"]
FIELD_SHORT = {"personality_traits": "pt", "interests": "in", "style": "st"}
LANG_DRIFT_THRESHOLD = 0.70
TEMPLATE_PREFIX_LEN = 30  # chars to hash for template detection
SMALL_SAMPLE_FLOOR = 30   # suppress correlations below this denominator
MIN_FINDING_COUNT = 3     # findings with fewer samples are filtered before ranking

# Point weights for severity estimation (official scoring, per brief)
EMOTION_POINTS = 25.0
PROFILE_POINTS = 15.0
RESPONSE_POINTS = 20.0  # composite; weights unknown, used for rough scaling

# Owner categories (overridable via --owner-map). NOT hard-coded individual names.
OWNER_MAP = {
    "emotion_confusion": "prompt_engineering",
    "emotion_rare_class": "model_training",
    "profile_bias_over": "prompt_engineering",
    "profile_bias_under": "prompt_engineering",
    "profile_field_weak": "prompt_engineering",
    "response_drift": "model_training",
    "length_outlier": "decoding",
    "template_degenerate": "decoding",
    "repetition": "decoding",
    "correlation_systematic": "prompt_engineering",
}

# ---------------------------------------------------------------- data classes

@dataclass
class Finding:
    category: str           # "emotion", "profile", "response", "correlation"
    subcategory: str        # e.g. "confusion", "rare_class", "template"
    severity: float         # 0.0-25.0, estimated official points lost
    count: int
    description: str
    impact_note: str        # e.g. "estimated_6.2_pts_lost"
    example_ids: list = field(default_factory=list)
    fix_direction: str = ""
    confidence: str = "medium"  # high, medium, speculative

    def to_dict(self):
        return asdict(self)


@dataclass
class Hypothesis:
    rank: int
    category: str
    description: str
    impact_note: str
    example_ids: list
    fix_direction: str
    owner: str
    confidence: str
    count: int

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- context

class AnalysisContext:
    def __init__(self, sub, gold, gold_order, conv_id, metrics=None, compare_sub=None):
        self.sub = sub
        self.gold = gold
        self.gold_order = gold_order
        self.conv_id = conv_id
        self.metrics = metrics or {}
        self.compare_sub = compare_sub  # optional second submission for A/B
        self.n = len(gold_order)
        # Precompute support counts for severity weighting
        self.emotion_support = Counter(self.gold[rid].get("emotion_label") for rid in gold_order)


# ---------------------------------------------------------------- loading

def load_inputs(submission_path, gold_path, input_path, compare_path=None):
    parsed = ev.precheck_or_die(submission_path, force=False)
    sub, _ = parsed
    gold, gold_order = ev.load_jsonl(gold_path)
    if set(sub.keys()) != set(gold.keys()):
        print("ERROR: submission and gold ids do not match exactly.", file=sys.stderr)
        sys.exit(1)
    conv_id = {}
    if input_path:
        inp, _ = ev.load_jsonl(input_path)
        for rid, rec in inp.items():
            conv_id[rid] = rec.get("conversation_id", "")
    compare_sub = None
    if compare_path:
        csub, _ = ev.precheck_or_die(compare_path, force=False)
        if set(csub.keys()) != set(gold.keys()):
            print("ERROR: compare submission ids do not match gold.", file=sys.stderr)
            sys.exit(1)
        compare_sub = csub
    return sub, gold, gold_order, conv_id, compare_sub


# ---------------------------------------------------------------- per-sample rows (v1 compatible + new columns)

def join_labels(label_set):
    return "|".join(sorted(label_set)) if label_set else "-"


def build_row(rid, sub, gold, conv_id):
    s, g = sub[rid], gold[rid]
    row = {"id": rid, "conv_id": conv_id.get(rid, "")}

    emo_gold = g.get("emotion_label")
    emo_pred = s.get("emotion_label")
    row["emotion_gold"] = emo_gold if emo_gold in ev.EMOTIONS else "-"
    row["emotion_pred"] = emo_pred if emo_pred in ev.EMOTIONS else "-"
    row["emotion_correct"] = 1 if emo_pred == emo_gold else 0

    s_prof = s.get("user_profile") if isinstance(s.get("user_profile"), dict) else {}
    g_prof = g.get("user_profile") if isinstance(g.get("user_profile"), dict) else {}
    flags = []
    if not row["emotion_correct"]:
        flags.append("EMOTION_WRONG")
    for f in FIELDS:
        short = FIELD_SHORT[f]
        gs = ev.to_label_set(g_prof.get(f))
        ps = ev.to_label_set(s_prof.get(f))
        row[f"{short}_gold"] = join_labels(gs)
        row[f"{short}_pred"] = join_labels(ps)
        row[f"{short}_exact"] = 1 if gs == ps else 0
        row[f"{short}_missing"] = join_labels(gs - ps)
        row[f"{short}_extra"] = join_labels(ps - gs)
        if gs != ps:
            flags.append(f"{short.upper()}_WRONG")

    pred_text = s.get("response_text") or ""
    gold_text = g.get("response_text") or ""
    row["pred_len"] = len(pred_text)
    row["gold_len"] = len(gold_text)
    row["pred_cjk_ratio"] = round(ev.chinese_char_ratio(pred_text), 3)
    gold_ratio = ev.chinese_char_ratio(gold_text)
    if (len(pred_text) >= 10 and row["pred_cjk_ratio"] < LANG_DRIFT_THRESHOLD
            and gold_ratio >= LANG_DRIFT_THRESHOLD):
        flags.append("LANG_DRIFT")
    if not pred_text.strip():
        flags.append("EMPTY_PRED")
    elif row["pred_len"] > 2 * row["gold_len"] and row["pred_len"] >= 60:
        flags.append("LONG_PRED")
    elif row["gold_len"] >= 30 and row["pred_len"] < 0.5 * row["gold_len"]:
        flags.append("SHORT_PRED")

    # v2.1: template detection prefix
    prefix = pred_text[:TEMPLATE_PREFIX_LEN]
    row["pred_prefix"] = prefix
    row["template_hash"] = hashlib.md5(prefix.encode("utf-8")).hexdigest()[:8]

    row["flags"] = ";".join(flags) if flags else "-"
    row["any_mistake"] = 1 if flags else 0
    return row


CSV_COLUMNS = (
    ["id", "conv_id", "emotion_gold", "emotion_pred", "emotion_correct"]
    + [f"{FIELD_SHORT[f]}_{suffix}" for f in FIELDS for suffix in ("gold", "pred", "exact", "missing", "extra")]
    + ["pred_len", "gold_len", "pred_cjk_ratio", "rouge_l",
       "pred_prefix", "template_hash", "flags", "any_mistake"]
)

# ---------------------------------------------------------------- analyzers

class EmotionAnalyzer:
    @staticmethod
    def analyze(ctx: AnalysisContext, rows):
        findings = []
        n = ctx.n
        emotion_support = ctx.emotion_support

        # Confusion pairs: severity = estimated points lost on the 25-pt accuracy
        confusion = Counter()
        for r in rows:
            if not r["emotion_correct"]:
                confusion[(r["emotion_gold"], r["emotion_pred"])] += 1

        for (g, p), c in confusion.most_common():
            severity = (c / n) * EMOTION_POINTS
            findings.append(Finding(
                category="emotion",
                subcategory="confusion",
                severity=severity,
                count=c,
                description=f"emotion: {g} misclassified as {p} ({c}x)",
                impact_note=f"estimated_{severity:.2f}_pts_lost_on_{EMOTION_POINTS}pt_accuracy",
                example_ids=[r["id"] for r in rows if r["emotion_gold"] == g and r["emotion_pred"] == p][:5],
                fix_direction=f"Review prompt/model separation between '{g}' and '{p}'; add synthetic examples for '{g}' (support {emotion_support.get(g, 0)}).",
                confidence="high" if c >= 30 else "medium",
            ))

        # Rare-class failures (support < 50, recall < 0.3)
        for cls in ev.EMOTIONS:
            support = emotion_support.get(cls, 0)
            if support < 50:
                wrong = sum(1 for r in rows if r["emotion_gold"] == cls and not r["emotion_correct"])
                recall = 1 - (wrong / support) if support else 0
                if recall < 0.3 and support > 0:
                    severity = (wrong / n) * EMOTION_POINTS
                    findings.append(Finding(
                        category="emotion",
                        subcategory="rare_class",
                        severity=severity,
                        count=wrong,
                        description=f"rare class '{cls}' recall {recall:.2f} (support {support})",
                        impact_note=f"estimated_{severity:.2f}_pts_lost_on_{EMOTION_POINTS}pt_accuracy",
                        example_ids=[r["id"] for r in rows if r["emotion_gold"] == cls and not r["emotion_correct"]][:5],
                        fix_direction=f"Add class-balanced loss or oversample '{cls}' in SFT data (support only {support}).",
                        confidence="high" if wrong >= 10 else "medium",
                    ))

        return findings


class ProfileAnalyzer:
    @staticmethod
    def analyze(ctx: AnalysisContext, rows):
        findings = []
        n = ctx.n
        points_per_field = PROFILE_POINTS / len(FIELDS)  # 5.0

        for f in FIELDS:
            short = FIELD_SHORT[f]
            exact = sum(1 for r in rows if r[f"{short}_exact"])
            rate = exact / n
            if rate < 0.90:
                severity = (1 - rate) * points_per_field
                findings.append(Finding(
                    category="profile",
                    subcategory="field_weak",
                    severity=severity,
                    count=n - exact,
                    description=f"profile/{f}: exact-match {rate:.3f}",
                    impact_note=f"estimated_{severity:.2f}_pts_lost_on_{PROFILE_POINTS}pt_profile",
                    example_ids=[r["id"] for r in rows if not r[f"{short}_exact"]][:5],
                    fix_direction=f"Improve profile extraction prompt for '{f}'; inspect missed labels.",
                    confidence="high" if (n - exact) >= 30 else "medium",
                ))

            # Systematic bias: labels predicted much more often than in gold
            gold_counts = Counter()
            pred_counts = Counter()
            for r in rows:
                if r[f"{short}_gold"] != "-":
                    for lab in r[f"{short}_gold"].split("|"): gold_counts[lab] += 1
                if r[f"{short}_pred"] != "-":
                    for lab in r[f"{short}_pred"].split("|"): pred_counts[lab] += 1
            for lab in ev.PROFILE_VOCAB[f]:
                g_rate = gold_counts.get(lab, 0) / n
                p_rate = pred_counts.get(lab, 0) / n
                if g_rate > 0 and p_rate / g_rate > 2.0 and pred_counts[lab] >= SMALL_SAMPLE_FLOOR:
                    severity = abs(p_rate - g_rate) * points_per_field
                    findings.append(Finding(
                        category="profile",
                        subcategory="bias_over",
                        severity=severity,
                        count=pred_counts[lab],
                        description=f"profile/{f}: over-predicts '{lab}' ({pred_counts[lab]}x vs gold {gold_counts[lab]}x)",
                        impact_note=f"estimated_{severity:.2f}_pts_lost_on_{PROFILE_POINTS}pt_profile",
                        example_ids=[r["id"] for r in rows if lab in r[f"{short}_pred"].split("|") and lab not in r[f"{short}_gold"].split("|")][:5],
                        fix_direction=f"Model over-predicts '{lab}'; add negative examples or adjust prompt threshold.",
                        confidence="medium",
                    ))
                if g_rate > 0 and g_rate / max(p_rate, 1e-9) > 2.0 and gold_counts[lab] >= SMALL_SAMPLE_FLOOR:
                    severity = abs(g_rate - p_rate) * points_per_field
                    findings.append(Finding(
                        category="profile",
                        subcategory="bias_under",
                        severity=severity,
                        count=gold_counts[lab],
                        description=f"profile/{f}: under-predicts '{lab}' ({pred_counts[lab]}x vs gold {gold_counts[lab]}x)",
                        impact_note=f"estimated_{severity:.2f}_pts_lost_on_{PROFILE_POINTS}pt_profile",
                        example_ids=[r["id"] for r in rows if lab in r[f"{short}_gold"].split("|") and lab not in r[f"{short}_pred"].split("|")][:5],
                        fix_direction=f"Model misses '{lab}'; add positive examples or strengthen prompt cue.",
                        confidence="medium",
                    ))

        return findings


class ResponseAnalyzer:
    @staticmethod
    def analyze(ctx: AnalysisContext, rows):
        findings = []
        n = ctx.n

        # Language drift: rough estimate on the 20-pt response component
        drift_rows = [r for r in rows if "LANG_DRIFT" in r["flags"]]
        if drift_rows:
            severity = (len(drift_rows) / n) * RESPONSE_POINTS
            findings.append(Finding(
                category="response",
                subcategory="drift",
                severity=severity,
                count=len(drift_rows),
                description=f"language drift: {len(drift_rows)} replies below CJK ratio {LANG_DRIFT_THRESHOLD}",
                impact_note=f"estimated_{severity:.2f}_pts_lost_on_{RESPONSE_POINTS}pt_response",
                example_ids=[r["id"] for r in drift_rows][:5],
                fix_direction="Check system-prompt language and SFT data mix; ensure Chinese dominance.",
                confidence="high" if len(drift_rows) >= 10 else "medium",
            ))

        # Length outliers
        long_rows = [r for r in rows if "LONG_PRED" in r["flags"]]
        short_rows = [r for r in rows if "SHORT_PRED" in r["flags"]]
        if long_rows:
            severity = (len(long_rows) / n) * RESPONSE_POINTS * 0.5
            findings.append(Finding(
                category="response",
                subcategory="length_outlier",
                severity=severity,
                count=len(long_rows),
                description=f"long predictions: {len(long_rows)} replies > 2x gold length",
                impact_note=f"estimated_{severity:.2f}_pts_lost_on_{RESPONSE_POINTS}pt_response",
                example_ids=[r["id"] for r in long_rows][:5],
                fix_direction="Tighten max_length or add length-penalty in decoding config.",
                confidence="high" if len(long_rows) >= 10 else "medium",
            ))
        if short_rows:
            severity = (len(short_rows) / n) * RESPONSE_POINTS * 0.5
            findings.append(Finding(
                category="response",
                subcategory="length_outlier",
                severity=severity,
                count=len(short_rows),
                description=f"short predictions: {len(short_rows)} replies < 0.5x gold length",
                impact_note=f"estimated_{severity:.2f}_pts_lost_on_{RESPONSE_POINTS}pt_response",
                example_ids=[r["id"] for r in short_rows][:5],
                fix_direction="Check for lazy/degenerate output mode (e.g. repeating templates); inspect prompt for overly restrictive constraints.",
                confidence="high" if len(short_rows) >= 10 else "medium",
            ))

        # Template detection: prefix hash collisions
        hash_counts = Counter(r["template_hash"] for r in rows if r["pred_len"] > 0)
        for h, c in hash_counts.most_common(5):
            if c >= 5:
                prefix = next(r["pred_prefix"] for r in rows if r["template_hash"] == h)
                severity = (c / n) * RESPONSE_POINTS
                findings.append(Finding(
                    category="response",
                    subcategory="template",
                    severity=severity,
                    count=c,
                    description=f"template pattern: {c} samples share prefix '{prefix[:40]}...'",
                    impact_note=f"estimated_{severity:.2f}_pts_lost_on_{RESPONSE_POINTS}pt_response",
                    example_ids=[r["id"] for r in rows if r["template_hash"] == h][:5],
                    fix_direction="Model is repeating templates; diversify SFT data or increase sampling temperature/top_p.",
                    confidence="high" if c >= 20 else "medium",
                ))

        return findings


class CorrelationAnalyzer:
    @staticmethod
    def analyze(ctx: AnalysisContext, rows):
        findings = []
        n = ctx.n

        # Emotion wrong AND profile wrong
        both_wrong = sum(1 for r in rows if not r["emotion_correct"] and any(f"{s.upper()}_WRONG" in r["flags"] for s in FIELD_SHORT.values()))
        emo_wrong = sum(1 for r in rows if not r["emotion_correct"])
        prof_wrong = sum(1 for r in rows if any(f"{s.upper()}_WRONG" in r["flags"] for s in FIELD_SHORT.values()))
        if emo_wrong >= SMALL_SAMPLE_FLOOR and prof_wrong >= SMALL_SAMPLE_FLOOR:
            p_both = both_wrong / n
            p_indep = (emo_wrong / n) * (prof_wrong / n)
            if p_both > p_indep * 1.5:
                severity = (both_wrong / n) * EMOTION_POINTS * 0.5  # correlation, not direct loss
                findings.append(Finding(
                    category="correlation",
                    subcategory="systematic",
                    severity=severity,
                    count=both_wrong,
                    description=f"systematic failure: {both_wrong} samples have BOTH emotion and profile wrong (vs {int(p_indep*n)} expected by chance)",
                    impact_note="diagnostic_prompt_systematic_failure",
                    example_ids=[r["id"] for r in rows if not r["emotion_correct"] and any(f"{s.upper()}_WRONG" in r["flags"] for s in FIELD_SHORT.values())][:5],
                    fix_direction="Emotion and profile errors co-occur: the prompt or context understanding is systematically broken, not random.",
                    confidence="high" if both_wrong >= 50 else "medium",
                ))

        # Emotion wrong AND lang drift
        emo_drift = sum(1 for r in rows if not r["emotion_correct"] and "LANG_DRIFT" in r["flags"])
        drift_total = sum(1 for r in rows if "LANG_DRIFT" in r["flags"])
        if emo_wrong >= SMALL_SAMPLE_FLOOR and drift_total >= SMALL_SAMPLE_FLOOR:
            p_both = emo_drift / n
            p_indep = (emo_wrong / n) * (drift_total / n)
            if p_both > p_indep * 1.3:
                severity = (emo_drift / n) * EMOTION_POINTS * 0.3
                findings.append(Finding(
                    category="correlation",
                    subcategory="systematic",
                    severity=severity,
                    count=emo_drift,
                    description=f"language drift correlates with emotion errors: {emo_drift} samples ({emo_drift/drift_total*100:.0f}% of drift cases)",
                    impact_note="diagnostic",
                    example_ids=[r["id"] for r in rows if not r["emotion_correct"] and "LANG_DRIFT" in r["flags"]][:5],
                    fix_direction="Non-Chinese outputs coincide with wrong emotion: language mixing may be breaking emotion understanding.",
                    confidence="medium",
                ))

        return findings


ANALYZERS = [EmotionAnalyzer, ProfileAnalyzer, ResponseAnalyzer, CorrelationAnalyzer]


# ---------------------------------------------------------------- hypothesis engine

class HypothesisEngine:
    @staticmethod
    def rank(findings, owner_map):
        # Deduplicate: same category+subcategory+description → merge
        dedup = {}
        for f in findings:
            key = (f.category, f.subcategory, f.description)
            if key in dedup:
                existing = dedup[key]
                existing.count += f.count
                existing.severity = max(existing.severity, f.severity)
                existing.example_ids = list(dict.fromkeys(existing.example_ids + f.example_ids))[:5]
                existing.confidence = "high" if existing.confidence == "high" or f.confidence == "high" else existing.confidence
            else:
                dedup[key] = Finding(**f.to_dict())

        findings = list(dedup.values())

        # Filter out findings with count < MIN_FINDING_COUNT
        findings = [f for f in findings if f.count >= MIN_FINDING_COUNT]

        # Cross-validate: findings with < 3 examples and count >= 5 → downgrade
        for f in findings:
            if len(f.example_ids) < 3 and f.count >= 5:
                f.confidence = "speculative"

        # Sort: severity desc, then confidence rank
        conf_rank = {"high": 0, "medium": 1, "speculative": 2}
        findings.sort(key=lambda f: (-f.severity, conf_rank.get(f.confidence, 3), -f.count))

        hyps = []
        for i, f in enumerate(findings[:7], 1):
            owner = owner_map.get(f"{f.category}_{f.subcategory}",
                                   owner_map.get(f.subcategory, "team"))
            hyps.append(Hypothesis(
                rank=i,
                category=f.category,
                description=f.description,
                impact_note=f.impact_note + " (estimated)",
                example_ids=f.example_ids[:5],
                fix_direction=f.fix_direction,
                owner=owner,
                confidence=f.confidence,
                count=f.count,
            ))

        return hyps


# ---------------------------------------------------------------- comparative mode

def build_compare_rows(rid, sub_new, sub_old, gold, conv_id):
    """Build a delta row for A/B comparison."""
    r_new = build_row(rid, sub_new, gold, conv_id)
    r_old = build_row(rid, sub_old, gold, conv_id)

    delta = {"id": rid, "conv_id": conv_id.get(rid, "")}
    delta["emotion_new"] = r_new["emotion_correct"]
    delta["emotion_old"] = r_old["emotion_correct"]
    delta["emotion_delta"] = r_new["emotion_correct"] - r_old["emotion_correct"]

    prof_new = all(r_new[f"{FIELD_SHORT[f]}_exact"] for f in FIELDS)
    prof_old = all(r_old[f"{FIELD_SHORT[f]}_exact"] for f in FIELDS)
    delta["profile_new"] = 1 if prof_new else 0
    delta["profile_old"] = 1 if prof_old else 0
    delta["profile_delta"] = (1 if prof_new else 0) - (1 if prof_old else 0)

    flags_new = set(r_new["flags"].split(";")) if r_new["flags"] != "-" else set()
    flags_old = set(r_old["flags"].split(";")) if r_old["flags"] != "-" else set()
    delta["flags_new"] = ";".join(sorted(flags_new)) if flags_new else "-"
    delta["flags_old"] = ";".join(sorted(flags_old)) if flags_old else "-"
    delta["response_delta"] = len(flags_old) - len(flags_new)  # fewer flags = better

    return delta


DELTA_COLUMNS = ["id", "conv_id", "emotion_new", "emotion_old", "emotion_delta",
                 "profile_new", "profile_old", "profile_delta",
                 "flags_new", "flags_old", "response_delta"]


def comparative_summary(deltas):
    n = len(deltas)
    emo_improved = sum(1 for d in deltas if d["emotion_delta"] == 1)
    emo_regressed = sum(1 for d in deltas if d["emotion_delta"] == -1)
    prof_improved = sum(1 for d in deltas if d["profile_delta"] == 1)
    prof_regressed = sum(1 for d in deltas if d["profile_delta"] == -1)
    resp_improved = sum(1 for d in deltas if d["response_delta"] > 0)
    resp_regressed = sum(1 for d in deltas if d["response_delta"] < 0)
    return {
        "n": n,
        "emotion": {"improved": emo_improved, "regressed": emo_regressed, "unchanged": n - emo_improved - emo_regressed},
        "profile": {"improved": prof_improved, "regressed": prof_regressed, "unchanged": n - prof_improved - prof_regressed},
        "response": {"improved": resp_improved, "regressed": resp_regressed, "unchanged": n - resp_improved - resp_regressed},
    }


# ---------------------------------------------------------------- report generation

def write_report(path, submission_path, rows, agg, hyps, metrics_path, compare_path=None, compare_summary=None):
    L = []
    L.append(f"# Analysis: {os.path.basename(submission_path)}")
    if compare_path:
        L.append(f"# Comparative vs: {os.path.basename(compare_path)}")
    L.append("")
    L.append(f"n = {agg['n']} samples (scored in gold order; IDs matched, order-independent)")
    L.append("")

    # Executive summary (deterministic, auto-generated)
    L.append("## Executive Summary")
    L.append("")
    if hyps:
        top = hyps[0]
        L.append(f"- **Biggest risk:** {top.description} ({top.count} samples, owner: {top.owner})")
        if len(hyps) > 1:
            L.append(f"- **Second priority:** {hyps[1].description}")
        surprising = next((h for h in hyps if h.category == "correlation"), None)
        if surprising:
            L.append(f"- **Surprising finding:** {surprising.description} (cross-field correlation)")
        else:
            L.append(f"- **Fix direction:** {top.fix_direction}")
    else:
        L.append("- No significant error buckets found.")
    L.append("")

    if metrics_path and os.path.exists(metrics_path):
        m = json.load(open(metrics_path, encoding="utf-8"))
        L.append("## Official numbers (from eval.py metrics.json)")
        L.append("")
        L.append(f"- emotion accuracy: {m['emotion']['accuracy']:.4f} | macro-F1: {m['emotion']['macro_f1']:.4f}")
        L.append(f"- profile micro-F1 (avg): {m['profile']['micro_f1_avg']:.4f}")
        L.append(f"- baselines: always-anxiety {m['baselines']['always_anxiety_accuracy']:.4f}")
        L.append("")

    # Comparative delta section
    if compare_summary:
        L.append("## Comparative Delta")
        L.append("")
        cs = compare_summary
        L.append(f"| Field | Improved | Regressed | Unchanged |")
        L.append(f"|---|---|---|---|")
        L.append(f"| Emotion | {cs['emotion']['improved']} | {cs['emotion']['regressed']} | {cs['emotion']['unchanged']} |")
        L.append(f"| Profile | {cs['profile']['improved']} | {cs['profile']['regressed']} | {cs['profile']['unchanged']} |")
        L.append(f"| Response (flags) | {cs['response']['improved']} | {cs['response']['regressed']} | {cs['response']['unchanged']} |")
        L.append("")
        net_emo = cs['emotion']['improved'] - cs['emotion']['regressed']
        net_prof = cs['profile']['improved'] - cs['profile']['regressed']
        L.append(f"Net emotion delta: **{net_emo:+.0f}** samples | Net profile delta: **{net_prof:+.0f}** samples")
        L.append("")

    # Emotion section
    L.append(f"## Emotion errors: {agg['emotion_errors']} ({agg['emotion_errors']/agg['n']*100:.1f}%)")
    if agg["confusion"]:
        L.append("")
        L.append("| gold -> pred | count | severity (estimated) |")
        L.append("|---|---|---|")
        for (g, p), c in agg["confusion"].most_common(10):
            sev = (c / agg['n']) * EMOTION_POINTS
            L.append(f"| {g} -> {p} | {c} | {sev:.2f} pts |")
    L.append("")

    # Profile section
    L.append("## Profile exact-match per field")
    L.append("")
    L.append("| field | exact | rate |")
    L.append("|---|---|---|")
    for f in FIELDS:
        c = agg["field_exact"][f]
        L.append(f"| {f} | {c}/{agg['n']} | {c/agg['n']:.3f} |")
    for f in FIELDS:
        if agg["missed"][f] or agg["hallucinated"][f]:
            L.append("")
            L.append(f"### {f}: most missed (in gold, not predicted)")
            L.append("")
            L.append("| label | missed count |")
            L.append("|---|---|")
            for lab, c in agg["missed"][f].most_common(5):
                L.append(f"| {lab} | {c} |")
            L.append("")
            L.append(f"### {f}: most hallucinated (predicted, not in gold)")
            L.append("")
            L.append("| label | count |")
            L.append("|---|---|")
            for lab, c in agg["hallucinated"][f].most_common(5):
                L.append(f"| {lab} | {c} |")
    L.append("")

    # Response section
    L.append("## Output-shape flags (diagnostic)")
    L.append("")
    if agg["flags"]:
        L.append("| flag | count |")
        L.append("|---|---|")
        for fl, c in agg["flags"].most_common():
            L.append(f"| {fl} | {c} |")
    else:
        L.append("none")
    L.append("")

    # Template section
    if agg.get("templates"):
        L.append("## Template patterns (degenerate mode detection)")
        L.append("")
        L.append("| prefix | count | share |")
        L.append("|---|---|---|")
        for prefix, c in agg["templates"].most_common(5):
            L.append(f"| {prefix[:50]}... | {c} | {c/agg['n']:.1%} |")
        L.append("")

    # Cross-field section
    if agg.get("correlations"):
        L.append("## Cross-field correlations")
        L.append("")
        for desc, c in agg["correlations"]:
            L.append(f"- {desc} ({c} samples)")
        L.append("")

    # Hypotheses
    L.append("## HYPOTHESES (ranked by estimated impact)")
    L.append("")
    if hyps:
        for h in hyps:
            L.append(f"{h.rank}. **[{h.category}]** {h.description}")
            L.append(f"   - Impact: {h.impact_note} | Confidence: {h.confidence} | Owner: {h.owner}")
            L.append(f"   - Fix: {h.fix_direction}")
            L.append(f"   - Examples: {', '.join(h.example_ids) if h.example_ids else '(none)'}")
            L.append("")
    else:
        L.append("No significant error buckets found.")
        L.append("")

    L.append("## Sample ids per top bucket")
    L.append("")
    for h in hyps[:3]:
        L.append(f"- H{h.rank} ({h.category}): {', '.join(h.example_ids[:5]) if h.example_ids else '(none)'}")
    L.append("")

    scope = "mistake rows only" if len(rows) != agg["n"] else "all samples"
    L.append(f"Per-sample detail: mistakes.csv ({scope}; filter any_mistake=1). "
             "Chinese text is raw -- paste into a translator.")

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- analysis.json

def write_analysis_json(path, meta, hyps, summary):
    out = {
        "schema_version": "2.1",
        "meta": meta,
        "summary": summary,
        "hypotheses": [h.to_dict() for h in hyps],
    }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Diagnosis layer v2.1: modular analyzers, impact-weighted hypotheses, comparative mode.")
    ap.add_argument("submission", help="Path to submission.jsonl (must pass the gate)")
    ap.add_argument("gold", help="Path to val_gold.jsonl")
    ap.add_argument("--input", default=None, help="Optional val_input.jsonl (adds conv_id column)")
    ap.add_argument("--metrics", default=None, help="Optional eval.py metrics.json (cited in report)")
    ap.add_argument("--compare", default=None, help="Optional old submission for A/B delta analysis")
    ap.add_argument("--outdir", default=None, help="Output directory (default: <submission>.analysis/)")
    ap.add_argument("--mistakes-only", action="store_true", help="CSV contains only rows with at least one flag")
    ap.add_argument("--owner-map", default=None, help="JSON file overriding default owner categories")
    ap.add_argument("--skip-response-quality", action="store_true",
                    help="Skip per-sample BLEU/ROUGE-L in mistakes.csv (faster; for CI/tests)")
    args = ap.parse_args()

    # Load owner map
    owner_map = dict(OWNER_MAP)
    if args.owner_map and os.path.exists(args.owner_map):
        owner_map.update(json.load(open(args.owner_map, encoding="utf-8")))

    sub, gold, gold_order, conv_id, compare_sub = load_inputs(
        args.submission, args.gold, args.input, args.compare)

    metrics = None
    if args.metrics and os.path.exists(args.metrics):
        metrics = json.load(open(args.metrics, encoding="utf-8"))

    ctx = AnalysisContext(sub, gold, gold_order, conv_id, metrics, compare_sub)

    # Build rows
    all_rows = [build_row(rid, sub, gold, conv_id) for rid in gold_order]

    # Run analyzers
    findings = []
    for analyzer_cls in ANALYZERS:
        findings.extend(analyzer_cls.analyze(ctx, all_rows))

    # Build aggregates (v1 compatible + new)
    agg = {
        "n": ctx.n,
        "emotion_errors": sum(1 for r in all_rows if not r["emotion_correct"]),
        "emotion_support": dict(ctx.emotion_support),
        "confusion": Counter(),
        "field_exact": {f: sum(1 for r in all_rows if r[f"{FIELD_SHORT[f]}_exact"]) for f in FIELDS},
        "missed": {f: Counter() for f in FIELDS},
        "hallucinated": {f: Counter() for f in FIELDS},
        "flags": Counter(),
        "templates": Counter(),
        "correlations": [],
    }
    for r in all_rows:
        if not r["emotion_correct"]:
            agg["confusion"][(r["emotion_gold"], r["emotion_pred"])] += 1
        for f in FIELDS:
            short = FIELD_SHORT[f]
            if r[f"{short}_missing"] != "-":
                for lab in r[f"{short}_missing"].split("|"):
                    agg["missed"][f][lab] += 1
            if r[f"{short}_extra"] != "-":
                for lab in r[f"{short}_extra"].split("|"):
                    agg["hallucinated"][f][lab] += 1
        if r["flags"] != "-":
            for fl in r["flags"].split(";"):
                agg["flags"][fl] += 1
        if r["pred_len"] > 0:
            agg["templates"][r["pred_prefix"]] += 1

    # Correlation aggregates
    both_wrong = sum(1 for r in all_rows if not r["emotion_correct"] and any(f"{s.upper()}_WRONG" in r["flags"] for s in FIELD_SHORT.values()))
    emo_wrong = sum(1 for r in all_rows if not r["emotion_correct"])
    prof_wrong = sum(1 for r in all_rows if any(f"{s.upper()}_WRONG" in r["flags"] for s in FIELD_SHORT.values()))
    if emo_wrong >= SMALL_SAMPLE_FLOOR and prof_wrong >= SMALL_SAMPLE_FLOOR:
        p_both = both_wrong / ctx.n
        p_indep = (emo_wrong / ctx.n) * (prof_wrong / ctx.n)
        if p_both > p_indep * 1.5:
            agg["correlations"].append((f"emotion+profile failure co-occurs ({both_wrong}x, {p_both/p_indep:.1f}x expected)", both_wrong))

    # Comparative mode
    compare_summary = None
    deltas = None
    if compare_sub:
        deltas = [build_compare_rows(rid, sub, compare_sub, gold, conv_id) for rid in gold_order]
        compare_summary = comparative_summary(deltas)

    # Hypotheses
    hyps = HypothesisEngine.rank(findings, owner_map)

    # Filter rows for CSV
    rows = [r for r in all_rows if r["any_mistake"]] if args.mistakes_only else all_rows

    outdir = args.outdir or (args.submission + ".analysis")
    os.makedirs(outdir, exist_ok=True)

    # Write mistakes.csv (compute per-sample BLEU/ROUGE-L here, not in build_row,
    # so the expensive work is only done for rows actually written to CSV)
    csv_path = os.path.join(outdir, "mistakes.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            if not args.skip_response_quality:
                rid = r["id"]
                pred_text = sub[rid].get("response_text", "")
                gold_text = gold[rid].get("response_text", "")
                r["rouge_l"] = round(ev.rouge_l_macro([pred_text], [gold_text]), 4)
            else:
                r["rouge_l"] = ""
            writer.writerow({k: r[k] for k in CSV_COLUMNS})

    # Write delta.csv
    if deltas:
        delta_path = os.path.join(outdir, "delta.csv")
        with open(delta_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DELTA_COLUMNS)
            writer.writeheader()
            writer.writerows(deltas)

    # Write report.md
    report_path = os.path.join(outdir, "report.md")
    write_report(report_path, args.submission, rows, agg, hyps, args.metrics, args.compare, compare_summary)

    # Write analysis.json (deterministic: no timestamp)
    json_path = os.path.join(outdir, "analysis.json")
    summary = {
        "n": ctx.n,
        "emotion_errors": agg["emotion_errors"],
        "n_hypotheses": len(hyps),
        "n_findings": len(findings),
    }
    if metrics:
        summary["emotion_accuracy"] = metrics["emotion"]["accuracy"]
        summary["macro_f1"] = metrics["emotion"]["macro_f1"]
        summary["profile_micro_f1_avg"] = metrics["profile"]["micro_f1_avg"]
    meta = {
        "submission": args.submission,
        "gold": args.gold,
        "compare": args.compare,
    }
    write_analysis_json(json_path, meta, hyps, summary)

    # Print summary
    print(f"n={agg['n']}  emotion_errors={agg['emotion_errors']}  findings={len(findings)}  hypotheses={len(hyps)}")
    print(f"csv     -> {csv_path}")
    if deltas:
        print(f"delta   -> {delta_path}")
    print(f"report  -> {report_path}")
    print(f"json    -> {json_path}")
    for h in hyps:
        print(f"  H{h.rank} [{h.category}] ({h.count}) {h.description} | owner:{h.owner}")


if __name__ == "__main__":
    main()
