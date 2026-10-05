#!/usr/bin/env python3
r"""
scripts/analyze.py -- Layer 3 DIAGNOSIS v4.0

Reads a submission + gold (+ input) and explains WHERE the errors are.
It never computes official numbers itself: every score comes from eval.py.

v4 (Oct 6) over v3:
  * one signature per error type (taxonomy, gold, pred); context is tallied, not hashed
  * regimes: a condition is attached to an error group only if it is over-represented
    there (>= 60% of the group, >= 1.5x its overall rate, <= 80% of all samples)
  * counterfactuals are MEASURED: the fix is applied and re-scored with eval.py
    (emotion = oracle upper bound; profile false positive = the exact post-filter;
    profile points shown for both F1 and exact match; no points for replies)
  * new tables: prediction distribution / collapse check, emotion accuracy by turn and
    history length, bidirectional confusions
  * genome: stable signatures, deterministic run order
  * fix advice limited to what the plan allows (no synthetic data, no temperature)

Usage:
    python scripts/analyze.py <submission.jsonl> <gold.jsonl> [--input <input.jsonl>]
        [--compare <old_submission.jsonl>] [--outdir DIR] [--force]
        [--run-label NAME] [--registry DIR] [--experiment-notes TEXT]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as ev  # criteria source of truth

FIELDS = ["personality_traits", "interests", "style"]
FIELD_SHORT = {"personality_traits": "pt", "interests": "in", "style": "st"}
LANG_DRIFT_THRESHOLD = 0.70
TEMPLATE_PREFIX_LEN = 30
SMALL_SAMPLE_FLOOR = 30
MIN_FINDING_COUNT = 3

# Point weights for severity estimation (official scoring)
EMOTION_POINTS = 25.0
PROFILE_POINTS = 15.0
RESPONSE_POINTS = 20.0

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

# ---------------------------------------------------------------------------
# v3.0 TAXONOMY
# ---------------------------------------------------------------------------

TAXONOMY = {
    "emotion": {
        "confusion": "Gold emotion misclassified as different emotion",
        "overprediction": "Emotion predicted more often than in gold",
        "underprediction": "Emotion predicted less often than in gold",
        "trigger_bias": "Specific keyword triggers specific emotion",
    },
    "profile": {
        "false_positive": "Label predicted but not in gold",
        "false_negative": "Label in gold but not predicted",
        "field_weak": "Exact-match rate below threshold for a field",
        "bias_over": "Label over-predicted relative to gold rate",
        "bias_under": "Label under-predicted relative to gold rate",
    },
    "response": {
        "drift": "CJK ratio below threshold while gold is Chinese",
        "length_outlier": "Prediction length far from gold length",
        "template": "Identical/near-identical prefix across samples",
    },
    "correlation": {
        "systematic": "Cross-field failure co-occurs more than chance",
    },
}

# Keyword lists for condition extraction (Chinese + English)
KEYWORDS = {
    "exam": ["\u8003\u8bd5", "\u6210\u7ee9", "\u5206\u6570", "\u4f5c\u4e1a", "\u5b66\u6821", "\u5927\u5b66", "\u8003\u7814", "\u9ad8\u8003", "\u6d4b\u9a8c",
             "exam", "test", "grade", "score", "homework", "school", "university"],
    "relationship": ["\u5206\u624b", "\u604b\u7231", "\u670b\u53cb", "\u5173\u7cfb", "\u611f\u60c5", "\u5435\u67b6", "\u590d\u5408", "\u5931\u604b",
                     "breakup", "relationship", "boyfriend", "girlfriend", "love", "friend"],
    "work": ["\u5de5\u4f5c", "\u52a0\u73ed", "\u8001\u677f", "\u540c\u4e8b", "\u8f9e\u804c", "\u9762\u8bd5", "\u9879\u76ee",
             "work", "job", "boss", "colleague", "interview", "project", "overtime"],
    "health": ["\u751f\u75c5", "\u533b\u9662", "\u533b\u751f", "\u5403\u836f", "\u53d1\u70e7", "\u611f\u5192", "\u75bc\u75db", "\u5931\u7720",
               "health", "sick", "doctor", "hospital", "medicine", "fever", "pain", "insomnia"],
    "family": ["\u7236\u6bcd", "\u5988\u5988", "\u7238\u7238", "\u5bb6\u5ead", "\u79bb\u5a5a", "\u4eb2\u4eba",
               "family", "parents", "mom", "dad", "mother", "father", "divorce"],
}

# ---------------------------------------------------------------- data classes

@dataclass
class Finding:
    category: str
    subcategory: str
    severity: float
    count: int
    description: str
    impact_note: str
    example_ids: list = field(default_factory=list)
    fix_direction: str = ""
    confidence: str = "medium"

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


@dataclass
class Signature:
    sig_id: str
    taxonomy: str       # e.g. "emotion.confusion"
    category: str       # e.g. "emotion"
    subcategory: str    # e.g. "confusion"
    gold: str
    pred: str
    conditions: list
    sample_ids: list
    count: int

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- context

class AnalysisContext:
    def __init__(self, sub, gold, gold_order, conv_id, input_data=None,
                 metrics=None, compare_sub=None):
        self.sub = sub
        self.gold = gold
        self.gold_order = gold_order
        self.conv_id = conv_id
        self.input_data = input_data or {}
        self.metrics = metrics or {}
        self.compare_sub = compare_sub
        self.n = len(gold_order)
        self.emotion_support = Counter(self.gold[rid].get("emotion_label") for rid in gold_order)
        # v3: conversation map
        self.conversation_map = self._build_conversation_map()

    def _build_conversation_map(self):
        """Map conversation_id -> ordered list of sample ids."""
        cmap = defaultdict(list)
        for rid in self.gold_order:
            cid = self.conv_id.get(rid, "")
            if cid:
                cmap[cid].append(rid)
        return dict(cmap)


# ---------------------------------------------------------------- loading

def load_inputs(submission_path, gold_path, input_path, compare_path=None, force=False):
    parsed = ev.precheck_or_die(submission_path, force=force)
    if parsed is None:
        # force=True: precheck skipped, load at our own risk
        sub, _ = ev.load_jsonl(submission_path)
    else:
        sub, _ = parsed
    gold, gold_order = ev.load_jsonl(gold_path)
    if set(sub.keys()) != set(gold.keys()):
        print("ERROR: submission and gold ids do not match exactly.", file=sys.stderr)
        sys.exit(1)
    conv_id = {}
    input_data = {}
    if input_path:
        inp, _ = ev.load_jsonl(input_path)
        for rid, rec in inp.items():
            conv_id[rid] = rec.get("conversation_id", "")
            input_data[rid] = rec
    compare_sub = None
    if compare_path:
        cparsed = ev.precheck_or_die(compare_path, force=force)
        if cparsed is None:
            csub, _ = ev.load_jsonl(compare_path)
        else:
            csub, _ = cparsed
        if set(csub.keys()) != set(gold.keys()):
            print("ERROR: compare submission ids do not match gold.", file=sys.stderr)
            sys.exit(1)
        compare_sub = csub
    return sub, gold, gold_order, conv_id, input_data, compare_sub


# ---------------------------------------------------------------- per-sample rows

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

# ---------------------------------------------------------------- analyzers (v2.2)

class EmotionAnalyzer:
    @staticmethod
    def analyze(ctx, rows):
        findings = []
        n = ctx.n
        emotion_support = ctx.emotion_support

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
                fix_direction=f"Read the examples; '{g}' vs '{p}' is a label-convention question SFT on gold should learn. Check it again after the pilot.",
                confidence="high" if c >= 30 else "medium",
            ))

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
                        fix_direction=f"Low recall on a rare class (support {support}). Check its count in the SFT data; oversampling is a P2 option only.",
                        confidence="high" if wrong >= 10 else "medium",
                    ))

        return findings


class ProfileAnalyzer:
    @staticmethod
    def analyze(ctx, rows):
        findings = []
        n = ctx.n
        points_per_field = PROFILE_POINTS / len(FIELDS)

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
                        fix_direction=f"Model over-predicts '{lab}'. See the measured filter result in Counterfactual Score Impact before acting.",
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
                        fix_direction=f"Model misses '{lab}'. Expected to improve with SFT; re-check after the pilot.",
                        confidence="medium",
                    ))

        return findings


class ResponseAnalyzer:
    @staticmethod
    def analyze(ctx, rows):
        findings = []
        n = ctx.n

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
                fix_direction="Replies much longer than gold cost latency; check reply-length median vs gold (decoding stays greedy).",
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
                fix_direction="Replies much shorter than gold. Do NOT add a length rule (EXP-011/012: costs emotion accuracy); SFT learns gold length.",
                confidence="high" if len(short_rows) >= 10 else "medium",
            ))

        hash_counts = Counter(r["template_hash"] for r in rows if r["pred_prefix"].strip())
        degenerate = [(h, c) for h, c in hash_counts.items() if c >= 5]
        if degenerate:
            for h, c in degenerate:
                ex_ids = [r["id"] for r in rows if r["template_hash"] == h][:5]
                severity = (c / n) * RESPONSE_POINTS
                findings.append(Finding(
                    category="response",
                    subcategory="template",
                    severity=severity,
                    count=c,
                    description=f"template repetition: {c} samples share prefix hash {h}",
                    impact_note=f"estimated_{severity:.2f}_pts_lost_on_{RESPONSE_POINTS}pt_response",
                    example_ids=ex_ids,
                    fix_direction="Read the repeated replies; if real, check the SFT targets for the same opening.",
                    confidence="high" if c >= 10 else "medium",
                ))

        return findings


class CorrelationAnalyzer:
    @staticmethod
    def analyze(ctx, rows):
        findings = []
        n = ctx.n

        combos = Counter()
        for r in rows:
            failures = []
            if not r["emotion_correct"]:
                failures.append("emotion")
            for f in FIELDS:
                if not r[f"{FIELD_SHORT[f]}_exact"]:
                    failures.append(f)
            if len(failures) >= 2:
                combos[tuple(sorted(failures))] += 1

        for combo, c in combos.most_common():
            if c < MIN_FINDING_COUNT:
                continue
            # crude independence check
            joint_rate = c / n
            marginal = 1.0
            for item in combo:
                if item == "emotion":
                    marginal *= sum(1 for r in rows if not r["emotion_correct"]) / n
                else:
                    marginal *= sum(1 for r in rows if not r[f"{FIELD_SHORT[item]}_exact"]) / n
            if joint_rate > marginal * 1.5:
                severity = (c / n) * EMOTION_POINTS
                findings.append(Finding(
                    category="correlation",
                    subcategory="systematic",
                    severity=severity,
                    count=c,
                    description=f"systematic multi-failure: {' + '.join(combo)} co-occur in {c} samples",
                    impact_note=f"estimated_{severity:.2f}_pts_lost_on_{EMOTION_POINTS}pt_accuracy",
                    example_ids=[r["id"] for r in rows if all(
                        (not r["emotion_correct"]) if item == "emotion" else (not r[f"{FIELD_SHORT[item]}_exact"])
                        for item in combo
                    )][:5],
                    fix_direction=f"Failures in {' + '.join(combo)} co-occur more than chance; inspect shared prompt/model root cause.",
                    confidence="medium",
                ))

        return findings

# ---------------------------------------------------------------------------
# v4.0 COMPONENTS
#   * signatures keyed on WHAT went wrong (taxonomy, gold, pred), not on context
#   * regimes = conditions OVER-REPRESENTED inside an error group (lift test),
#     never a union of conditions
#   * counterfactuals are MEASURED: apply the fix, re-score with eval.py
# ---------------------------------------------------------------------------

REGIME_MIN_SHARE = 0.60   # condition present in >= 60% of the group
REGIME_MIN_LIFT = 1.5     # ... and >= 1.5x its rate over all samples
REGIME_MAX_BASE = 0.80    # conditions present in > 80% of all samples say nothing
REGIME_MIN_HITS = 3       # ... and seen in >= 3 samples of the group
COLLAPSE_SHARE = 0.60     # one emotion label > 60% of predictions = collapse


class ConditionExtractor:
    """Per-sample context facets used to describe WHERE errors happen."""

    @staticmethod
    def get_user_message(input_rec):
        if not input_rec:
            return ""
        for turn in reversed(input_rec.get("history", []) or []):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return turn.get("content", "") or ""
        return ""

    @staticmethod
    def detect_keywords(text):
        text_lower = text.lower()
        return [cat for cat, kws in KEYWORDS.items() if any(kw in text_lower for kw in kws)]

    @staticmethod
    def turn_bucket(input_rec):
        tid = (input_rec or {}).get("target_user_turn_id")
        if tid is None:
            return None
        return "turn_1_3" if tid <= 3 else "turn_4_7" if tid <= 7 else "turn_8plus"

    @staticmethod
    def history_bucket(input_rec):
        hist = (input_rec or {}).get("history")
        if not hist:
            return None
        chars = sum(len(t.get("content", "") or "") for t in hist if isinstance(t, dict))
        return "hist_lt500" if chars < 500 else "hist_500_1500" if chars < 1500 else "hist_1500plus"

    @staticmethod
    def extract_conditions(row, input_rec):
        conds = []
        for c in (ConditionExtractor.turn_bucket(input_rec), ConditionExtractor.history_bucket(input_rec)):
            if c:
                conds.append(c)
        conds += [f"kw_{k}" for k in ConditionExtractor.detect_keywords(
            ConditionExtractor.get_user_message(input_rec))]
        flags = row["flags"]
        conds.append("reply_short" if "SHORT_PRED" in flags else "reply_long" if "LONG_PRED" in flags else "reply_ok")
        if "LANG_DRIFT" in flags:
            conds.append("lang_drift")
        return sorted(conds)


class SignatureBuilder:
    """One signature per error TYPE; conditions are tallied, not hashed."""

    @staticmethod
    def hash_signature(taxonomy, gold, pred):
        payload = json.dumps({"t": taxonomy, "g": gold, "p": pred}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def build_all(ctx, rows):
        groups = {}
        cond_cache = {}

        def add(taxonomy, gold, pred, row):
            key = (taxonomy, gold, pred)
            if key not in groups:
                cat, sub = taxonomy.split(".", 1)
                groups[key] = Signature(
                    sig_id=SignatureBuilder.hash_signature(taxonomy, gold, pred),
                    taxonomy=taxonomy, category=cat, subcategory=sub,
                    gold=gold, pred=pred, conditions=[], sample_ids=[], count=0)
            sig = groups[key]
            sig.sample_ids.append(row["id"])
            sig.count += 1

        for row in rows:
            cond_cache[row["id"]] = ConditionExtractor.extract_conditions(row, ctx.input_data.get(row["id"]))
            if not row["emotion_correct"]:
                add("emotion.confusion", row["emotion_gold"], row["emotion_pred"], row)
            for f in FIELDS:
                short = FIELD_SHORT[f]
                gs = set() if row[f"{short}_gold"] == "-" else set(row[f"{short}_gold"].split("|"))
                ps = set() if row[f"{short}_pred"] == "-" else set(row[f"{short}_pred"].split("|"))
                for lab in sorted(ps - gs):
                    add("profile.false_positive", f"{f}:{lab}", "", row)
                for lab in sorted(gs - ps):
                    add("profile.false_negative", f"{f}:{lab}", "", row)
            for flag, tax in (("LANG_DRIFT", "response.lang_drift"), ("EMPTY_PRED", "response.empty"),
                              ("LONG_PRED", "response.too_long"), ("SHORT_PRED", "response.too_short")):
                if flag in row["flags"]:
                    add(tax, "", "", row)

        ctx.conditions = cond_cache
        sigs = sorted(groups.values(), key=lambda s: (-s.count, s.taxonomy, s.gold, s.pred))
        return sigs


class RegimeAnalyzer:
    """Describe WHERE an error group concentrates: over-represented conditions only."""

    @staticmethod
    def base_rates(ctx):
        n = ctx.n
        counts = Counter(c for conds in ctx.conditions.values() for c in conds)
        return {c: k / n for c, k in counts.items()}

    @staticmethod
    def regime_for(sig, ctx, base):
        inside = Counter(c for sid in sig.sample_ids for c in ctx.conditions.get(sid, []))
        out = []
        for cond, hits in inside.items():
            share = hits / sig.count
            b = base.get(cond, 0.0)
            if (hits >= REGIME_MIN_HITS and share >= REGIME_MIN_SHARE and b <= REGIME_MAX_BASE
                    and b > 0 and share / b >= REGIME_MIN_LIFT):
                out.append({"condition": cond, "share": round(share, 2), "base": round(b, 2),
                            "lift": round(share / b, 2)})
        return sorted(out, key=lambda d: (-d["lift"], d["condition"]))

    @staticmethod
    def build_clusters(signatures, ctx, min_size=3, max_clusters=10):
        base = RegimeAnalyzer.base_rates(ctx)
        clusters = []
        for sig in signatures:
            if sig.count < min_size:
                continue
            regime = RegimeAnalyzer.regime_for(sig, ctx, base)
            sig.conditions = [r["condition"] for r in regime]
            clusters.append({
                "sig_id": sig.sig_id, "taxonomy": sig.taxonomy, "gold": sig.gold, "pred": sig.pred,
                "count": sig.count, "regime": regime, "conditions": sig.conditions,
                "sample_ids": list(sig.sample_ids),
            })
        return clusters  # ranked later by measured impact


class CounterfactualAnalyzer:
    """Apply the fix, re-score with eval.py. No formulas, no guessed weights."""

    @staticmethod
    def _profile_pairs(ctx, field, transform=None):
        pairs = []
        for rid in ctx.gold_order:
            s_prof = ctx.sub[rid].get("user_profile") if isinstance(ctx.sub[rid].get("user_profile"), dict) else {}
            g_prof = ctx.gold[rid].get("user_profile") if isinstance(ctx.gold[rid].get("user_profile"), dict) else {}
            pred = ev.to_label_set(s_prof.get(field))
            gold = ev.to_label_set(g_prof.get(field))
            if transform:
                pred = transform(rid, set(pred), gold)
            pairs.append((pred, gold))
        return pairs

    @staticmethod
    def _profile_scores(ctx, field, transform=None):
        res = ev.score_profile_field(CounterfactualAnalyzer._profile_pairs(ctx, field, transform),
                                     ev.PROFILE_VOCAB[field])
        return res["micro_f1"], res["exact_match"]

    @staticmethod
    def compute_impact(cluster, ctx):
        tax, n = cluster["taxonomy"], ctx.n
        ids = set(cluster["sample_ids"])
        if tax == "emotion.confusion":
            acc_now = sum(1 for rid in ctx.gold_order
                          if ctx.sub[rid].get("emotion_label") == ctx.gold[rid].get("emotion_label")) / n
            acc_fix = acc_now + len(ids) / n
            return {"kind": "oracle: every sample in this group labelled correctly",
                    "metric": "emotion_accuracy", "before": round(acc_now, 4), "after": round(acc_fix, 4),
                    "points": round((acc_fix - acc_now) * EMOTION_POINTS, 2), "points_exact": None,
                    "action": "fine-tuning target (SFT on gold labels); re-check after the pilot",
                    "measured": True}
        if tax.startswith("profile."):
            field, label = cluster["gold"].split(":", 1)
            f1_now, ex_now = CounterfactualAnalyzer._profile_scores(ctx, field)
            if tax == "profile.false_positive":
                # what a post-filter would do: drop the label EVERYWHERE (true positives lost too)
                f1_fix, ex_fix = CounterfactualAnalyzer._profile_scores(
                    ctx, field, lambda rid, p, g: p - {label})
                kind = f"filter: drop '{label}' from every prediction"
                action = (f"post-filter '{label}' in normalize_prediction"
                          if (f1_fix - f1_now) > 0 and (ex_fix - ex_now) >= 0
                          else f"do NOT filter '{label}' (hurts at least one profile metric)")
            else:
                f1_fix, ex_fix = CounterfactualAnalyzer._profile_scores(
                    ctx, field, lambda rid, p, g: (p | {label}) if rid in ids else p)
                kind = f"oracle: add '{label}' where gold has it (upper bound)"
                action = "upper bound only; SFT should learn when this label applies"
            w = PROFILE_POINTS / len(FIELDS)
            return {"kind": kind, "metric": f"profile/{field}",
                    "before": {"micro_f1": round(f1_now, 4), "exact_match": round(ex_now, 4)},
                    "after": {"micro_f1": round(f1_fix, 4), "exact_match": round(ex_fix, 4)},
                    "points": round((f1_fix - f1_now) * w, 2),          # if profile is scored by F1
                    "points_exact": round((ex_fix - ex_now) * w, 2),    # if scored by exact match
                    "action": action, "measured": True}
        # response: the official composite (BLEU/ROUGE/BERTScore weights) is unknown -> no points
        rl_in = [r for r in ctx.rows if r["id"] in ids and r.get("rouge_l") not in ("", None)]
        rl_out = [r for r in ctx.rows if r["id"] not in ids and r.get("rouge_l") not in ("", None)]
        mean = lambda xs: round(sum(float(r["rouge_l"]) for r in xs) / len(xs), 4) if xs else None
        return {"kind": "no point estimate (official reply composite unknown)",
                "metric": "rouge_l (group vs rest)", "before": mean(rl_in), "after": mean(rl_out),
                "points": None, "points_exact": None,
                "action": "inspect examples", "measured": rl_in != []}

    @staticmethod
    def rank_key(cf):
        pts = [p for p in (cf["impact"]["points"], cf["impact"]["points_exact"]) if p is not None]
        return -(max(pts) if pts else -1e9)


class DiagnosticTables:
    """Small, decision-relevant tables (pilot gate, ambiguity, context)."""

    @staticmethod
    def emotion_by_condition(ctx, rows):
        out = {}
        for prefix in ("turn_", "hist_"):
            buckets = defaultdict(lambda: [0, 0])
            for r in rows:
                for c in ctx.conditions.get(r["id"], []):
                    if c.startswith(prefix):
                        buckets[c][0] += r["emotion_correct"]
                        buckets[c][1] += 1
            out[prefix.rstrip("_")] = {c: {"n": t, "accuracy": round(k / t, 3)} for c, (k, t) in sorted(buckets.items())}
        return out

    @staticmethod
    def bidirectional_confusions(rows, min_each=2):
        conf = Counter((r["emotion_gold"], r["emotion_pred"]) for r in rows if not r["emotion_correct"])
        pairs = []
        for (a, b), ab in conf.items():
            ba = conf.get((b, a), 0)
            if a < b and ab >= min_each and ba >= min_each:
                pairs.append({"a": a, "b": b, "a_to_b": ab, "b_to_a": ba})
        return sorted(pairs, key=lambda d: -(d["a_to_b"] + d["b_to_a"]))

    @staticmethod
    def prediction_distribution(rows):
        n = len(rows)
        pred = Counter(r["emotion_pred"] for r in rows)
        gold = Counter(r["emotion_gold"] for r in rows)
        top, top_n = pred.most_common(1)[0] if pred else ("-", 0)
        return {"pred": dict(pred.most_common()), "gold": dict(gold.most_common()),
                "top_label": top, "top_share": round(top_n / n, 3) if n else 0.0,
                "collapse": (top_n / n) > COLLAPSE_SHARE if n else False}


class Genome:
    """Persistent error signature registry."""

    def __init__(self, registry_dir):
        self.registry_dir = registry_dir
        self.genome_path = os.path.join(registry_dir, "error_genome.json")
        self.experiments_path = os.path.join(registry_dir, "experiments.json")
        self.data = {"version": "4.0", "runs": [], "signatures": {}}
        self.experiments = {}
        self._load()

    def _load(self):
        if os.path.exists(self.genome_path):
            try:
                with open(self.genome_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.data = {"version": "4.0", "runs": [], "signatures": {}}
        if os.path.exists(self.experiments_path):
            try:
                with open(self.experiments_path, "r", encoding="utf-8") as f:
                    self.experiments = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.experiments = {}

    def save(self):
        os.makedirs(self.registry_dir, exist_ok=True)
        with open(self.genome_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        with open(self.experiments_path, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False)

    def update(self, run_label, signatures, experiment_notes=None):
        """Add current run signatures to genome. Returns classification.

        Classifications are relative to OTHER run_labels only, so rerunning
        with the same run_label produces stable output.
        """
        if not run_label:
            run_label = "unknown"

        novel = []
        persistent = []
        regressed = []
        resolved = []

        current_sig_ids = {sig.sig_id for sig in signatures}
        runs = self.data.setdefault("runs", [])
        previous_runs = [r for r in runs if r != run_label]   # registration order, deterministic

        for sig in signatures:
            sid = sig.sig_id
            if sid not in self.data["signatures"]:
                self.data["signatures"][sid] = {
                    "taxonomy": sig.taxonomy,
                    "category": sig.category,
                    "subcategory": sig.subcategory,
                    "gold": sig.gold,
                    "pred": sig.pred,
                    "conditions": [],
                    "first_seen": run_label,
                    "last_seen": run_label,
                    "sample_sets": {},
                    "count_history": {},
                }
                novel.append(sig)
            else:
                entry = self.data["signatures"][sid]
                entry["last_seen"] = run_label
                history = entry.get("count_history", {})
                # Only count OTHER run_labels for persistence
                prev_runs = [r for r in history if r != run_label]
                if len(prev_runs) >= 1:
                    persistent.append(sig)
                else:
                    novel.append(sig)

            self.data["signatures"][sid]["sample_sets"][run_label] = sig.sample_ids
            self.data["signatures"][sid]["count_history"][run_label] = sig.count

        # Resolved: present in last OTHER run, absent now
        last_other_run = self._get_last_other_run_label(run_label)
        if last_other_run:
            for sid, entry in self.data["signatures"].items():
                if sid in current_sig_ids:
                    continue
                history = entry.get("count_history", {})
                if last_other_run in history and history[last_other_run] > 0:
                    prev_counts = [history.get(r, 0) for r in list(history.keys()) if r != run_label]
                    solid_history = sum(1 for c in prev_counts if c >= 3) >= 2
                    if solid_history or len(prev_counts) < 2:
                        resolved.append(entry)

        # Regressed: was 0 in last OTHER run, back now
        for sig in signatures:
            sid = sig.sig_id
            entry = self.data["signatures"][sid]
            history = entry.get("count_history", {})
            prev_runs = [r for r in history if r != run_label]
            if len(prev_runs) >= 1 and previous_runs:
                last_count = history.get(previous_runs[-1], 0)
                if last_count == 0 and sig.count > 0:
                    if sig in persistent:
                        persistent.remove(sig)
                    if sig not in regressed:
                        regressed.append(sig)

        if run_label not in runs:
            runs.append(run_label)

        # Prune old runs per signature
        for sid, entry in self.data["signatures"].items():
            history = entry.get("count_history", {})
            if len(history) > 20:
                sorted_runs = sorted(history.keys())
                to_remove = sorted_runs[:-20]
                for r in to_remove:
                    del history[r]
                    if r in entry.get("sample_sets", {}):
                        del entry["sample_sets"][r]

        # Update experiment record (replace if same run_label)
        self.experiments[run_label] = {
            "notes": experiment_notes or "",
            "signatures": {
                "novel": len(novel),
                "persistent": len(persistent),
                "regressed": len(regressed),
                "resolved": len(resolved),
            }
        }

        return {
            "novel": novel,
            "persistent": persistent,
            "regressed": regressed,
            "resolved": resolved,
        }

    def _get_last_other_run_label(self, current_label):
        previous = [r for r in self.data.get("runs", []) if r != current_label]
        return previous[-1] if previous else None

    def get_signature_history(self, sig_id):
        return self.data["signatures"].get(sig_id, {})


# ---------------------------------------------------------------- hypothesis ranking

class HypothesisEngine:
    @staticmethod
    def rank(findings, max_results=7):
        dedup = {}
        for f in findings:
            key = (f.category, f.subcategory, f.description)
            if key not in dedup or f.count > dedup[key].count:
                dedup[key] = f
        filtered = [f for f in dedup.values() if f.count >= MIN_FINDING_COUNT]

        ranked = sorted(filtered, key=lambda f: (-f.severity, -f.count, f.description))
        hypotheses = []
        for i, f in enumerate(ranked[:max_results], 1):
            owner = OWNER_MAP.get(f"{f.category}_{f.subcategory}", "prompt_engineering")
            hypotheses.append(Hypothesis(
                rank=i,
                category=f.category,
                description=f.description,
                impact_note=f.impact_note,
                example_ids=f.example_ids,
                fix_direction=f.fix_direction,
                owner=owner,
                confidence=f.confidence,
                count=f.count,
            ))
        return hypotheses

# ---------------------------------------------------------------- reporting

def write_report(outdir, ctx, hypotheses, rows, signatures, genome_result,
                 clusters, counterfactuals, run_label, experiment_notes,
                 has_compare=False):
    n = ctx.n
    lines = []
    lines.append("# Error Analysis Report (analyze.py v4)")
    lines.append("")
    lines.append("> Exploratory diagnosis. Scores come from eval.py; regimes are statistical descriptions, "
                 "not causes. Decide on n50 + n50b, never on one slice.")
    lines.append("")
    lines.append(f"**submission**: `{os.path.basename(getattr(ctx, '_submission_path', 'unknown'))}`  ")
    lines.append(f"**samples**: n = {n}  ")
    lines.append(f"**run_label**: {run_label}  ")
    if experiment_notes:
        lines.append(f"**experiment_notes**: {experiment_notes}  ")
    lines.append("")

    # summary (backwards compatible)
    emo_correct = sum(r["emotion_correct"] for r in rows)
    emo_errors = n - emo_correct
    lines.append(f"Emotion errors: {emo_errors} / {n}  ({emo_errors/n:.1%})")
    lines.append("")

    # metrics citation (A07)
    if ctx.metrics:
        m = ctx.metrics
        if "emotion" in m and "accuracy" in m["emotion"]:
            lines.append(f"Official metrics -- emotion accuracy: {m['emotion']['accuracy']:.4f}")
        if "profile" in m:
            for f in FIELDS:
                if f in m["profile"] and "exact_match" in m["profile"][f]:
                    lines.append(f"Official metrics -- profile/{f} exact-match: {m['profile'][f]['exact_match']:.4f}")
        lines.append("")

    # v2 hypotheses
    lines.append("## Ranked Hypotheses")
    lines.append("")
    if not hypotheses:
        lines.append("No significant error buckets.")
    for h in hypotheses:
        lines.append(f"### {h.rank}. [{h.category}] {h.description}")
        lines.append(f"- **impact**: {h.impact_note}")
        lines.append(f"- **owner**: {h.owner}")
        lines.append(f"- **confidence**: {h.confidence}")
        lines.append(f"- **fix**: {h.fix_direction}")
        if h.example_ids:
            lines.append(f"- **examples**: {', '.join(h.example_ids[:5])}")
        lines.append("")

    # comparative delta (A12)
    if has_compare:
        lines.append("## Comparative Delta")
        lines.append("")
        lines.append("See `delta.csv` for per-sample A/B comparison.")
        lines.append("")

    # v3 taxonomy
    lines.append("## Error Taxonomy")
    lines.append("")
    tax_counts = Counter(s.taxonomy for s in signatures)
    lines.append("| taxonomy | count | description |")
    lines.append("|---|---|---|")
    for tax, count in tax_counts.most_common():
        cat, sub = tax.split(".", 1)
        desc = TAXONOMY.get(cat, {}).get(sub, "")
        lines.append(f"| {tax} | {count} | {desc} |")
    lines.append("")

    # v3 genome
    lines.append("## Error Genome")
    lines.append("")
    lines.append("| status | count | description |")
    lines.append("|---|---|---|")
    for status, sigs in genome_result.items():
        if status == "resolved":
            desc = "Was present in previous runs, absent now"
        elif status == "persistent":
            desc = "Seen in previous runs, still present"
        elif status == "regressed":
            desc = "Was resolved in last run, back now"
        elif status == "novel":
            desc = "First time seen"
        else:
            desc = ""
        lines.append(f"| {status} | {len(sigs)} | {desc} |")
    lines.append("")

    # v4 diagnostic tables
    diag = ctx.diagnostics
    lines.append("## Emotion Prediction Distribution (collapse check)")
    lines.append("")
    pdist = diag["prediction_distribution"]
    verdict = "**COLLAPSE**" if pdist["collapse"] else "ok"
    lines.append(f"Top predicted label: `{pdist['top_label']}` = {pdist['top_share']:.0%} of predictions "
                 f"(collapse if > {COLLAPSE_SHARE:.0%}): {verdict}")
    lines.append("")
    lines.append("| label | predicted | gold |")
    lines.append("|---|---|---|")
    for lab in sorted(set(pdist["pred"]) | set(pdist["gold"]), key=lambda l: (-pdist["gold"].get(l, 0), -pdist["pred"].get(l, 0), l)):
        lines.append(f"| {lab} | {pdist['pred'].get(lab, 0)} | {pdist['gold'].get(lab, 0)} |")
    lines.append("")

    lines.append("## Emotion Accuracy by Context")
    lines.append("")
    for dim, title in (("turn", "target turn"), ("hist", "history length (chars)")):
        table = diag["emotion_by_condition"].get(dim, {})
        if not table:
            lines.append(f"_{title}: needs --input_")
            continue
        lines.append(f"| {title} | n | accuracy |")
        lines.append("|---|---|---|")
        for c, v in table.items():
            lines.append(f"| {c} | {v['n']} | {v['accuracy']:.3f} |")
        lines.append("")

    lines.append("## Bidirectional Confusions (label-ambiguity signal)")
    lines.append("")
    bidi = diag["bidirectional"]
    if not bidi:
        lines.append("None (no pair confused both ways >= 2 times).")
    else:
        lines.append("| a | b | a->b | b->a |")
        lines.append("|---|---|---|---|")
        for d in bidi:
            lines.append(f"| {d['a']} | {d['b']} | {d['a_to_b']} | {d['b_to_a']} |")
    lines.append("")

    # v4 regimes
    lines.append("## Error Groups and Regimes")
    lines.append("")
    lines.append(f"_A regime condition is listed only if it covers >= {REGIME_MIN_SHARE:.0%} of the group, "
                 f"is >= {REGIME_MIN_LIFT}x more frequent than overall, and appears in <= {REGIME_MAX_BASE:.0%} of all samples._")
    lines.append("")
    if not clusters:
        lines.append("No error group with >= 3 samples.")
    else:
        lines.append("| error | size | regime (share / overall, lift) |")
        lines.append("|---|---|---|")
        for c in clusters:
            reg = "; ".join(f"{r['condition']} ({r['share']:.0%} vs {r['base']:.0%}, x{r['lift']})" for r in c["regime"]) or "none (spread out)"
            lines.append(f"| {c['taxonomy']} {c['gold']}{('->' + c['pred']) if c['pred'] else ''} | {c['count']} | {reg} |")
    lines.append("")

    # v4 measured counterfactuals
    lines.append("## Counterfactual Score Impact (measured)")
    lines.append("")
    lines.append("_Each fix is applied to the predictions and re-scored with eval.py. Emotion rows are oracle "
                 "upper bounds; profile filter rows are exactly what a post-filter would do (true positives "
                 "lost included). Profile points are shown for both candidate metrics because the official "
                 "one is unconfirmed. Reply groups get no points: the official composite is unknown._")
    lines.append("")
    if not counterfactuals:
        lines.append("No counterfactuals available.")
    else:
        lines.append("| error | fix simulated | before -> after | pts (emotion; profile if F1) | pts (profile if exact) | action |")
        lines.append("|---|---|---|---|---|---|")
        for cf in counterfactuals:
            c, im = cf["cluster"], cf["impact"]
            label = f"{c['taxonomy']} {c['gold']}{('->' + c['pred']) if c['pred'] else ''}"
            if isinstance(im["before"], dict):
                ba = (f"F1 {im['before']['micro_f1']:.3f}->{im['after']['micro_f1']:.3f}, "
                      f"exact {im['before']['exact_match']:.3f}->{im['after']['exact_match']:.3f}")
            elif im["before"] is None:
                ba = "-"
            else:
                ba = f"{im['before']}->{im['after']}"
            fmt = lambda v: "-" if v is None else f"{v:+.2f}"
            lines.append(f"| {label} | {im['kind']} | {ba} | {fmt(im['points'])} | {fmt(im['points_exact'])} | {im['action']} |")
    lines.append("")

    actionable = [cf for cf in counterfactuals if cf["cluster"]["taxonomy"] == "profile.false_positive"
                  and (cf["impact"]["points"] or 0) > 0 and (cf["impact"]["points_exact"] or 0) >= 0]
    lines.append("## Recommended Next Experiment")
    lines.append("")
    if actionable:
        best = actionable[0]
        c, im = best["cluster"], best["impact"]
        lines.append(f"**Cheapest measured win:** {im['action']} "
                     f"({fmt(im['points'])} pts if profile is F1, {fmt(im['points_exact'])} pts if exact match).")
        lines.append("Confirm on a second val slice before shipping: a filter found and measured on one slice overfits it.")
    else:
        lines.append("No post-filter improves both profile metrics here. Emotion and reply errors are SFT targets; "
                     "compare the next model with `--compare` on n50 and n50b.")
    lines.append("")

    # owner summary (counts only: summing overlapping estimates is meaningless)
    lines.append("## Owner Summary")
    lines.append("")
    owner_counts = Counter()
    for h in hypotheses:
        owner_counts[h.owner] += h.count
    lines.append("| owner | hypotheses' affected samples |")
    lines.append("|---|---|")
    for owner, cnt in sorted(owner_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {owner} | {cnt} |")
    lines.append("")

    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_path}")


# ---------------------------------------------------------------- analysis.json

def write_analysis_json(outdir, ctx, hypotheses, rows, signatures, genome_result,
                        clusters, counterfactuals, run_label, experiment_notes):
    out = {
        "schema_version": "4.0",
        "meta": {
            "submission": getattr(ctx, '_submission_path', ''),
            "run_label": run_label,
            "experiment_notes": experiment_notes or "",
            "n": ctx.n,
        },
        "summary": {
            "total_mistakes": sum(r["any_mistake"] for r in rows),
            "emotion_accuracy": sum(r["emotion_correct"] for r in rows) / ctx.n,
            "profile_fields": {
                f: sum(r[f"{FIELD_SHORT[f]}_exact"] for r in rows) / ctx.n
                for f in FIELDS
            },
        },
        "hypotheses": [h.to_dict() for h in hypotheses],
        "genome": {
            "novel_count": len(genome_result.get("novel", [])),
            "persistent_count": len(genome_result.get("persistent", [])),
            "regressed_count": len(genome_result.get("regressed", [])),
            "resolved_count": len(genome_result.get("resolved", [])),
        },
        "taxonomy": [
            {"taxonomy": tax, "count": count}
            for tax, count in Counter(s.taxonomy for s in signatures).most_common()
        ],
        "diagnostics": ctx.diagnostics,
        "clusters": [{k: v for k, v in c.items() if k != "sample_ids"} | {"examples": c["sample_ids"][:5]}
                     for c in clusters],
        "counterfactuals": [
            {"taxonomy": cf["cluster"]["taxonomy"], "gold": cf["cluster"]["gold"], "pred": cf["cluster"]["pred"],
             "count": cf["cluster"]["count"], **cf["impact"],
             "score_gain": max([p for p in (cf["impact"]["points"], cf["impact"]["points_exact"]) if p is not None] or [0.0])}
            for cf in counterfactuals
        ],
    }
    path = os.path.join(outdir, "analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}")


# ---------------------------------------------------------------- delta report

def write_delta_csv(outdir, gold, gold_order, sub, compare_sub, skip_rq):
    rows = []
    for rid in gold_order:
        sg, cg = sub[rid].get("emotion_label"), compare_sub[rid].get("emotion_label")
        sg_prof = sub[rid].get("user_profile") if isinstance(sub[rid].get("user_profile"), dict) else {}
        cg_prof = compare_sub[rid].get("user_profile") if isinstance(compare_sub[rid].get("user_profile"), dict) else {}
        row = {"id": rid}
        row["emotion_old"] = cg if cg in ev.EMOTIONS else "-"
        row["emotion_new"] = sg if sg in ev.EMOTIONS else "-"
        row["emotion_delta"] = 1 if sg != cg else 0
        for f in FIELDS:
            short = FIELD_SHORT[f]
            row[f"{short}_old"] = join_labels(ev.to_label_set(cg_prof.get(f)))
            row[f"{short}_new"] = join_labels(ev.to_label_set(sg_prof.get(f)))
            row[f"{short}_delta"] = 0 if row[f"{short}_old"] == row[f"{short}_new"] else 1
        if not skip_rq:
            s_text = sub[rid].get("response_text") or ""
            c_text = compare_sub[rid].get("response_text") or ""
            row["rouge_l"] = round(ev.rouge_l_macro([s_text], [c_text]), 3)
        else:
            row["rouge_l"] = ""
        rows.append(row)

    delta_path = os.path.join(outdir, "delta.csv")
    cols = ["id", "emotion_old", "emotion_new", "emotion_delta"]
    for f in FIELDS:
        short = FIELD_SHORT[f]
        cols += [f"{short}_old", f"{short}_new", f"{short}_delta"]
    cols.append("rouge_l")
    with open(delta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {delta_path}")


# ---------------------------------------------------------------- mistakes.csv

def fill_rouge(rows, skip_rq, sub=None, gold=None):
    """Per-sample ROUGE-L via eval.py (computed once, reused by CSV and reply groups)."""
    for r in rows:
        if r.get("rouge_l") not in ("", None):
            continue
        if skip_rq or sub is None or gold is None:
            r["rouge_l"] = ""
        else:
            s_text = sub.get(r["id"], {}).get("response_text", "") or ""
            g_text = gold.get(r["id"], {}).get("response_text", "") or ""
            r["rouge_l"] = round(ev.rouge_l_macro([s_text], [g_text]), 3)


def write_mistakes_csv(outdir, rows, skip_rq, sub=None, gold=None, mistakes_only=False):
    fill_rouge(rows, skip_rq, sub, gold)
    mistakes_path = os.path.join(outdir, "mistakes.csv")
    with open(mistakes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            if mistakes_only and r.get("any_mistake") != 1:
                continue
            out = {k: r.get(k, "") for k in CSV_COLUMNS}
            writer.writerow(out)
    print(f"Wrote {mistakes_path}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Layer-3 DIAGNOSIS v4 (measured counterfactuals, regimes, genome)")
    ap.add_argument("submission", help="Submission JSONL file")
    ap.add_argument("gold", help="Gold JSONL file")
    ap.add_argument("--input", dest="input_path", help="Input JSONL (for conversation-level features)")
    ap.add_argument("--metrics", dest="metrics_path", help="Optional metrics.json (used for counterfactual baselines)")
    ap.add_argument("--compare", dest="compare_path", help="Optional old submission JSONL for A/B delta")
    ap.add_argument("--outdir", help="Output directory (default: <submission>.analysis/)")
    ap.add_argument("--mistakes-only", action="store_true",
                    help="Only emit mistakes.csv (skip full report)")
    ap.add_argument("--skip-response-quality", action="store_true",
                    help="Skip per-sample BLEU/ROUGE-L in mistakes.csv (faster; for CI/tests)")
    ap.add_argument("--force", action="store_true",
                    help="Pass through to precheck_or_die: allow submissions that would normally be rejected (e.g. blank emotion labels)")
    ap.add_argument("--run-label", help="Label for this run in the genome registry")
    ap.add_argument("--registry", default=".analyze_registry",
                    help="Directory for genome + experiment registry")
    ap.add_argument("--experiment-notes", help="Notes about what changed in this run")
    ap.add_argument("--owner-map", dest="owner_map", help="JSON file mapping category_subcategory -> owner")
    args = ap.parse_args()

    # load owner map override
    if args.owner_map and os.path.exists(args.owner_map):
        with open(args.owner_map, "r", encoding="utf-8") as f:
            override = json.load(f)
        OWNER_MAP.update(override)

    outdir = args.outdir or (args.submission + ".analysis")
    os.makedirs(outdir, exist_ok=True)

    # Default registry inside outdir for isolation; override with --registry
    registry = args.registry
    if registry == ".analyze_registry":
        registry = os.path.join(outdir, ".analyze_registry")

    sub, gold, gold_order, conv_id, input_data, compare_sub = load_inputs(
        args.submission, args.gold, args.input_path, args.compare_path, force=args.force)

    run_label = args.run_label or os.path.basename(args.submission).replace(".jsonl", "")

    rows = [build_row(rid, sub, gold, conv_id) for rid in gold_order]
    for r in rows:
        r["rouge_l"] = ""

    ctx = AnalysisContext(sub, gold, gold_order, conv_id, input_data)
    ctx._submission_path = args.submission

    if args.metrics_path and os.path.exists(args.metrics_path):
        with open(args.metrics_path, "r", encoding="utf-8") as f:
            ctx.metrics = json.load(f)

    analyzers = [EmotionAnalyzer, ProfileAnalyzer, ResponseAnalyzer, CorrelationAnalyzer]
    findings = []
    for A in analyzers:
        findings.extend(A.analyze(ctx, rows))

    hypotheses = HypothesisEngine.rank(findings)

    signatures = SignatureBuilder.build_all(ctx, rows)

    genome = Genome(registry)
    genome_result = genome.update(run_label, signatures, args.experiment_notes)
    genome.save()

    # per-sample ROUGE-L first: reply-group comparisons need it
    fill_rouge(rows, args.skip_response_quality, sub, gold)
    ctx.rows = rows
    ctx.diagnostics = {
        "prediction_distribution": DiagnosticTables.prediction_distribution(rows),
        "emotion_by_condition": DiagnosticTables.emotion_by_condition(ctx, rows),
        "bidirectional": DiagnosticTables.bidirectional_confusions(rows),
    }

    clusters = RegimeAnalyzer.build_clusters(signatures, ctx)
    counterfactuals = [{"cluster": c, "impact": CounterfactualAnalyzer.compute_impact(c, ctx)} for c in clusters]
    counterfactuals.sort(key=lambda cf: (CounterfactualAnalyzer.rank_key(cf), -cf["cluster"]["count"],
                                         cf["cluster"]["taxonomy"], cf["cluster"]["gold"], cf["cluster"]["pred"]))
    clusters = [cf["cluster"] for cf in counterfactuals][:12]
    counterfactuals = counterfactuals[:12]

    write_mistakes_csv(outdir, rows, args.skip_response_quality, sub, gold,
                       mistakes_only=args.mistakes_only)

    if args.compare_path and compare_sub:
        write_delta_csv(outdir, gold, gold_order, sub, compare_sub, args.skip_response_quality)

    write_report(outdir, ctx, hypotheses, rows, signatures, genome_result,
                 clusters, counterfactuals, run_label, args.experiment_notes,
                 args.compare_path and compare_sub)
    write_analysis_json(outdir, ctx, hypotheses, rows, signatures, genome_result,
                        clusters, counterfactuals, run_label, args.experiment_notes)


if __name__ == "__main__":
    main()
