#!/usr/bin/env python3
r"""
scripts/analyze.py -- Layer 3 DIAGNOSIS v4.1

Reads a submission + gold (+ input) and explains WHERE the errors are.
It never computes official numbers itself: every score comes from eval.py.

v4 (Oct 6) over v3:
  * one signature per error type (taxonomy, gold, pred); context is tallied, not hashed
  * counterfactuals are MEASURED: the fix is applied and re-scored with eval.py
  * tables: prediction distribution / collapse check, emotion accuracy by turn and
    history length, bidirectional confusions
  * genome: stable signatures, deterministic run order
  * fix advice limited to what the plan allows (no synthetic data, no temperature)

v4.1 (Oct 6, review fixes):
  * regimes need a group of >= 8 and a hypergeometric test (Bonferroni p < 0.05):
    the null test (conditions shuffled 200x) fell from 6.7 "regimes" to 0.05 on n50
  * --compare: improved / regressed / unchanged vs gold with net deltas (v2.2 semantics)
  * hypotheses ranked by affected samples; the formula point estimates are gone
  * counterfactuals in three separate tables: filters (achievable), oracle upper bounds, replies
  * genome compares only runs on the same gold file (fingerprint) and is hidden without history

Usage:
    python scripts/analyze.py <submission.jsonl> <gold.jsonl> [--input <input.jsonl>]
        [--compare <old_submission.jsonl>] [--outdir DIR] [--force]
        [--run-label NAME] [--registry DIR] [--experiment-notes TEXT]
"""

import argparse
import csv
import hashlib
import json
import math
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

# Official point weights (used only by MEASURED counterfactuals)
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
        "lang_drift": "Reply mostly non-Chinese while gold is Chinese (CJK ratio < 0.70)",
        "empty": "Empty reply",
        "too_long": "Reply > 2x gold length (and >= 60 chars)",
        "too_short": "Reply < 0.5x gold length (gold >= 30 chars)",
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


def short_path(path, parts=3):
    """Last few path components: submissions are all called submission.jsonl."""
    comps = os.path.normpath(path).replace("\\", "/").split("/")
    return "/".join(comps[-parts:])


def file_fingerprint(path):
    """sha256 of the gold file (first 16 hex): the genome only compares runs on the same slice."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


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
            findings.append(Finding(
                category="emotion",
                subcategory="confusion",
                severity=0.0,
                count=c,
                description=f"emotion: {g} misclassified as {p} ({c}x)",
                impact_note="",
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
                    findings.append(Finding(
                        category="emotion",
                        subcategory="rare_class",
                        severity=0.0,
                        count=wrong,
                        description=f"rare class '{cls}' recall {recall:.2f} (support {support})",
                        impact_note="",
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

        for f in FIELDS:
            short = FIELD_SHORT[f]
            exact = sum(1 for r in rows if r[f"{short}_exact"])
            rate = exact / n
            if rate < 0.90:
                findings.append(Finding(
                    category="profile",
                    subcategory="field_weak",
                    severity=0.0,
                    count=n - exact,
                    description=f"profile/{f}: exact-match {rate:.3f}",
                    impact_note="",
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
                    findings.append(Finding(
                        category="profile",
                        subcategory="bias_over",
                        severity=0.0,
                        count=pred_counts[lab],
                        description=f"profile/{f}: over-predicts '{lab}' ({pred_counts[lab]}x vs gold {gold_counts[lab]}x)",
                        impact_note="",
                        example_ids=[r["id"] for r in rows if lab in r[f"{short}_pred"].split("|") and lab not in r[f"{short}_gold"].split("|")][:5],
                        fix_direction=f"Model over-predicts '{lab}'. See the measured filter result in Counterfactual Score Impact before acting.",
                        confidence="medium",
                    ))
                if g_rate > 0 and g_rate / max(p_rate, 1e-9) > 2.0 and gold_counts[lab] >= SMALL_SAMPLE_FLOOR:
                    findings.append(Finding(
                        category="profile",
                        subcategory="bias_under",
                        severity=0.0,
                        count=gold_counts[lab],
                        description=f"profile/{f}: under-predicts '{lab}' ({pred_counts[lab]}x vs gold {gold_counts[lab]}x)",
                        impact_note="",
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
            findings.append(Finding(
                category="response",
                subcategory="drift",
                severity=0.0,
                count=len(drift_rows),
                description=f"language drift: {len(drift_rows)} replies below CJK ratio {LANG_DRIFT_THRESHOLD}",
                impact_note="",
                example_ids=[r["id"] for r in drift_rows][:5],
                fix_direction="Check system-prompt language and SFT data mix; ensure Chinese dominance.",
                confidence="high" if len(drift_rows) >= 10 else "medium",
            ))

        long_rows = [r for r in rows if "LONG_PRED" in r["flags"]]
        short_rows = [r for r in rows if "SHORT_PRED" in r["flags"]]
        if long_rows:
            findings.append(Finding(
                category="response",
                subcategory="length_outlier",
                severity=0.0,
                count=len(long_rows),
                description=f"long predictions: {len(long_rows)} replies > 2x gold length",
                impact_note="",
                example_ids=[r["id"] for r in long_rows][:5],
                fix_direction="Replies much longer than gold cost latency; check reply-length median vs gold (decoding stays greedy).",
                confidence="high" if len(long_rows) >= 10 else "medium",
            ))
        if short_rows:
            findings.append(Finding(
                category="response",
                subcategory="length_outlier",
                severity=0.0,
                count=len(short_rows),
                description=f"short predictions: {len(short_rows)} replies < 0.5x gold length",
                impact_note="",
                example_ids=[r["id"] for r in short_rows][:5],
                fix_direction="Replies much shorter than gold. Do NOT add a length rule (EXP-011/012: costs emotion accuracy); SFT learns gold length.",
                confidence="high" if len(short_rows) >= 10 else "medium",
            ))

        hash_counts = Counter(r["template_hash"] for r in rows if r["pred_prefix"].strip())
        degenerate = [(h, c) for h, c in hash_counts.items() if c >= 5]
        if degenerate:
            for h, c in degenerate:
                ex_ids = [r["id"] for r in rows if r["template_hash"] == h][:5]
                findings.append(Finding(
                    category="response",
                    subcategory="template",
                    severity=0.0,
                    count=c,
                    description=f"template repetition: {c} samples share prefix hash {h}",
                    impact_note="",
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
                findings.append(Finding(
                    category="correlation",
                    subcategory="systematic",
                    severity=0.0,
                    count=c,
                    description=f"systematic multi-failure: {' + '.join(combo)} co-occur in {c} samples",
                    impact_note="",
                    example_ids=[r["id"] for r in rows if all(
                        (not r["emotion_correct"]) if item == "emotion" else (not r[f"{FIELD_SHORT[item]}_exact"])
                        for item in combo
                    )][:5],
                    fix_direction=f"Failures in {' + '.join(combo)} co-occur more than chance; inspect shared prompt/model root cause.",
                    confidence="medium",
                ))

        return findings

# ---------------------------------------------------------------------------
# v4 COMPONENTS
#   * signatures keyed on WHAT went wrong (taxonomy, gold, pred), not on context
#   * regimes = conditions OVER-REPRESENTED inside an error group (lift test),
#     never a union of conditions
#   * counterfactuals are MEASURED: apply the fix, re-score with eval.py
# ---------------------------------------------------------------------------

REGIME_MIN_SHARE = 0.60   # condition present in >= 60% of the group
REGIME_MIN_LIFT = 1.5     # ... and >= 1.5x its rate over all samples
REGIME_MAX_BASE = 0.80    # conditions present in > 80% of all samples say nothing
REGIME_MIN_HITS = 3       # ... and seen in >= 3 samples of the group
REGIME_MIN_GROUP = 8      # smaller groups are not tested for a regime at all
REGIME_MAX_P = 0.05       # one-sided hypergeometric p, Bonferroni-corrected per group
COLLAPSE_SHARE = 0.60     # one emotion label > 60% of predictions = collapse
CF_TYPES = ("filter", "oracle", "reply")
CF_PER_TYPE = 8
MAX_GROUPS_SHOWN = 15


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
    def hypergeom_sf(hits, group, cond_total, n):
        """P(X >= hits): X = samples with the condition among `group` drawn from n without replacement."""
        denom = math.comb(n, group)
        top = min(group, cond_total)
        return min(1.0, sum(math.comb(cond_total, k) * math.comb(n - cond_total, group - k)
                            for k in range(hits, top + 1)) / denom)

    @staticmethod
    def regime_for(sig, ctx, base):
        """Over-represented conditions that also pass a significance test.

        Without the test, n=50 produces 'regimes' from pure noise (null test: shuffling
        conditions across samples gave as many regimes as the real data).
        """
        if sig.count < REGIME_MIN_GROUP:
            return None  # too small to test
        ids = set(sig.sample_ids)   # a sample can appear once per signature
        inside = Counter(c for sid in ids for c in ctx.conditions.get(sid, []))
        k, n = len(ids), ctx.n
        # reply-length / language conditions DEFINE the response groups: testing them there is circular
        circular = ("reply_", "lang_drift") if sig.category == "response" else ()
        tested = [c for c in inside if base.get(c, 0.0) <= REGIME_MAX_BASE and not (circular and c.startswith(circular))]
        m = max(1, len(tested))     # Bonferroni over the conditions tested for this group
        out = []
        for cond in sorted(tested):
            hits = inside[cond]
            share = hits / k
            b = base.get(cond, 0.0)
            if not (hits >= REGIME_MIN_HITS and share >= REGIME_MIN_SHARE and b > 0
                    and share / b >= REGIME_MIN_LIFT):
                continue
            p = RegimeAnalyzer.hypergeom_sf(hits, k, round(b * n), n) * m
            if p < REGIME_MAX_P:
                out.append({"condition": cond, "share": round(share, 2), "base": round(b, 2),
                            "lift": round(share / b, 2), "p_adj": round(min(p, 1.0), 4)})
        return sorted(out, key=lambda d: (d["p_adj"], -d["lift"], d["condition"]))

    @staticmethod
    def build_clusters(signatures, ctx, min_size=3):
        base = RegimeAnalyzer.base_rates(ctx)
        clusters = []
        for sig in signatures:
            if sig.count < min_size:
                continue
            regime = RegimeAnalyzer.regime_for(sig, ctx, base)
            sig.conditions = [r["condition"] for r in (regime or [])]
            clusters.append({
                "sig_id": sig.sig_id, "taxonomy": sig.taxonomy, "gold": sig.gold, "pred": sig.pred,
                "count": sig.count, "regime": regime or [], "regime_tested": regime is not None,
                "conditions": sig.conditions,
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
            return {"type": "oracle", "kind": "oracle: every sample in this group labelled correctly",
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
                cf_type = "filter"
                kind = f"filter: drop '{label}' from every prediction"
                action = (f"post-filter '{label}' in normalize_prediction"
                          if (f1_fix - f1_now) > 0 and (ex_fix - ex_now) >= 0
                          else f"do NOT filter '{label}' (hurts at least one profile metric)")
            else:
                f1_fix, ex_fix = CounterfactualAnalyzer._profile_scores(
                    ctx, field, lambda rid, p, g: (p | {label}) if rid in ids else p)
                cf_type = "oracle"
                kind = f"oracle: add '{label}' where gold has it (upper bound)"
                action = "upper bound only; SFT should learn when this label applies"
            w = PROFILE_POINTS / len(FIELDS)
            # points assume profile score = mean of the 3 per-field metrics x 15 (5 pts per field)
            return {"type": cf_type, "kind": kind, "metric": f"profile/{field}",
                    "before": {"micro_f1": round(f1_now, 4), "exact_match": round(ex_now, 4)},
                    "after": {"micro_f1": round(f1_fix, 4), "exact_match": round(ex_fix, 4)},
                    "points": round((f1_fix - f1_now) * w, 2),          # if profile is scored by F1
                    "points_exact": round((ex_fix - ex_now) * w, 2),    # if scored by exact match
                    "action": action, "measured": True}
        # response: the official composite (BLEU/ROUGE/BERTScore weights) is unknown -> no points
        rl_in = [r for r in ctx.rows if r["id"] in ids and r.get("rouge_l") not in ("", None)]
        rl_out = [r for r in ctx.rows if r["id"] not in ids and r.get("rouge_l") not in ("", None)]
        mean = lambda xs: round(sum(float(r["rouge_l"]) for r in xs) / len(xs), 4) if xs else None
        return {"type": "reply", "kind": "no point estimate (official reply composite unknown)",
                "metric": "rouge_l (group vs rest)", "before": None, "after": None,
                "group_rouge_l": mean(rl_in), "rest_rouge_l": mean(rl_out),
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
        self.data = {"version": "4.1", "runs": [], "run_meta": {}, "signatures": {}}
        self.experiments = {}
        self._load()

    def _load(self):
        if os.path.exists(self.genome_path):
            try:
                with open(self.genome_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.data = {"version": "4.1", "runs": [], "run_meta": {}, "signatures": {}}
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

    def update(self, run_label, signatures, experiment_notes=None, gold_fp=None, n=None):
        """Add current run signatures to genome. Returns classification.

        Classifications are relative to OTHER run_labels only, so rerunning
        with the same run_label produces stable output. v4.1: only runs scored on
        the SAME gold file (fingerprint) are compared; others are listed as skipped.
        """
        if not run_label:
            run_label = "unknown"

        novel, persistent, regressed, resolved = [], [], [], []

        current_sig_ids = {sig.sig_id for sig in signatures}
        runs = self.data.setdefault("runs", [])
        meta = self.data.setdefault("run_meta", {})
        # registration order, deterministic; same slice only
        comparable = [r for r in runs if r != run_label and meta.get(r, {}).get("gold_fp") == gold_fp]
        self.skipped_runs = [r for r in runs if r != run_label and r not in comparable]
        self.comparable_runs = comparable
        comp = set(comparable)

        for sig in signatures:
            sid = sig.sig_id
            if sid not in self.data["signatures"]:
                self.data["signatures"][sid] = {
                    "taxonomy": sig.taxonomy, "category": sig.category, "subcategory": sig.subcategory,
                    "gold": sig.gold, "pred": sig.pred, "conditions": [],
                    "first_seen": run_label, "last_seen": run_label,
                    "sample_sets": {}, "count_history": {},
                }
            entry = self.data["signatures"][sid]
            entry["last_seen"] = run_label
            history = entry.get("count_history", {})
            prev = [r for r in comparable if history.get(r, 0) > 0]
            if not prev:
                novel.append(sig)
            elif history.get(comparable[-1], 0) == 0:
                regressed.append(sig)     # seen before on this slice, absent in the last run, back now
            else:
                persistent.append(sig)
            entry["sample_sets"][run_label] = sig.sample_ids
            entry["count_history"][run_label] = sig.count

        # Resolved: present in the last comparable run, absent now
        if comparable:
            last = comparable[-1]
            for sid in sorted(self.data["signatures"]):
                entry = self.data["signatures"][sid]
                if sid in current_sig_ids:
                    continue
                if entry.get("count_history", {}).get(last, 0) > 0:
                    resolved.append(entry)

        if run_label not in runs:
            runs.append(run_label)
        meta[run_label] = {"gold_fp": gold_fp, "n": n}

        # Prune: keep the 20 most recent runs (registration order) per signature
        keep = set(runs[-20:])
        for entry in self.data["signatures"].values():
            for key in ("count_history", "sample_sets"):
                for r in [r for r in entry.get(key, {}) if r not in keep]:
                    del entry[key][r]

        self.experiments[run_label] = {
            "notes": experiment_notes or "",
            "gold_fp": gold_fp,
            "compared_with": list(comparable),
            "signatures": {"novel": len(novel), "persistent": len(persistent),
                           "regressed": len(regressed), "resolved": len(resolved)},
        }
        return {"novel": novel, "persistent": persistent, "regressed": regressed, "resolved": resolved}

    def _get_last_other_run_label(self, current_label):
        previous = [r for r in self.data.get("runs", []) if r != current_label]
        return previous[-1] if previous else None

    def get_signature_history(self, sig_id):
        return self.data["signatures"].get(sig_id, {})


# ---------------------------------------------------------------- hypothesis ranking

class HypothesisEngine:
    @staticmethod
    def rank(findings, max_results=7, n=None):
        """Rank by affected samples. No point estimates here: measured points live in
        'Counterfactual Score Impact' (v4.1: the old per-category formulas were removed)."""
        dedup = {}
        for f in findings:
            key = (f.category, f.subcategory, f.description)
            if key not in dedup or f.count > dedup[key].count:
                dedup[key] = f
        filtered = [f for f in dedup.values() if f.count >= MIN_FINDING_COUNT]

        for f in filtered:
            f.severity = round(f.count / n, 4) if n else 0.0
            f.impact_note = (f"{f.count} samples affected ({f.severity:.0%} of n); "
                             "measured points: see Counterfactual Score Impact")
        ranked = sorted(filtered, key=lambda f: (-f.count, f.category, f.description))
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
                 cmp_summary=None, genome_info=None):
    n = ctx.n
    lines = []
    lines.append("# Error Analysis Report (analyze.py v4.1)")
    lines.append("")
    lines.append("> Exploratory diagnosis. Scores come from eval.py; regimes are statistical descriptions, "
                 "not causes. Decide on n50 + n50b, never on one slice.")
    lines.append("")
    lines.append(f"**submission**: `{short_path(getattr(ctx, '_submission_path', 'unknown'))}`  ")
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

    # comparative delta (A12; v4.1: improved/regressed vs gold, v2.2 semantics)
    if cmp_summary:
        cs = cmp_summary
        lines.append("## Comparative Delta")
        lines.append("")
        lines.append(f"New = this submission, old = `{cs['old']}`. Improved = wrong before and right now (vs gold); "
                     "a change from one wrong answer to another wrong answer counts as unchanged. "
                     "Profile fields use exact match per sample.")
        lines.append("")
        lines.append("| Field | Old correct | New correct | Improved | Regressed | Unchanged | Net |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, v in cs["fields"].items():
            lines.append(f"| {name} | {v['old_correct']} | {v['new_correct']} | {v['improved']} | {v['regressed']} | "
                         f"{v['unchanged']} | {v['net']:+d} |")
        lines.append("")
        e = cs["fields"]["emotion"]
        lines.append(f"Net emotion delta: **{e['net']:+d}** samples "
                     f"(accuracy {e['old_correct'] / cs['n']:.3f} -> {e['new_correct'] / cs['n']:.3f}); "
                     f"{cs['emotion_changed']} emotion labels changed in total.")
        if "rouge_l" in cs:
            r = cs["rouge_l"]
            lines.append(f"Per-sample ROUGE-L vs gold: mean {r['mean_old']:.4f} -> {r['mean_new']:.4f} "
                         f"({r['improved']} up, {r['regressed']} down).")
        lines.append("")
        lines.append("_Noise on 50 samples is about +/-7 accuracy points (+/-3-4 samples); decide on n50 + n50b together. "
                     "Per-sample rows: `delta.csv`._")
        lines.append("")

    # v3 taxonomy
    lines.append("## Error Taxonomy")
    lines.append("")
    tax_counts = Counter(s.taxonomy for s in signatures)
    tax_samples = Counter()
    for s_ in signatures:
        tax_samples[s_.taxonomy] += s_.count
    lines.append("| taxonomy | distinct error types | occurrences | description |")
    lines.append("|---|---|---|---|")
    for tax, count in sorted(tax_counts.items(), key=lambda x: (-tax_samples[x[0]], x[0])):
        cat, sub = tax.split(".", 1)
        desc = TAXONOMY.get(cat, {}).get(sub, "")
        lines.append(f"| {tax} | {count} | {tax_samples[tax]} | {desc} |")
    lines.append("")

    # v3 genome (v4.1: shown only when there is an earlier run on the SAME gold file)
    lines.append("## Error Genome")
    lines.append("")
    gi = genome_info or {}
    if gi.get("skipped_other_slice"):
        lines.append(f"**Warning:** {len(gi['skipped_other_slice'])} run(s) in this registry were scored on a different "
                     f"gold file and are ignored: {', '.join(gi['skipped_other_slice'])}.")
        lines.append("")
    if not gi.get("compared_with"):
        lines.append(f"No earlier run on this gold file (fingerprint `{gi.get('gold_fp', '?')}`) in the registry, so "
                     "nothing to compare. To track error types across models, pass the same `--registry` to every run "
                     "on the same slice, e.g. `--registry experiments/.analyze_registry`.")
        lines.append("")
    else:
        lines.append(f"Compared with: {', '.join(gi['compared_with'])} (same gold file `{gi['gold_fp']}`).")
        lines.append("")
        lines.append("| status | count | description |")
        lines.append("|---|---|---|")
        desc = {"novel": "Not seen in earlier runs on this slice",
                "persistent": "Seen before, still present",
                "regressed": "Absent in the last run, back now",
                "resolved": "Present in the last run, absent now"}
        for status in ("novel", "persistent", "regressed", "resolved"):
            lines.append(f"| {status} | {len(genome_result.get(status, []))} | {desc[status]} |")
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

    # v4 regimes (v4.1: significance-tested)
    lines.append("## Error Groups and Regimes")
    lines.append("")
    lines.append(f"_A regime condition is listed only if the group has >= {REGIME_MIN_GROUP} samples, the condition "
                 f"covers >= {REGIME_MIN_SHARE:.0%} of the group, is >= {REGIME_MIN_LIFT}x more frequent than overall, "
                 f"appears in <= {REGIME_MAX_BASE:.0%} of all samples, AND passes a one-sided hypergeometric test "
                 f"(Bonferroni-corrected p < {REGIME_MAX_P}). On a 50-sample slice almost no group is large enough: "
                 "that is expected, not a bug. Regimes describe, they do not explain._")
    lines.append("")
    if not clusters:
        lines.append("No error group with >= 3 samples.")
    else:
        lines.append("| error | size | regime (share / overall, lift, p_adj) |")
        lines.append("|---|---|---|")
        for c in clusters:
            if not c.get("regime_tested"):
                reg = f"not tested (< {REGIME_MIN_GROUP} samples)"
            else:
                reg = "; ".join(f"{r['condition']} ({r['share']:.0%} vs {r['base']:.0%}, x{r['lift']}, p={r['p_adj']})"
                                for r in c["regime"]) or "none significant (spread out)"
            lines.append(f"| {c['taxonomy']} {c['gold']}{('->' + c['pred']) if c['pred'] else ''} | {c['count']} | {reg} |")
    lines.append("")

    # v4.1 measured counterfactuals, three separate tables
    fmt = lambda v: "-" if v is None else f"{v:+.2f}"
    cf_label = lambda c: f"{c['taxonomy']} {c['gold']}{('->' + c['pred']) if c['pred'] else ''}"
    def prof_ba(im):
        return (f"F1 {im['before']['micro_f1']:.3f}->{im['after']['micro_f1']:.3f}, "
                f"exact {im['before']['exact_match']:.3f}->{im['after']['exact_match']:.3f}")
    by_type = {t: [cf for cf in counterfactuals if cf["impact"]["type"] == t] for t in CF_TYPES}

    lines.append("## Counterfactual Score Impact (measured)")
    lines.append("")
    lines.append("_Every number below is a re-score with eval.py after applying the change to the predictions. "
                 "Profile points assume the official profile score = mean of the three per-field metrics x 15 "
                 "(5 pts per field); they are shown for micro-F1 and for exact match because the organisers have "
                 "not confirmed which one is used. The three tables are NOT comparable with each other._")
    lines.append("")

    lines.append("### A. Achievable now: post-filters")
    lines.append("")
    lines.append("_Exactly what dropping the label in `normalize_prediction` would do on this slice, true positives lost included._")
    lines.append("")
    if not by_type["filter"]:
        lines.append("No over-predicted profile label with >= 3 false positives.")
    else:
        lines.append("| error | filter | before -> after | pts if F1 | pts if exact | action |")
        lines.append("|---|---|---|---|---|---|")
        for cf in by_type["filter"]:
            c, im = cf["cluster"], cf["impact"]
            lines.append(f"| {cf_label(c)} ({c['count']}) | {im['kind']} | {prof_ba(im)} | {fmt(im['points'])} | "
                         f"{fmt(im['points_exact'])} | {im['action']} |")
    lines.append("")

    lines.append("### B. Upper bounds: oracle (not achievable by a rule)")
    lines.append("")
    lines.append("_What a PERFECT fix of the group would give. Use it to see where training can win the most, "
                 "never as an expected gain._")
    lines.append("")
    if not by_type["oracle"]:
        lines.append("None.")
    else:
        lines.append("| error | upper bound | before -> after | pts (emotion, or profile if F1) | pts if exact |")
        lines.append("|---|---|---|---|---|")
        for cf in by_type["oracle"]:
            c, im = cf["cluster"], cf["impact"]
            ba = prof_ba(im) if isinstance(im["before"], dict) else f"{im['before']}->{im['after']}"
            lines.append(f"| {cf_label(c)} ({c['count']}) | {im['kind']} | {ba} | {fmt(im['points'])} | {fmt(im['points_exact'])} |")
    lines.append("")

    lines.append("### C. Reply groups (no points)")
    lines.append("")
    lines.append("_The official reply composite (BLEU/ROUGE/BERTScore weights) is unknown, so no points. "
                 "The columns compare the mean per-sample ROUGE-L inside the group with all other samples; "
                 "they are not before/after._")
    lines.append("")
    if not by_type["reply"]:
        lines.append("None.")
    else:
        lines.append("| error | size | ROUGE-L in group | ROUGE-L rest |")
        lines.append("|---|---|---|---|")
        for cf in by_type["reply"]:
            c, im = cf["cluster"], cf["impact"]
            g = "-" if im.get("group_rouge_l") is None else f"{im['group_rouge_l']:.3f}"
            r = "-" if im.get("rest_rouge_l") is None else f"{im['rest_rouge_l']:.3f}"
            lines.append(f"| {cf_label(c)} | {c['count']} | {g} | {r} |")
    lines.append("")

    actionable = [cf for cf in by_type["filter"]
                  if (cf["impact"]["points"] or 0) > 0 and (cf["impact"]["points_exact"] or 0) >= 0]
    lines.append("## Recommended Next Experiment")
    lines.append("")
    if actionable:
        best = actionable[0]
        im = best["impact"]
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
                        clusters, counterfactuals, run_label, experiment_notes,
                        cmp_summary=None, genome_info=None):
    out = {
        "schema_version": "4.1",
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
            **(genome_info or {}),
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
        "compare": cmp_summary,
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

def build_delta(gold, gold_order, sub, compare_sub, skip_rq):
    """Per-sample A/B against GOLD. delta = new_correct - old_correct (+1 improved, -1 regressed).

    v4.0 counted any change as a delta; a change from one wrong label to another wrong
    label is 'unchanged' here (v2.2 semantics restored).
    """
    rows = []
    for rid in gold_order:
        g = gold[rid]
        g_prof = g.get("user_profile") if isinstance(g.get("user_profile"), dict) else {}
        new, old = sub[rid], compare_sub[rid]
        n_prof = new.get("user_profile") if isinstance(new.get("user_profile"), dict) else {}
        o_prof = old.get("user_profile") if isinstance(old.get("user_profile"), dict) else {}
        row = {"id": rid}
        eo, en = old.get("emotion_label"), new.get("emotion_label")
        row["emotion_gold"] = g.get("emotion_label")
        row["emotion_old"] = eo if eo in ev.EMOTIONS else "-"
        row["emotion_new"] = en if en in ev.EMOTIONS else "-"
        row["emotion_old_correct"] = int(eo == g.get("emotion_label"))
        row["emotion_new_correct"] = int(en == g.get("emotion_label"))
        row["emotion_delta"] = row["emotion_new_correct"] - row["emotion_old_correct"]
        row["emotion_changed"] = int(eo != en)
        for f in FIELDS:
            short = FIELD_SHORT[f]
            gs = ev.to_label_set(g_prof.get(f))
            os_, ns = ev.to_label_set(o_prof.get(f)), ev.to_label_set(n_prof.get(f))
            row[f"{short}_gold"] = join_labels(gs)
            row[f"{short}_old"] = join_labels(os_)
            row[f"{short}_new"] = join_labels(ns)
            row[f"{short}_old_exact"] = int(os_ == gs)
            row[f"{short}_new_exact"] = int(ns == gs)
            row[f"{short}_delta"] = row[f"{short}_new_exact"] - row[f"{short}_old_exact"]
        if not skip_rq:
            gt = g.get("response_text") or ""
            row["rouge_l_old"] = round(ev.rouge_l_macro([old.get("response_text") or ""], [gt]), 3)
            row["rouge_l_new"] = round(ev.rouge_l_macro([new.get("response_text") or ""], [gt]), 3)
            row["rouge_l_delta"] = round(row["rouge_l_new"] - row["rouge_l_old"], 3)
        else:
            row["rouge_l_old"] = row["rouge_l_new"] = row["rouge_l_delta"] = ""
        rows.append(row)
    return rows


DELTA_COLUMNS = (["id", "emotion_gold", "emotion_old", "emotion_new", "emotion_old_correct",
                  "emotion_new_correct", "emotion_delta", "emotion_changed"]
                 + [f"{FIELD_SHORT[f]}_{x}" for f in FIELDS
                    for x in ("gold", "old", "new", "old_exact", "new_exact", "delta")]
                 + ["rouge_l_old", "rouge_l_new", "rouge_l_delta"])


def compare_summary(delta_rows):
    """Improved / regressed / unchanged per field, vs gold."""
    n = len(delta_rows)
    out = {"n": n, "fields": {}}
    keys = [("emotion", "emotion")] + [(f, FIELD_SHORT[f]) for f in FIELDS]
    for name, short in keys:
        d = [r[f"{short}_delta"] for r in delta_rows]
        old_ok = sum(r[f"{short}_old_correct" if short == "emotion" else f"{short}_old_exact"] for r in delta_rows)
        new_ok = sum(r[f"{short}_new_correct" if short == "emotion" else f"{short}_new_exact"] for r in delta_rows)
        imp, reg = d.count(1), d.count(-1)
        out["fields"][name] = {"improved": imp, "regressed": reg, "unchanged": n - imp - reg,
                               "net": imp - reg, "old_correct": old_ok, "new_correct": new_ok,
                               "metric": "accuracy" if name == "emotion" else "exact match"}
    out["emotion_changed"] = sum(r["emotion_changed"] for r in delta_rows)
    rl = [r["rouge_l_delta"] for r in delta_rows if r["rouge_l_delta"] != ""]
    if rl:
        out["rouge_l"] = {"improved": sum(1 for x in rl if x > 0), "regressed": sum(1 for x in rl if x < 0),
                          "mean_old": round(sum(r["rouge_l_old"] for r in delta_rows) / n, 4),
                          "mean_new": round(sum(r["rouge_l_new"] for r in delta_rows) / n, 4)}
    return out


def write_delta_csv(outdir, delta_rows):
    delta_path = os.path.join(outdir, "delta.csv")
    with open(delta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DELTA_COLUMNS)
        writer.writeheader()
        writer.writerows(delta_rows)
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
                    help="Write only the rows with a mistake to mistakes.csv (report.md and analysis.json still cover all samples)")
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

    hypotheses = HypothesisEngine.rank(findings, n=len(gold_order))

    signatures = SignatureBuilder.build_all(ctx, rows)

    gold_fp = file_fingerprint(args.gold)
    genome = Genome(registry)
    genome_result = genome.update(run_label, signatures, args.experiment_notes, gold_fp=gold_fp, n=len(gold_order))
    genome.save()
    genome_info = {"registry": registry, "gold_fp": gold_fp,
                   "compared_with": genome.comparable_runs, "skipped_other_slice": genome.skipped_runs}
    if genome.skipped_runs:
        print(f"WARNING: genome skipped {len(genome.skipped_runs)} run(s) scored on a different gold file: "
              f"{', '.join(genome.skipped_runs)}", file=sys.stderr)

    # per-sample ROUGE-L first: reply-group comparisons need it
    fill_rouge(rows, args.skip_response_quality, sub, gold)
    ctx.rows = rows
    ctx.diagnostics = {
        "prediction_distribution": DiagnosticTables.prediction_distribution(rows),
        "emotion_by_condition": DiagnosticTables.emotion_by_condition(ctx, rows),
        "bidirectional": DiagnosticTables.bidirectional_confusions(rows),
    }

    clusters = RegimeAnalyzer.build_clusters(signatures, ctx)
    all_cfs = [{"cluster": c, "impact": CounterfactualAnalyzer.compute_impact(c, ctx)} for c in clusters]
    sort_key = lambda cf: (CounterfactualAnalyzer.rank_key(cf), -cf["cluster"]["count"],
                           cf["cluster"]["taxonomy"], cf["cluster"]["gold"], cf["cluster"]["pred"])
    # v4.1: achievable filters, oracle upper bounds and reply groups are ranked SEPARATELY
    counterfactuals = []
    for t in CF_TYPES:
        counterfactuals += sorted([cf for cf in all_cfs if cf["impact"]["type"] == t], key=sort_key)[:CF_PER_TYPE]
    # groups with a significant regime first, then by size
    clusters = sorted(clusters, key=lambda c: (not c["regime"], -c["count"], c["taxonomy"], c["gold"], c["pred"]))[:MAX_GROUPS_SHOWN]

    write_mistakes_csv(outdir, rows, args.skip_response_quality, sub, gold,
                       mistakes_only=args.mistakes_only)

    cmp_summary = None
    if args.compare_path and compare_sub:
        delta_rows = build_delta(gold, gold_order, sub, compare_sub, args.skip_response_quality)
        write_delta_csv(outdir, delta_rows)
        cmp_summary = compare_summary(delta_rows)
        cmp_summary["old"] = short_path(args.compare_path)

    write_report(outdir, ctx, hypotheses, rows, signatures, genome_result,
                 clusters, counterfactuals, run_label, args.experiment_notes,
                 cmp_summary, genome_info)
    write_analysis_json(outdir, ctx, hypotheses, rows, signatures, genome_result,
                        clusters, counterfactuals, run_label, args.experiment_notes,
                        cmp_summary, genome_info)


if __name__ == "__main__":
    main()
