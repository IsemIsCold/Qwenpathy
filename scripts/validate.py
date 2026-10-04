#!/usr/bin/env python3
r"""
scripts/validate.py -- B3: strict submission validator (fail-closed gate).

Usage:
    python scripts/validate.py outputs/submission.jsonl --input data/val_input.jsonl

Rules (see docs/eval-design.md):
  HARD FAIL  - malformed JSON, non-object lines, missing/extra top-level fields,
               empty response_text, emotion_label not in the official 16,
               user_profile values outside the official vocabulary
               (confirmed via participant/templates/submission_schema.json;
               see PROFILE_OOV_HARD_FAIL to revert to warn-only),
               malformed memory_refs (must match mem_\d{6}),
               ID misalignment (missing / duplicate / extra / count mismatch)
  WARN ONLY  - prediction order differs from input order (official evaluator
               most likely matches by ID)

Tolerated without failure (documented contract, not accidental):
  blank lines, CRLF line endings, UTF-8 BOM (stripped, counted as warning),
  duplicate labels inside one profile list or one memory_refs list
  (set semantics downstream; not prohibited by the official contract),
  control characters inside strings (the official contract does not prohibit
  them; the val gold itself contains none, so this is theoretical).

Exit code 0 = PASS, 1 = FAIL, 2 = cannot read input file(s) or the input file
itself fails integrity checks (empty, duplicate input ids).
"""

import argparse
import json
import re
import sys

EMOTIONS = frozenset({
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
})

PROFILE_VOCAB = {
    "personality_traits": frozenset({
        "sensitive", "conservative", "high_conscientiousness", "assertive",
        "agreeable", "introverted", "emotionally_stable", "casual", "open",
        "extroverted",
    }),
    "interests": frozenset({
        "career_development", "pets", "study_exam", "sports_fitness", "games",
        "reading_writing", "art_design", "social", "travel_outdoor", "music",
        "film_animation", "programming_technology",
    }),
    "style": frozenset({
        "rational", "direct", "detailed", "brief", "indirect",
        "high_emotional_expression", "colloquial", "formal", "humorous",
        "emoji_user", "low_emotional_expression",
    }),
}

TOP_LEVEL_KEYS = frozenset({
    "id", "response_text", "emotion_label", "user_profile", "memory_refs",
})
PROFILE_KEYS = frozenset(PROFILE_VOCAB.keys())
MEM_RE = re.compile(r"^mem_\d{6}$")

# The official participant package (templates/submission_schema.json) confirms
# the profile vocabulary (team review, Oct 4), so out-of-vocab values are
# HARD REJECTED. Flip to False to return to warn-only if the organisers ever
# clarify that the schema is not enforced by the official evaluator.
PROFILE_OOV_HARD_FAIL = True
MAX_DETAIL = 20  # cap on detailed error lines printed per category


class Counters:
    def __init__(self):
        self.lines = 0
        self.json_errors = 0
        self.structural_errors = 0
        self.bad_labels = 0
        self.mem_errors = 0
        self.missing_ids = 0
        self.extra_ids = 0
        self.duplicate_ids = 0
        self.count_mismatch = 0
        self.order_warnings = 0
        self.profile_warnings = 0
        self.profile_oov_errors = 0
        self.bom_warnings = 0
        self.details = []  # (category, message)

    def err(self, category, message):
        self.details.append((category, message))

    def summary(self):
        return (
            f"{self.lines} lines | {self.json_errors} json errors | "
            f"{self.bad_labels} bad labels | {self.mem_errors} bad memory_refs | "
            f"{self.structural_errors} structural violations | "
            f"{self.missing_ids} missing ids | {self.duplicate_ids} duplicate ids | "
            f"{self.extra_ids} extra ids | "
            f"{self.count_mismatch} count mismatch | "
            f"{self.order_warnings} order warnings | "
            f"{self.profile_oov_errors} bad profile values | "
            f"{self.profile_warnings} profile vocab warnings"
        )


# ---------------------------------------------------------------- loaders

def read_text_strict(path, counters):
    """Read a file as UTF-8, rejecting UTF-16 (the Windows PowerShell trap)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(2)

    sample = raw[:200]
    if b"\x00" in sample:
        print(
            f"ERROR: {path} looks like UTF-16 (null bytes found).\n"
            f"       Do NOT create JSONL with PowerShell '>' -- it writes UTF-16.\n"
            f"       Re-save as UTF-8, e.g. from Python with encoding='utf-8'.",
            file=sys.stderr,
        )
        sys.exit(2)

    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
        counters.bom_warnings += 1

    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        print(f"ERROR: {path} is not valid UTF-8: {exc}", file=sys.stderr)
        sys.exit(2)


def reject_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")


def load_records(path, counters, label):
    """Load a JSONL file into [(line_no, record)]. Counts errors in counters."""
    text = read_text_strict(path, counters)
    records = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        if label == "submission":
            counters.lines += 1
        try:
            obj = json.loads(
                raw_line,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            counters.json_errors += 1
            counters.err("json", f"line {line_no}: invalid JSON: {exc}")
            continue
        if not isinstance(obj, dict):
            counters.json_errors += 1
            counters.err("json", f"line {line_no}: not a JSON object")
            continue
        records.append((line_no, obj))
    return records


# ---------------------------------------------------------------- record checks

def is_nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def check_record(rec, line_no, counters):
    keys = set(rec.keys())
    missing = TOP_LEVEL_KEYS - keys
    extra = keys - TOP_LEVEL_KEYS
    if missing:
        counters.structural_errors += 1
        counters.err("structural", f"line {line_no}: missing fields {sorted(missing)}")
    if extra:
        counters.structural_errors += 1
        counters.err("structural", f"line {line_no}: unexpected fields {sorted(extra)}")

    if "id" in rec and not is_nonempty_string(rec["id"]):
        counters.structural_errors += 1
        counters.err("structural", f"line {line_no}: 'id' must be a non-empty string")

    if "response_text" in rec:
        if not is_nonempty_string(rec["response_text"]):
            counters.structural_errors += 1
            counters.err("structural", f"line {line_no}: 'response_text' must be a non-empty string")

    if "emotion_label" in rec:
        emo = rec["emotion_label"]
        if not isinstance(emo, str) or emo not in EMOTIONS:
            counters.bad_labels += 1
            shown = emo if isinstance(emo, str) else type(emo).__name__
            counters.err("label", f"line {line_no}: invalid emotion_label {shown!r}")

    if "user_profile" in rec:
        prof = rec["user_profile"]
        if not isinstance(prof, dict):
            counters.structural_errors += 1
            counters.err("structural", f"line {line_no}: 'user_profile' must be an object")
        else:
            pkeys = set(prof.keys())
            pmissing = PROFILE_KEYS - pkeys
            pextra = pkeys - PROFILE_KEYS
            if pmissing:
                counters.structural_errors += 1
                counters.err("structural", f"line {line_no}: user_profile missing {sorted(pmissing)}")
            if pextra:
                counters.structural_errors += 1
                counters.err("structural", f"line {line_no}: user_profile unexpected {sorted(pextra)}")
            for field in PROFILE_KEYS:
                if field not in prof:
                    continue
                value = prof[field]
                if not isinstance(value, list):
                    counters.structural_errors += 1
                    counters.err("structural", f"line {line_no}: user_profile.{field} must be a list")
                    continue
                for i, item in enumerate(value):
                    if not isinstance(item, str):
                        counters.structural_errors += 1
                        counters.err("structural", f"line {line_no}: user_profile.{field}[{i}] must be a string")
                    elif item not in PROFILE_VOCAB[field]:
                        if PROFILE_OOV_HARD_FAIL:
                            counters.profile_oov_errors += 1
                            counters.err("profile", f"line {line_no}: {field} value {item!r} not in the official vocabulary")
                        else:
                            counters.profile_warnings += 1
                            counters.err("profile_warn", f"line {line_no}: {field} value {item!r} not in verified vocabulary")

    if "memory_refs" in rec:
        mem = rec["memory_refs"]
        if not isinstance(mem, list):
            counters.mem_errors += 1
            counters.err("memory", f"line {line_no}: 'memory_refs' must be a list")
        else:
            for i, item in enumerate(mem):
                if not isinstance(item, str) or not MEM_RE.match(item):
                    counters.mem_errors += 1
                    counters.err("memory", f"line {line_no}: memory_refs[{i}]={item!r} does not match mem_[0-9]{{6}}")


# ---------------------------------------------------------------- ID alignment

def check_alignment(input_records, sub_records, counters):
    # Only string ids enter set operations; non-string ids were already flagged
    # as structural violations by check_record, so coverage is not lost.
    input_ids = [rec.get("id") for _, rec in input_records if isinstance(rec.get("id"), str)]
    sub_ids = [rec.get("id") for _, rec in sub_records if isinstance(rec.get("id"), str)]

    if len(input_ids) != len(sub_ids):
        counters.count_mismatch = 1
        counters.err("ids", f"record count mismatch: input={len(input_ids)} submission={len(sub_ids)}")

    seen = set()
    for line_no, rec in sub_records:
        rid = rec.get("id")
        if not isinstance(rid, str):
            continue  # non-string ids already flagged as structural violations
        if rid in seen:
            counters.duplicate_ids += 1
            counters.err("ids", f"line {line_no}: duplicate id {rid!r}")
        else:
            seen.add(rid)

    input_set, sub_set = set(input_ids), set(sub_ids)
    missing = input_set - sub_set
    extra = sub_set - input_set
    if missing:
        counters.missing_ids = len(missing)
        preview = sorted(missing)[:3]
        counters.err("ids", f"{len(missing)} input ids have no prediction, e.g. {preview}")
    if extra:
        counters.extra_ids = len(extra)
        preview = sorted(extra)[:3]
        counters.err("ids", f"{len(extra)} prediction ids not in input, e.g. {preview}")

    if not missing and not extra and input_ids != sub_ids:
        counters.order_warnings += 1
        counters.err("order_warn", "prediction order differs from input order (matched by id; not a failure)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Strict submission validator (fail-closed gate).")
    ap.add_argument("submission", help="Path to submission.jsonl")
    ap.add_argument("--input", required=True, help="Path to test-format input JSONL")
    args = ap.parse_args()

    counters = Counters()
    sub_records = load_records(args.submission, counters, "submission")

    input_counters = Counters()
    input_records = load_records(args.input, input_counters, "input")
    if input_counters.json_errors > 0:
        print("ERROR: the --input file itself has JSON errors; fix the data pipeline first.", file=sys.stderr)
        sys.exit(2)
    if not input_records:
        print("ERROR: --input file contains no records.", file=sys.stderr)
        sys.exit(2)
    input_ids_raw = [rec.get("id") for _, rec in input_records]
    if not all(isinstance(i, str) for i in input_ids_raw):
        print("ERROR: --input file has non-string ids; fix the data pipeline (convert_val.py).", file=sys.stderr)
        sys.exit(2)
    if len(input_ids_raw) != len(set(input_ids_raw)):
        print("ERROR: --input file has duplicate ids; fix the data pipeline (convert_val.py).", file=sys.stderr)
        sys.exit(2)

    for line_no, rec in sub_records:
        check_record(rec, line_no, counters)

    check_alignment(input_records, sub_records, counters)

    hard_fail = any([
        counters.json_errors, counters.structural_errors, counters.bad_labels,
        counters.mem_errors, counters.profile_oov_errors,
        counters.missing_ids, counters.extra_ids,
        counters.duplicate_ids, counters.count_mismatch,
    ])

    print()
    print("=" * 64)
    print("VALIDATION FAILED" if hard_fail else "VALIDATION PASSED")
    print("=" * 64)
    print(counters.summary())
    if counters.bom_warnings:
        print(f"note: UTF-8 BOM stripped ({counters.bom_warnings} file(s))")

    shown = 0
    for category, message in counters.details:
        if category == "profile_warn" and not hard_fail:
            tag = "warning"
        elif category == "order_warn":
            tag = "warning"
        else:
            tag = "error"
        if shown < MAX_DETAIL:
            print(f"  [{tag}] {message}")
            shown += 1
        elif shown == MAX_DETAIL:
            remaining = len(counters.details) - MAX_DETAIL
            print(f"  ... and {remaining} more (see summary counts)")
            shown += 1

    print()
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
