#!/usr/bin/env python3
"""
scripts/audit_labels.py -- Data Label Auditor for the 17 error patterns.

Queries train.jsonl for keyword patterns and reports whether errors are
"model wrong", "gold arguable", or "gold wrong".

Usage:
    python scripts/audit_labels.py data/train_public.jsonl --query "怕" --and "轻松"
    python scripts/audit_labels.py data/train_public.jsonl --query "考试" --or "论文"

Output:
  - Emotion distribution for matched messages
  - Most common profile tags for matched messages
  - Suggested verdict: model wrong / gold arguable / gold wrong
"""

import argparse
import json
from collections import Counter


def load_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def matches_query(text, queries, mode="and"):
    if not text:
        return False
    text = str(text)
    if mode == "and":
        return all(q in text for q in queries)
    return any(q in text for q in queries)


def audit(path, queries, mode="and", limit=None):
    matched = []
    for i, rec in enumerate(load_jsonl(path)):
        if limit and i >= limit:
            break
        text = rec.get("text", "") or rec.get("message", "") or ""
        if matches_query(text, queries, mode):
            matched.append(rec)

    if not matched:
        print(f"No matches for {' & '.join(queries) if mode == 'and' else ' | '.join(queries)}")
        return

    print(f"\nMatched {len(matched)} records ({len(matched) / (i+1) * 100:.1f}% of scanned)")
    print(f"Query: {' & '.join(queries) if mode == 'and' else ' | '.join(queries)}")
    print("=" * 60)

    emotions = Counter(r.get("emotion_label", "UNKNOWN") for r in matched)
    print("\nEmotion distribution:")
    for emo, count in emotions.most_common():
        pct = count / len(matched) * 100
        print(f"  {emo:<16} {count:>4} ({pct:>5.1f}%)")

    print("\nProfile tags (most common):")
    for field in ["personality_traits", "interests", "style"]:
        tags = Counter()
        for r in matched:
            prof = r.get("user_profile", {})
            if isinstance(prof, dict):
                tags.update(prof.get(field, []))
        if tags:
            print(f"  {field}:")
            for tag, count in tags.most_common(5):
                print(f"    {tag:<30} {count:>3}")

    print("\nVerdict suggestion:")
    if len(emotions) == 1:
        emo = list(emotions.keys())[0]
        print(f"  All matched records are labeled '{emo}'.")
        print(f"  If the model predicts something else, the model is likely wrong.")
    elif emotions.most_common(1)[0][1] / len(matched) > 0.7:
        emo = emotions.most_common(1)[0][0]
        print(f"  {emotions.most_common(1)[0][1] / len(matched) * 100:.0f}% are '{emo}'.")
        print(f"  Gold is probably correct; model errors on this pattern are 'model wrong'.")
    else:
        print(f"  Emotions are mixed. Gold labels may be arguable for this pattern.")
        print(f"  Review individual cases to decide.")

    print("\nSample messages (first 3):")
    for r in matched[:3]:
        text = r.get("text", "") or r.get("message", "")
        emo = r.get("emotion_label", "?")
        print(f"  [{emo}] {text[:100]}...")


def main():
    ap = argparse.ArgumentParser(description="Audit training data labels for keyword patterns")
    ap.add_argument("train_path", help="Path to train.jsonl")
    ap.add_argument("--query", nargs="+", required=True, help="Keywords to search for")
    ap.add_argument("--mode", choices=["and", "or"], default="and", help="Match mode")
    ap.add_argument("--limit", type=int, default=None, help="Max records to scan")
    args = ap.parse_args()

    audit(args.train_path, args.query, args.mode, args.limit)


if __name__ == "__main__":
    main()
