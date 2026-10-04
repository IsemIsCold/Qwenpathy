#!/usr/bin/env python3
"""
B6: Compute dataset statistics and emit docs/data-stats.md.

Usage:
    python scripts/compute_stats.py \
        --train data/train_public.jsonl \
        --val data/val_public.jsonl \
        --out docs/data-stats.md
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Try to import transformers for tokenization; if unavailable, skip token counts.
try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except Exception:
    HAS_TOKENIZER = False

# Emotion vocabulary (fixed — matches official spec and data)
EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
]

# Profile labels are DISCOVERED DYNAMICALLY from the data itself.
# Earlier documentation gave incorrect vocabularies; the data is the ground truth.
PROFILE_KEYS = ["personality_traits", "interests", "style"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset statistics.")
    parser.add_argument("--train", default="data/train_public.jsonl",
                        help="Training data path")
    parser.add_argument("--val", default="data/val_public.jsonl",
                        help="Validation data path")
    parser.add_argument("--out", default="docs/data-stats.md",
                        help="Output markdown file")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Tokenizer name or path for token counting")
    return parser.parse_args()


def analyze_file(path: Path, tokenizer=None) -> dict:
    """Analyze one JSONL file and return a stats dictionary."""
    if not path.exists():
        return {"error": f"File not found: {path}"}

    stats = {
        "conversations": 0,
        "prediction_points": 0,
        "emotions": Counter(),
        "profile_empty": 0,
        "profile_labels": {k: Counter() for k in PROFILE_KEYS},
        "turns_per_conv": [],
        "history_chars": [],
        "target_reply_chars": [],
        "target_reply_tokens": [] if tokenizer else None,
        "history_tokens": [] if tokenizer else None,
    }

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            stats["conversations"] += 1
            stats["turns_per_conv"].append(len(conv.get("turns", [])))

            for pp in conv.get("prediction_points", []):
                stats["prediction_points"] += 1
                tgt = pp["target"]

                # Emotion
                stats["emotions"][tgt["emotion_label"]] += 1

                # Profile
                prof = tgt.get("user_profile", {})
                is_empty = True
                for key in PROFILE_KEYS:
                    labels = prof.get(key, [])
                    if labels:
                        is_empty = False
                    for lab in labels:
                        stats["profile_labels"][key][lab] += 1
                if is_empty:
                    stats["profile_empty"] += 1

                # Reply length
                reply = tgt["response_text"]
                stats["target_reply_chars"].append(len(reply))
                if tokenizer:
                    stats["target_reply_tokens"].append(len(tokenizer.encode(reply)))

                # History length (all turns up to target_user_turn_id)
                target_turn_id = pp["target_user_turn_id"]
                history_text = ""
                for t in conv.get("turns", []):
                    if t["turn_id"] <= target_turn_id:
                        history_text += t["content"] + "\n"
                stats["history_chars"].append(len(history_text))
                if tokenizer:
                    stats["history_tokens"].append(len(tokenizer.encode(history_text)))

    return stats


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    if f == c:
        return float(s[f])
    return s[f] * (c - k) + s[c] * (k - f)


def fmt_percent(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{n/total*100:.1f}%"


def build_markdown(train_stats: dict, val_stats: dict, tokenizer_name: str) -> str:
    lines = [
        "# Dataset Statistics",
        "",
        "> **Generated:** 2026-10-04",
        "> **Tokenizer:** " + (tokenizer_name if tokenizer_name else "None (install transformers for token counts)"),
        "",
        "> ⚠️ **CRITICAL:** Profile label vocabularies were discovered by scanning the gold data itself. They **differ** from the official README documentation. See Section 5.2 for the actual labels. `validate.py` must **warn, not reject**, on out-of-vocab profile values until the official vocabulary is confirmed.",
    ]
    if train_stats.get("error") and val_stats.get("error"):
        lines.append(
            "> **WARNING:** Both train and val files were missing — this report contains placeholders."
        )
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Split | Conversations | Prediction Points | Avg Turns / Conv |")
    lines.append("|-------|--------------:|------------------:|-----------------:|")

    for split_name, stats in [("Train", train_stats), ("Val", val_stats)]:
        if "error" in stats:
            lines.append(f"| {split_name} | — | — | — |")
            continue
        avg_turns = sum(stats["turns_per_conv"]) / len(stats["turns_per_conv"]) if stats["turns_per_conv"] else 0
        lines.append(
            f"| {split_name} | {stats['conversations']:,} | {stats['prediction_points']:,} | {avg_turns:.1f} |"
        )
    lines.append("")

    # Emotion distribution
    lines.append("## 2. Emotion Distribution")
    lines.append("")
    lines.append("| Emotion | Train Count | Train % | Val Count | Val % |")
    lines.append("|---------|------------:|--------:|----------:|------:|")

    train_total = train_stats.get("prediction_points", 1)
    val_total = val_stats.get("prediction_points", 1)

    for emo in EMOTIONS:
        tc = train_stats["emotions"].get(emo, 0) if "emotions" in train_stats else 0
        vc = val_stats["emotions"].get(emo, 0) if "emotions" in val_stats else 0
        lines.append(
            f"| {emo} | {tc:,} | {fmt_percent(tc, train_total)} | {vc:,} | {fmt_percent(vc, val_total)} |"
        )
    lines.append("")

    # Profile stats
    lines.append("## 3. User Profile Statistics")
    lines.append("")

    for split_name, stats in [("Train", train_stats), ("Val", val_stats)]:
        if "error" in stats:
            continue
        total_pts = stats["prediction_points"]
        empty = stats["profile_empty"]
        lines.append(f"### 3.1 {split_name} – Empty Profile Rate")
        lines.append("")
        lines.append(f"- **Empty profiles:** {empty:,} / {total_pts:,} ({fmt_percent(empty, total_pts)})")
        lines.append("")
        lines.append(f"### 3.2 {split_name} – Profile Label Frequencies")
        lines.append("")

        for key in PROFILE_KEYS:
            lines.append(f"#### {key}")
            lines.append("")
            lines.append("| Label | Count | % of points |")
            lines.append("|-------|------:|------------:|")
            counter = stats["profile_labels"].get(key, Counter())
            for label, count in counter.most_common():
                lines.append(f"| {label} | {count:,} | {fmt_percent(count, total_pts)} |")
            if not counter:
                lines.append("| — | — | — |")
            lines.append("")

    # Length statistics
    lines.append("## 4. Length Statistics")
    lines.append("")
    lines.append("### 4.1 Characters")
    lines.append("")
    lines.append("| Split | Metric | History (chars) | Target Reply (chars) |")
    lines.append("|-------|--------|----------------:|---------------------:|")

    for split_name, stats in [("Train", train_stats), ("Val", val_stats)]:
        if "error" in stats:
            continue
        hist = stats.get("history_chars", [])
        reply = stats.get("target_reply_chars", [])
        # Compute mean properly using sum/len
        hist_mean = sum(hist) / len(hist) if hist else 0
        reply_mean = sum(reply) / len(reply) if reply else 0
        lines.append(f"| {split_name} | Mean | {hist_mean:,.0f} | {reply_mean:,.0f} |")
        for metric, p in [("Median (P50)", 50), ("P90", 90), ("P99", 99), ("Max", 100)]:
            lines.append(
                f"| {split_name} | {metric} | {percentile(hist, p):,.0f} | {percentile(reply, p):,.0f} |"
            )
        lines.append("")

    # Token statistics — use 1.4 chars/token heuristic when no real tokenizer
    lines.append("### 4.2 Approximate Tokens (1.4 chars/token for Chinese)")
    lines.append("")
    lines.append("> **Note:** System prompt adds ~500+ tokens on top of history. "
               "A 4K context budget leaves ~3.5K for history after system prompt."
    )
    lines.append("")
    lines.append("| Split | Metric | History (tokens) | Target Reply (tokens) |")
    lines.append("|-------|--------|-----------------:|----------------------:|")

    for split_name, stats in [("Train", train_stats), ("Val", val_stats)]:
        if "error" in stats:
            continue
        hist_chars = stats.get("history_chars", [])
        reply_chars = stats.get("target_reply_chars", [])
        # Approximate tokens at 1.4 chars/token (Chinese text)
        hist_tok = [h / 1.4 for h in hist_chars]
        reply_tok = [r / 1.4 for r in reply_chars]
        hist_mean_tok = sum(hist_tok) / len(hist_tok) if hist_tok else 0
        reply_mean_tok = sum(reply_tok) / len(reply_tok) if reply_tok else 0
        lines.append(f"| {split_name} | Mean | {hist_mean_tok:,.0f} | {reply_mean_tok:,.0f} |")
        for metric, p in [("Median (P50)", 50), ("P90", 90), ("P99", 99), ("Max", 100)]:
            lines.append(
                f"| {split_name} | {metric} | {percentile(hist_tok, p):,.0f} | {percentile(reply_tok, p):,.0f} |"
            )
        lines.append("")

    # Threshold analysis
    lines.append("### 4.3 Context-Length Thresholds")
    lines.append("")
    lines.append("How many prediction points exceed common context limits? "
               "(History tokens only; add ~500 tokens for system prompt.)")
    lines.append("")
    lines.append("| Split | Threshold | History > threshold | % |")
    lines.append("|-------|-----------|--------------------:|--:|")

    for split_name, stats in [("Train", train_stats), ("Val", val_stats)]:
        if "error" in stats:
            continue
        hist_chars = stats.get("history_chars", [])
        hist_tok = [h / 1.4 for h in hist_chars]
        total = len(hist_tok)
        for thresh in [1024, 2048, 4096, 8192]:
            over = sum(1 for h in hist_tok if h > thresh)
            lines.append(
                f"| {split_name} | {thresh} | {over:,} | {fmt_percent(over, total)} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("*End of report*")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    tokenizer = None
    if HAS_TOKENIZER:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer, trust_remote_code=True, local_files_only=True
            )
            print(f"Loaded tokenizer: {args.tokenizer}")
        except Exception as e:
            print(f"Tokenizer load failed ({e}); token counts will be omitted.")
            tokenizer = None
    else:
        print("transformers not installed; token counts will be omitted.")

    train_stats = analyze_file(Path(args.train), tokenizer)
    val_stats = analyze_file(Path(args.val), tokenizer)

    md = build_markdown(train_stats, val_stats, args.tokenizer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nReport written to: {out_path}")
    if "error" in train_stats:
        print(f"  Train: {train_stats['error']}")
    if "error" in val_stats:
        print(f"  Val:   {val_stats['error']}")


if __name__ == "__main__":
    main()
