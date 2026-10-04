#!/usr/bin/env python3
"""
B2: Convert val_public.jsonl → test-format input + gold labels.

Usage:
    python scripts/convert_val.py \
        --in data/val_public.jsonl \
        --out data/val_input.jsonl \
        --gold data/val_gold.jsonl \
        [--n 100]

Outputs:
    val_input.jsonl  – official test format {id, conversation_id, target_user_turn_id, history}
    val_gold.jsonl   – answer key {id, response_text, emotion_label, user_profile, memory_refs}

History is truncated to end at target_user_turn_id (the last user message before
we must predict the assistant reply).
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert validation set to test input + gold labels."
    )
    parser.add_argument(
        "--in", "-i", dest="input_file",
        default="data/val_public.jsonl",
        help="Path to val_public.jsonl (default: data/val_public.jsonl)",
    )
    parser.add_argument(
        "--out", "-o", dest="output_file",
        default="data/val_input.jsonl",
        help="Path for test-format output (default: data/val_input.jsonl)",
    )
    parser.add_argument(
        "--gold", "-g", dest="gold_file",
        default="data/val_gold.jsonl",
        help="Path for gold-label output (default: data/val_gold.jsonl)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Only process first N prediction points (default: all)",
    )
    return parser.parse_args()


def build_history_up_to_turn(turns: list[dict], target_turn_id: int) -> list[dict]:
    """
    Return all turns with turn_id <= target_turn_id.

    Each turn is stripped to the official test schema:
        {turn_id, role, content}
    """
    return [
        {"turn_id": t["turn_id"], "role": t["role"], "content": t["content"]}
        for t in turns
        if t["turn_id"] <= target_turn_id
    ]


def make_point_id(conversation_id: str, target_user_turn_id: int) -> str:
    """
    Create a unique point ID matching the official naming convention.

    Example: train_conv_000001 + turn 9 → "train_conv_000001_t9"
    """
    return f"{conversation_id}_t{target_user_turn_id}"


def convert_val(input_path: Path, out_path: Path, gold_path: Path, n: int | None) -> dict:
    """
    Main conversion routine.

    Returns a stats dict with counts and file paths for downstream logging.
    """
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    total_conversations = 0
    total_points = 0
    written_points = 0

    with (
        input_path.open("r", encoding="utf-8") as fin,
        out_path.open("w", encoding="utf-8") as fout,
        gold_path.open("w", encoding="utf-8") as fgold,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            conv = json.loads(line)
            total_conversations += 1
            conv_id = conv["conversation_id"]
            turns = conv["turns"]

            for pp in conv.get("prediction_points", []):
                total_points += 1

                if n is not None and written_points >= n:
                    break

                target_turn_id = pp["target_user_turn_id"]
                target = pp["target"]
                point_id = make_point_id(conv_id, target_turn_id)

                # --- test-format input ---
                test_input = {
                    "id": point_id,
                    "conversation_id": conv_id,
                    "target_user_turn_id": target_turn_id,
                    "history": build_history_up_to_turn(turns, target_turn_id),
                }
                fout.write(json.dumps(test_input, ensure_ascii=False) + "\n")

                # --- gold label ---
                gold_label = {
                    "id": point_id,
                    "response_text": target["response_text"],
                    "emotion_label": target["emotion_label"],
                    "user_profile": target["user_profile"],
                    "memory_refs": target["memory_refs"],
                }
                fgold.write(json.dumps(gold_label, ensure_ascii=False) + "\n")

                written_points += 1

            if n is not None and written_points >= n:
                break

    stats = {
        "conversations_read": total_conversations,
        "prediction_points_found": total_points,
        "prediction_points_written": written_points,
        "val_input_path": str(out_path),
        "val_gold_path": str(gold_path),
    }
    return stats


def main() -> None:
    args = parse_args()
    stats = convert_val(
        input_path=Path(args.input_file),
        out_path=Path(args.output_file),
        gold_path=Path(args.gold_file),
        n=args.n,
    )

    print("=" * 50)
    print("convert_val.py – Done")
    print("=" * 50)
    print(f"Conversations read:      {stats['conversations_read']:,}")
    print(f"Prediction points found: {stats['prediction_points_found']:,}")
    print(f"Prediction points written: {stats['prediction_points_written']:,}")
    print(f"Test input file:         {stats['val_input_path']}")
    print(f"Gold labels file:        {stats['val_gold_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
