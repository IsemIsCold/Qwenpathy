#!/usr/bin/env python3
"""
B2: Convert val_public.jsonl → test-format input + gold labels.

Usage:
    python scripts/convert_val.py \
        --in data/val_public.jsonl \
        --out data/val_input.jsonl \
        --gold data/val_gold.jsonl \
        [--n 100]

    # a second, independent eval slice (n50b): random points from conversations
    # NOT used in an existing gold file
    python scripts/convert_val.py --in data/val_public.jsonl \
        --out data/val_input_n50b.jsonl --gold data/val_gold_n50b.jsonl \
        --exclude data/val_gold_n50.jsonl --sample 50 --seed 42

Outputs:
    val_input.jsonl  – official test format {id, conversation_id, target_user_turn_id, history}
    val_gold.jsonl   – answer key {id, response_text, emotion_label, user_profile, memory_refs}

History is truncated to end at target_user_turn_id (the last user message before
we must predict the assistant reply).
"""

import argparse
import json
import random
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
    parser.add_argument(
        "--exclude", default=None,
        help="Gold/input jsonl whose conversations must be left out (e.g. data/val_gold_n50.jsonl)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N prediction points (after --exclude) instead of taking the first N",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample (default 42)")
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


def excluded_conversations(path: Path | None) -> set[str]:
    """Conversation ids used in an existing gold/input file (ids look like <conv_id>_t<turn>)."""
    if path is None:
        return set()
    if not path.exists():
        print(f"ERROR: --exclude file not found: {path}", file=sys.stderr)
        sys.exit(1)
    convs = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                point_id = json.loads(line)["id"]
                convs.add(point_id.rsplit("_t", 1)[0])
    return convs


def convert_val(input_path: Path, out_path: Path, gold_path: Path, n: int | None,
                exclude: set[str] | None = None, sample: int | None = None, seed: int = 42) -> dict:
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
    exclude = exclude or set()
    skipped_conversations = 0
    if sample is not None:
        n = None  # sampling needs every eligible point first
    pairs = []  # (test_input, gold_label)

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
            if conv_id in exclude:
                skipped_conversations += 1
                continue

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
                pairs.append((test_input, None))

                # --- gold label ---
                gold_label = {
                    "id": point_id,
                    "response_text": target["response_text"],
                    "emotion_label": target["emotion_label"],
                    "user_profile": target["user_profile"],
                    "memory_refs": target["memory_refs"],
                }
                pairs[-1] = (test_input, gold_label)

                written_points += 1

            if n is not None and written_points >= n:
                break

        if sample is not None:
            if sample > len(pairs):
                print(f"ERROR: --sample {sample} > {len(pairs)} eligible points", file=sys.stderr)
                sys.exit(1)
            rng = random.Random(seed)
            keep = sorted(rng.sample(range(len(pairs)), sample))  # keep file order
            pairs = [pairs[i] for i in keep]
            written_points = len(pairs)
        for test_input, gold_label in pairs:
            fout.write(json.dumps(test_input, ensure_ascii=False) + "\n")
            fgold.write(json.dumps(gold_label, ensure_ascii=False) + "\n")

    stats = {
        "conversations_excluded": skipped_conversations,
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
        exclude=excluded_conversations(Path(args.exclude) if args.exclude else None),
        sample=args.sample,
        seed=args.seed,
    )

    print("=" * 50)
    print("convert_val.py – Done")
    print("=" * 50)
    print(f"Conversations read:      {stats['conversations_read']:,}")
    print(f"Conversations excluded:  {stats['conversations_excluded']:,}")
    print(f"Prediction points found: {stats['prediction_points_found']:,}")
    print(f"Prediction points written: {stats['prediction_points_written']:,}")
    print(f"Test input file:         {stats['val_input_path']}")
    print(f"Gold labels file:        {stats['val_gold_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
