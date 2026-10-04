#!/usr/bin/env python3
"""
train/build_sft.py (I1): train_public.jsonl -> SFT data that matches inference exactly.

Design
  * Nothing is copied from the inference script: SYSTEM_PROMPT, EMOTIONS,
    PROFILE_LABELS, conversation_text() and normalize_prediction() are imported
    from the run_inference.py we will submit (--inference). Each example is
    [system, user, assistant]: the two messages run_inference.py sends, plus the
    gold answer.
  * Target = the 4-field JSON the inference parser expects, memory_refs forced
    to [] (no memory bank at test time; the official prompt says so too).
    Gold rows the official parser would blank are skipped.
  * Train/dev split by conversation_id, never by prediction point.
  * Histories longer than --max-user-tokens lose their oldest turns; the target
    user turn is always kept and a window never starts on an assistant turn.
    Inference must apply the same rule.

Emotion order
  The official script builds the label list with ", ".join(EMOTIONS) on a
  Python set, whose order changes between runs. This script uses sorted order;
  the submitted run_inference.py must use ", ".join(sorted(EMOTIONS)) too. The
  script warns if it doesn't.

Qwen3 non-thinking
  vLLM renders the chat with enable_thinking=False, which starts the assistant
  turn with an empty "<think>\\n\\n</think>\\n\\n" block. Train the same way:
    * LLaMA-Factory: a Qwen3 no-think template (qwen3_nothink in recent
      versions), not the Qwen2.5 "qwen" template.
    * TRL / custom loop: use *_text.jsonl (needs --tokenizer), which is rendered
      with the real Qwen3 chat template, and put the loss on "completion" only.

Usage
  python train/build_sft.py --train data/train_public.jsonl \
      --inference participant/run_inference.py --out-dir data/sft \
      [--tokenizer path/to/Qwen3-14B] [--max-user-tokens 3500] \
      [--dev-ratio 0.05] [--seed 42] \
      [--field-order emotion_label,user_profile,response_text,memory_refs]

Outputs (in --out-dir)
  train.jsonl, dev.jsonl                {"id", "messages": [system, user, assistant]}
  train_sharegpt.json, dev_sharegpt.json   LLaMA-Factory sharegpt format
  dataset_info.json                     LLaMA-Factory dataset registration
  train_text.jsonl, dev_text.jsonl      {"id", "prompt", "completion"}  (only with --tokenizer)
  build_report.json                     counts, truncation, emotion distribution
"""

import argparse
import collections
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

FIELDS = ("response_text", "emotion_label", "user_profile", "memory_refs")
CHARS_PER_TOKEN = 1.4  # Chinese text with the Qwen tokenizer, used only without --tokenizer


# ------------------------------------------------------------------ inference script

def load_inference_module(path):
    spec = importlib.util.spec_from_file_location("submitted_run_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("SYSTEM_PROMPT", "EMOTIONS", "PROFILE_LABELS", "conversation_text", "normalize_prediction"):
        if not hasattr(module, name):
            sys.exit(f"{path} has no '{name}'. Is this the official run_inference.py?")
    if '", ".join(EMOTIONS)' in Path(path).read_text(encoding="utf-8"):
        print('WARNING: the inference script uses ", ".join(EMOTIONS), whose order is random per run.\n'
              '         Change it to ", ".join(sorted(EMOTIONS)) so training and inference prompts match.')
    return module


def build_system_prompt(ri):
    return ri.SYSTEM_PROMPT.format(
        emotions=", ".join(sorted(ri.EMOTIONS)),
        personality=", ".join(sorted(ri.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(ri.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(ri.PROFILE_LABELS["style"])),
    )


# ------------------------------------------------------------------ tokens

class TokenCounter:
    def __init__(self, tokenizer_path):
        self.tokenizer = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                sys.exit("--tokenizer needs the 'transformers' package")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            print(f"Token counts: {tokenizer_path}")
        else:
            print(f"Token counts: estimate ({CHARS_PER_TOKEN} chars/token). Pass --tokenizer for exact counts.")

    @property
    def exact(self):
        return self.tokenizer is not None

    def count(self, text):
        if self.tokenizer is None:
            return math.ceil(len(text) / CHARS_PER_TOKEN)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def render(self, messages):
        """Prompt/completion strings exactly as the Qwen3 chat template renders them, thinking off."""
        prompt = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = self.tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        if not full.startswith(prompt):
            raise ValueError("chat template: full conversation does not start with the generation prompt")
        return prompt, full[len(prompt):]


# ------------------------------------------------------------------ examples

def make_sample(conversation, target_turn_id):
    """Test-format input for one prediction point (same keys as the official test file)."""
    turns = sorted(conversation["turns"], key=lambda t: t["turn_id"])
    history = [{"turn_id": t["turn_id"], "role": t["role"], "content": t["content"]}
               for t in turns if t["turn_id"] <= target_turn_id]
    return {
        "id": f"{conversation['conversation_id']}_t{target_turn_id}",
        "conversation_id": conversation["conversation_id"],
        "target_user_turn_id": target_turn_id,
        "history": history,
    }


def fit_history(sample, ri, counter, max_tokens):
    """Drop the oldest turns until the rendered user message fits. Returns (user_text, dropped)."""
    history = sample["history"]
    dropped = 0
    while True:
        user_text = ri.conversation_text({**sample, "history": history})
        if len(history) == 1 or counter.count(user_text) <= max_tokens:
            return user_text, dropped
        cut = 1
        while cut < len(history) - 1 and history[cut]["role"] == "assistant":
            cut += 1  # never start the window on an assistant turn
        history = history[cut:]
        dropped += cut


def build_target(gold, ri, field_order):
    """Gold answer as the JSON string the model must learn to output, or None if the parser would blank it."""
    raw = {
        "response_text": gold["response_text"],
        "emotion_label": gold["emotion_label"],
        "user_profile": gold["user_profile"],
        "memory_refs": [],  # no memory bank at test time
    }
    clean = ri.normalize_prediction("gold", raw)
    if (not clean["response_text"] or clean["emotion_label"] != raw["emotion_label"]
            or clean["user_profile"] != raw["user_profile"]):
        return None
    return json.dumps({key: clean[key] for key in field_order}, ensure_ascii=False)


def split_conversation_ids(conversations, dev_ratio, seed):
    ids = sorted(c["conversation_id"] for c in conversations)
    random.Random(seed).shuffle(ids)
    n_dev = max(1, round(len(ids) * dev_ratio))
    return set(ids[:n_dev])


def build_examples(conversations, ri, counter, args, field_order):
    system_prompt = build_system_prompt(ri)
    dev_ids = split_conversation_ids(conversations, args.dev_ratio, args.seed)
    examples = {"train": [], "dev": []}
    emotions = {"train": collections.Counter(), "dev": collections.Counter()}
    stats = collections.Counter()

    for conversation in conversations:
        split = "dev" if conversation["conversation_id"] in dev_ids else "train"
        for point in conversation["prediction_points"]:
            stats["points"] += 1
            sample = make_sample(conversation, point["target_user_turn_id"])
            if not sample["history"] or sample["history"][-1]["role"] != "user":
                stats["skipped_bad_history"] += 1
                continue
            target = build_target(point["target"], ri, field_order)
            if target is None:
                stats["skipped_invalid_gold"] += 1
                continue
            user_text, dropped = fit_history(sample, ri, counter, args.max_user_tokens)
            if dropped:
                stats["truncated_points"] += 1
                stats["dropped_turns"] += dropped
            examples[split].append({
                "id": sample["id"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": target},
                ],
            })
            emotions[split][point["target"]["emotion_label"]] += 1

    report = {
        "conversations": len(conversations),
        "dev_conversations": len(dev_ids),
        "points_total": stats["points"],
        "train_examples": len(examples["train"]),
        "dev_examples": len(examples["dev"]),
        "truncated_points": stats["truncated_points"],
        "dropped_turns": stats["dropped_turns"],
        "skipped_invalid_gold": stats["skipped_invalid_gold"],
        "skipped_bad_history": stats["skipped_bad_history"],
        "max_user_tokens": args.max_user_tokens,
        "system_prompt_tokens": counter.count(system_prompt),
        "token_counting": "tokenizer" if counter.exact else f"estimate {CHARS_PER_TOKEN} chars/token",
        "field_order": field_order,
        "emotion_distribution": {split: dict(c.most_common()) for split, c in emotions.items()},
    }
    return examples, report


# ------------------------------------------------------------------ output

def to_sharegpt(example):
    system, user, assistant = example["messages"]
    return {
        "system": system["content"],
        "conversations": [
            {"from": "human", "value": user["content"]},
            {"from": "gpt", "value": assistant["content"]},
        ],
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def write_outputs(examples, report, counter, out):
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in examples.items():
        write_jsonl(out / f"{split}.jsonl", rows)
        write_json(out / f"{split}_sharegpt.json", [to_sharegpt(r) for r in rows])
        if counter.exact:
            rendered = []
            for r in rows:
                prompt, completion = counter.render(r["messages"])
                rendered.append({"id": r["id"], "prompt": prompt, "completion": completion})
            write_jsonl(out / f"{split}_text.jsonl", rendered)

    dataset_info = {
        f"migu_{split}": {
            "file_name": f"{split}_sharegpt.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        }
        for split in examples
    }
    write_json(out / "dataset_info.json", dataset_info, indent=2)
    write_json(out / "build_report.json", report, indent=2)


# ------------------------------------------------------------------ main

def parse_args():
    ap = argparse.ArgumentParser(description="Build SFT data that matches run_inference.py exactly.")
    ap.add_argument("--train", required=True, help="train_public.jsonl")
    ap.add_argument("--inference", required=True, help="the run_inference.py we will submit")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", help="local Qwen3-14B folder (tokenizer files are enough)")
    ap.add_argument("--max-user-tokens", type=int, default=3500,
                    help="budget for the rendered user message; the system prompt (~500-700 tokens) comes on top")
    ap.add_argument("--dev-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--field-order", default=",".join(FIELDS),
                    help="key order of the target JSON; must match the order the inference prompt asks for")
    return ap.parse_args()


def main():
    args = parse_args()
    field_order = [f.strip() for f in args.field_order.split(",")]
    if sorted(field_order) != sorted(FIELDS):
        sys.exit(f"--field-order must list each of {', '.join(FIELDS)} exactly once")
    if not 0 < args.dev_ratio < 1:
        sys.exit("--dev-ratio must be between 0 and 1")

    ri = load_inference_module(args.inference)
    counter = TokenCounter(args.tokenizer)
    with open(args.train, encoding="utf-8-sig") as f:
        conversations = [json.loads(line) for line in f if line.strip()]

    examples, report = build_examples(conversations, ri, counter, args, field_order)
    out = Path(args.out_dir)
    write_outputs(examples, report, counter, out)

    summary = {k: v for k, v in report.items() if k != "emotion_distribution"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    if counter.exact:
        print("Check: every prompt in *_text.jsonl should end with '<think>\\n\\n</think>\\n\\n'.")


if __name__ == "__main__":
    main()
