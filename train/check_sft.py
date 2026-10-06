#!/usr/bin/env python3
"""
train/check_sft.py (T1 sign-off): verify data/sft/ before any GPU time, and make the pilot subset.

Checks
  1. No validation conversation appears in train/dev.
  2. Every target is valid JSON with the 4 fields, memory_refs == [], no \\u escapes,
     labels inside the official vocabulary.
  3. System prompt is identical to the one run_inference.py builds; untruncated user
     messages are identical to run_inference.conversation_text().
  4. Exact token lengths (Qwen3 chat template, thinking off): p50/p95/max, and how many
     examples exceed --cutoff-len (LLaMA-Factory would cut those, possibly the target).
Writes
  data/sft/pilot2k_sharegpt.json + a "migu_pilot2k" entry in data/sft/dataset_info.json
  data/sft/check_report.json

Usage
  python train/check_sft.py --tokenizer models/qwen3-8b-tok --cutoff-len 6144
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft import build_system_prompt, load_inference_module, make_sample, to_sharegpt  # noqa: E402

FIELDS = ["response_text", "emotion_label", "user_profile", "memory_refs"]


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def conv_id(example_id):
    return example_id.rsplit("_t", 1)[0]


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", default="data/sft")
    ap.add_argument("--train", default="data/train_public.jsonl")
    ap.add_argument("--val", default="data/val_public.jsonl")
    ap.add_argument("--inference", default="participant/run_inference.py")
    ap.add_argument("--tokenizer", help="local Qwen3-8B tokenizer folder; without it lengths are estimates")
    ap.add_argument("--cutoff-len", type=int, default=6144)
    ap.add_argument("--pilot-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sft = Path(args.sft_dir)
    splits = {s: read_jsonl(sft / f"{s}.jsonl") for s in ("train", "dev")}
    problems = []

    # 1. val leakage
    val_ids = {c["conversation_id"] for c in read_jsonl(args.val)}
    leaked = sorted({conv_id(r["id"]) for rows in splits.values() for r in rows} & val_ids)
    if leaked:
        problems.append(f"{len(leaked)} val conversations in SFT data, e.g. {leaked[:3]}")
    train_dev_overlap = {conv_id(r["id"]) for r in splits["train"]} & {conv_id(r["id"]) for r in splits["dev"]}
    if train_dev_overlap:
        problems.append(f"{len(train_dev_overlap)} conversations in both train and dev")

    # 2. targets
    ri = load_inference_module(args.inference)
    vocab = {"emotion": set(ri.EMOTIONS), **{k: set(v) for k, v in ri.PROFILE_LABELS.items()}}
    bad_targets = []
    for rows in splits.values():
        for r in rows:
            text = r["messages"][2]["content"]
            try:
                t = json.loads(text)
                ok = (list(t) == FIELDS and t["memory_refs"] == [] and "\\u" not in text
                      and t["response_text"].strip() and t["emotion_label"] in vocab["emotion"]
                      and all(set(t["user_profile"][k]) <= vocab[k] for k in ri.PROFILE_LABELS))
            except (ValueError, KeyError, TypeError, AttributeError):
                ok = False
            if not ok:
                bad_targets.append(r["id"])
    if bad_targets:
        problems.append(f"{len(bad_targets)} bad targets, e.g. {bad_targets[:3]}")

    # 3. prompt identical to inference
    system_prompt = build_system_prompt(ri)
    convs = {c["conversation_id"]: c for c in read_jsonl(args.train)}
    sys_mismatch = user_mismatch = truncated = 0
    for rows in splits.values():
        for r in rows:
            if r["messages"][0]["content"] != system_prompt:
                sys_mismatch += 1
            turn = int(r["id"].rsplit("_t", 1)[1])
            full = ri.conversation_text(make_sample(convs[conv_id(r["id"])], turn))
            user = r["messages"][1]["content"]
            if user != full:
                # a truncated history keeps the same tail (the latest user message)
                if full.endswith(user.split("\n对话历史:\n", 1)[1]):
                    truncated += 1
                else:
                    user_mismatch += 1
    if sys_mismatch:
        problems.append(f"{sys_mismatch} system prompts differ from run_inference.py (rebuild with build_sft.py)")
    if user_mismatch:
        problems.append(f"{user_mismatch} user messages differ from run_inference.conversation_text()")

    # 4. token lengths (render all, then tokenize in batches: per-example calls take ages on a laptop)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

        def lengths_of(message_lists):
            texts = [tok.apply_chat_template(m, tokenize=False, enable_thinking=False) for m in message_lists]
            out = []
            for i in range(0, len(texts), 256):
                out += [len(ids) for ids in tok(texts[i:i + 256], add_special_tokens=False)["input_ids"]]
                print(f"  tokenized {min(i + 256, len(texts))}/{len(texts)}", end="\r")
            print()
            return out

        def target_lengths(rows):
            texts = [r["messages"][2]["content"] for r in rows]
            return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]

        sample = tok.apply_chat_template(splits["dev"][0]["messages"], tokenize=False, enable_thinking=False)
        if "<think>\n\n</think>\n\n" not in sample:
            problems.append("rendered example has no empty <think> block: check enable_thinking / template")
        method = "tokenizer"
    else:
        def lengths_of(message_lists):
            return [int(sum(len(x["content"]) for x in m) / 1.4) + 30 for m in message_lists]

        def target_lengths(rows):
            return [int(len(r["messages"][2]["content"]) / 1.4) for r in rows]
        method = "estimate 1.4 chars/token (pass --tokenizer for the real numbers)"
    think_leaks = [r["id"] for rows in splits.values() for r in rows
                   if "<think>" in r["messages"][2]["content"] or "</think>" in r["messages"][2]["content"]]
    if think_leaks:
        problems.append(f"{len(think_leaks)} targets contain <think> reasoning, e.g. {think_leaks[:3]} (rebuild with the fixed build_sft.py)")
    lengths = {s: lengths_of([r["messages"] for r in rows]) for s, rows in splits.items()}
    target_lens = target_lengths(splits["train"])
    over = {s: sum(n > args.cutoff_len for n in ls) for s, ls in lengths.items()}
    if any(over.values()):
        problems.append(f"examples longer than cutoff_len={args.cutoff_len}: {over}. "
                        "Raise cutoff_len or rebuild with a smaller --max-user-tokens.")

    # pilot subset
    pilot = random.Random(args.seed).sample(splits["train"], min(args.pilot_size, len(splits["train"])))
    with open(sft / "pilot2k_sharegpt.json", "w", encoding="utf-8") as f:
        json.dump([to_sharegpt(r) for r in pilot], f, ensure_ascii=False)
    info_path = sft / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["migu_pilot2k"] = {**info["migu_train"], "file_name": "pilot2k_sharegpt.json"}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    pilot_emotions = {}
    for r in pilot:
        e = json.loads(r["messages"][2]["content"])["emotion_label"]
        pilot_emotions[e] = pilot_emotions.get(e, 0) + 1

    all_lens = lengths["train"] + lengths["dev"]
    report = {
        "examples": {s: len(rows) for s, rows in splits.items()},
        "val_conversations_leaked": len(leaked),
        "bad_targets": len(bad_targets),
        "system_prompt_mismatch": sys_mismatch,
        "user_message_mismatch": user_mismatch,
        "truncated_histories": truncated,
        "token_counting": method,
        "total_tokens": {"p50": pct(all_lens, .5), "p95": pct(all_lens, .95), "max": max(all_lens)},
        "target_tokens": {"p50": pct(target_lens, .5), "p95": pct(target_lens, .95), "max": max(target_lens)},
        "train_tokens_sum": sum(lengths["train"]),
        "cutoff_len": args.cutoff_len,
        "over_cutoff": over,
        "pilot": {"size": len(pilot), "seed": args.seed, "emotions": dict(sorted(pilot_emotions.items(), key=lambda kv: -kv[1]))},
        "problems": problems,
    }
    (sft / "check_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nT1 PASS" if not problems else "\nT1 FAIL — fix the problems above")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
