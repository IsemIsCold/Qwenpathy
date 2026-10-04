# Brief: Zakaria (dataset)

> **Original brief from Oct 1, kept for reference.** Some details are outdated (e.g. the base model is now Qwen3-8B, prompts live in `run_inference.py`). Current status: [docs/team-work.md](../team-work.md) and [TASKS.md](../../TASKS.md).

**You own:** turning the organisers' data into what everyone else needs. Evaluation needs val inputs and gold labels, training needs SFT data, and everyone needs the stats.

**Tasks:** B2 `convert_val.py` (Oct 2), B6 data stats (Oct 3), I1 `build_sft.py` (Oct 6).

Do B2 first: Yahya and Elly are both waiting on it.

## B2: `scripts/convert_val.py`

**Outputs:**

- `data/val_input.jsonl`: exactly the test input format, one line per prediction point: `{id, conversation_id, target_user_turn_id, history}`, with `history` **ending at the target user turn**
- `data/val_gold.jsonl`: `{id, response_text, emotion_label, user_profile, memory_refs}`

Add `--n N` to take only the first N points.

**Hints:**

- Open one val conversation by hand first and find where the gold labels live. The val set has 900 conversations but 2,354 prediction points, so each conversation produces several points.
- Match the sample test file in `participant/` **exactly**: key names, how `turn_id` is written.
- Always use `encoding="utf-8"` and `json.dumps(..., ensure_ascii=False)`.
- **Write the format down in `docs/data-format.md` on day 1.** Yahya builds `eval.py` against it.

## B6: `docs/data-stats.md`

- The emotion label distribution (train and val)
- Profile labels: how often each one appears, and how many profiles are empty
- Turns per conversation, characters per history, and target reply length. Give percentiles (50/90/99/max), not just the mean.
- How many histories would go over 2k, 4k and 8k **tokens** with the Qwen tokenizer. This decides the context length and the truncation rule.

## I1: `train/build_sft.py`

The train set has 8,338 conversations and 18,657 prediction points. Turn each point into one chat-format example ending in the assistant's target output. Agree the exact target format with Ismail M. (full JSON, or the reply only?).

- **Split by `conversation_id`**, never by point. Points from the same conversation must not end up in both train and dev.
- Use the **same system prompt as inference** (`prompts/`).
- Long conversations: drop the oldest turns first. Always keep the system prompt and the last user turn.
