# Brief: Yahya (data plumbing + validation)

**Tasks:** B2 `convert_val.py` and B3 `validate.py` (due Oct 2), then I5 latency (Oct 12).

Everyone else depends on your scripts, so they come first.

## B2: `scripts/convert_val.py`

**Input:** the val set from the organisers' package.
**Outputs:**

- `data/val_input.jsonl`: exactly the test input format, one line per prediction point:
  `{id, conversation_id, target_user_turn_id, history: [...]}`, with `history` **ending at the target user turn**
- `data/val_gold.jsonl`: `{id, response_text, emotion_label, user_profile, memory_refs}` for the same ids

Add a `--n N` flag to take only the first N points, for quick tests.

**Hints:**

- First open one val conversation and work out where the gold labels live. Are they on the assistant turn after the target user turn? The val set has 900 conversations but 2,354 prediction points, so one conversation gives several points.
- Look for a sample test file in `participant/` and match it **exactly**: key names, key order, how `turn_id` is written.
- Read and write with `encoding="utf-8"`, and use `ensure_ascii=False` in `json.dumps` so the Chinese stays readable.

## B3: `scripts/validate.py`

`python scripts/validate.py outputs/submission.jsonl [--input data/val_input.jsonl]`

Check and report:

- Every line is valid JSON, and the file is UTF-8.
- There's exactly one output per input `id`: none missing, none duplicated.
- `response_text` is a non-empty string.
- `emotion_label` is one of the 16 allowed labels.
- Every `user_profile` value is from the allowed lists (copy them from the official README).
- `memory_refs` is a list, and each item matches `mem_\d{6}`.

Exit code 1 if anything fails, with a summary like `2354 lines, 3 bad emotion labels, 0 missing ids`.

**Test it:** break a file on purpose (bad label, missing id, broken JSON) and check that each problem gets caught.
