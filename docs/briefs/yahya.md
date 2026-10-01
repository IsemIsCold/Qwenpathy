# Brief: Yahya (evaluation)

**You own:** the numbers. Every decision (which prompt, which model, whether a submission is worth one of our 5 attempts) goes through your scripts.

**Tasks:** B3 `validate.py` (Oct 2), B4 `eval.py` (Oct 3), I4 reply-quality judge (Oct 8).

The input formats come from Zakaria (`docs/data-format.md`), so agree on them with him on day 1.

## B3: `scripts/validate.py`

`python scripts/validate.py outputs/submission.jsonl --input data/val_input.jsonl`

Checks:

- Every line is valid JSON, and the file is UTF-8.
- There's exactly one output per input `id`: none missing, none duplicated.
- `response_text` is a non-empty string.
- `emotion_label` is one of the 16 allowed labels.
- `user_profile` values come from the allowed lists (copy them from the official README).
- `memory_refs` is a list of `mem_\d{6}`.

Exit code 1 on any failure, with a summary like `2354 lines, 3 bad labels, 0 missing ids`. Break files on purpose to test that each check fires.

## B4: `scripts/eval.py`

`python scripts/eval.py outputs/submission.jsonl data/val_gold.jsonl`

Report:

1. **Emotion:** accuracy, **macro-F1** and a per-label precision/recall table.
2. **Profile:** F1 per sub-field (`personality_traits`, `interests`, `style`).
3. **Baselines:** "always predict `anxiety`", and "empty profile".
4. **Reply length:** mean and median, predicted vs. gold.

Save the results as JSON and append a row to `docs/score-history.md`.

**Hints:**

- Why macro-F1? 34% of the labels are `anxiety`. A model that always says "anxiety" gets 34% accuracy and is useless. Macro-F1 makes the rare classes count.
- The profile is multi-label: read up on micro vs. macro F1 and Jaccard, pick one, and say why in your PR.
- About 37% of gold profiles are empty. Decide how to score empty vs. empty.
- `sklearn.metrics` already does most of this.

## I4: reply-quality judge

`response_text` is probably judged on empathy, coherence and memory. Build a judge that scores replies 1–5 on those three points. **Use it during development only.** The submission itself must stay offline, but the judge may call an API. Check that its scores roughly match human judgement on 20 samples.
