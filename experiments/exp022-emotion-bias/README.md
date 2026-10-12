# EXP-022: anxiety-prior correction + tag filter off (no training)

| | |
|---|---|
| **Date** | 2026-10-12 (scripts), 2026-10-13 (GPU run) |
| **Author** | Ismail F. |
| **Goal** | Raise emotion accuracy and profile F1 of submission #2 **without training**, and settle FP8 vs bf16 with the same run. |
| **Status** | Scripts written and tested on a stub server (no GPU). GPU session pending. |

## Why

- The model says **anxiety on 42%** of val points; gold has it on 33% (n100: recall 1.0, precision 0.76). Most remaining emotion errors are "anxiety" where gold is fear or relaxed (P1 review, EXP-018).
- 1 point of emotion accuracy ≈ 0.25 competition points.
- The tag filter deletes `study_exam`, `colloquial`, `emotionally_stable`, which the fine-tuned model was trained to output (EXP-018 §4).

## Method

1. `scripts/emotion_logprobs.py` sends each point to the running vLLM server with the **submission's own prompt and settings** (imported from `participant/run_inference.py` and its config) plus `logprobs`, and saves the raw answer and the top-20 alternatives at the first token of the emotion label. Each of the 16 labels starts with a different token (`an` = anxiety, `f` = fear, `rel` = relaxed, …; checked with the Qwen3 tokenizer), so those 20 alternatives give a probability for each label.
2. `scripts/tune_emotion_bias.py` (offline): when the parsed label is anxiety, switch to the best other label if `logprob(other) > logprob(anxiety) − b`. **b is picked on a 500-point tuning set** that shares no point with n100, then checked on n50 / n50b. The same raw answers are parsed with the tag filter on and off.
3. Why not vLLM `logit_bias`: it applies to every generated token, and `an` is common in replies and JSON.

## Data

| File | Points | Use |
|---|---|---|
| `t9_input.jsonl` = n100 (n50 + n50b) | 100 | report only (never used to pick b) |
| `val_*_tune500.jsonl` | 500 | pick b; second check of the filter |
| `t10_input.jsonl` = n100 + tune500 | 600 | bf16 logprob run |

Build tune500 (laptop): `python scripts/convert_val.py --in data/val_public.jsonl --out data/val_input_tune500.jsonl --gold data/val_gold_tune500.jsonl --exclude data/val_gold_n100.jsonl --sample 500 --seed 7`

## GPU session (one 2-GPU boot)

`bash /mnt/storage/scripts/exp022_session.sh`: bf16 server → logprobs on 600 points (8 requests at once) → 20-point latency pair (plain vs logprobs, one at a time) → FP8 server → logprobs on n100. Estimate: ~20–25 min, **~0.8 card-hours**.

Note: 8 requests at once lets vLLM batch them. Batched greedy decoding can differ from the one-at-a-time submission run in a few tokens, so these numbers can differ slightly from T9's. That's fine for comparing variants, because they all parse the same answers.

## Stub test (Oct 12, no GPU)

A fake server built from gold: it says anxiety on 40% of non-anxiety points, with gold label 0.1–1.5 logprob behind. It tokenises the answer with the real Qwen3 tokenizer.

- Label position found on 100/100 points; top logprob label = parsed label on 100/100.
- Concurrent output identical to the one-at-a-time output.
- The tuner picked b = 1.45 on n50 (the largest planted gap is 1.4, and wrong switches start at 1.8). On n50b, accuracy went 0.84 → 0.98.
- Filter-off profile F1 1.00 vs 0.99 with the filter on.
- `eval.py` accepts every written file.
- Full session script run with a fake `vllm`: all four outputs written, and no server left running after exit or after a crash.

## Decision rule (fill in Oct 13)

- **Bias:** adopt b only if it gains ≥ 1.5 points on tune500 **and** doesn't lose accuracy on n50 or n50b. Using it in the submission needs `logprobs` in the real request and a parser change. So also check the latency pair: if logprobs add > 5% latency, the gain has to cover that.
- **Filter off:** adopt if profile micro-F1 (avg) rises on tune500 **and** on n100, with exact match not lower.
- **FP8:** compare `lp_fp8` vs `lp_bf16` on n100 with the chosen parser.

## Results

(pending)
