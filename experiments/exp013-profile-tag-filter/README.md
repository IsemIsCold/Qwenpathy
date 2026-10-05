# EXP-013: Offline profile tag filter on V0 output

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Remove the three tags V0 over-predicts and re-score, with no API call. |
| **Part of** | [EXP-005 prompt series](../exp005-prompt/README.md) (version table + all prompts there) |
| **Status** | Done. +8.5 points profile micro-F1 for free; kept (V1.7), confirmed on n50b in EXP-016. |

## Setup

`Qwen/Qwen3-8B:nscale` via the Hugging Face router, non-thinking, greedy (temperature 0), val n50. Prompt = `participant_v1_x/run_inference.py` (only `SYSTEM_PROMPT` / `normalize_prediction` differ from `participant/`; the exact text is in the EXP-005 README).

Offline: re-scored V0's existing `experiments/exp005-prompt/v0/out_nothink/submission.jsonl` with `eval.py` functions after removing the tags (no model call).

Run outputs (submission, raw outputs, performance report): none (offline)

## Results

analyze.py showed V0 over-predicts three tags (V0 count vs gold count on n50): personality `emotionally_stable` 47 vs 2, interests `study_exam` 36 vs 0, style `colloquial` 38 vs 3. Removing them from V0's existing predictions, scored with eval.py:

| | micro-F1 avg | exact match avg | personality / interests / style micro-F1 |
|---|---|---|---|
| V0 as is | 0.275 | 0.067 | 0.27 / 0.36 / 0.19 |
| **drop 3 tags** | **0.360** | **0.240** | 0.36 / 0.51 / 0.21 |
| all-empty profile | 0.000 | 0.373 | — |

+8.5 points micro-F1 and +17 points exact match, with no API call, no emotion risk and no latency cost. **Caveat:** found and measured on the same 50 points, so it must be confirmed on a different val slice before shipping.

## Files

- none (offline experiment; numbers above)
