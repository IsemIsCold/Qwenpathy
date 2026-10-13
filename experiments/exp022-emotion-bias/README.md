# EXP-022: anxiety-prior correction + tag filter off (no training)

| | |
|---|---|
| **Date** | 2026-10-12 (scripts), 2026-10-13 (GPU run) |
| **Author** | Ismail F. |
| **Goal** | Raise emotion accuracy and profile F1 of submission #2 **without training**, and settle FP8 vs bf16 with the same run. |
| **Status** | **Done (Oct 13). Nothing adopted:** bias on bf16 fails the rule (+1.4 in-sample, 0 switches out of sample), tag filter stays on, FP8 + bias handed to [EXP-023](../exp023-fp8-bias/README.md). Submission #2 stays. |

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

## Decision rule (fixed Oct 12, before the run)

- **Bias:** adopt b only if it gains ≥ 1.5 points on tune500 **and** doesn't lose accuracy on n50 or n50b. Using it in the submission needs `logprobs` in the real request and a parser change. So also check the latency pair: if logprobs add > 5% latency, the gain has to cover that.
- **Filter off:** adopt if profile micro-F1 (avg) rises on tune500 **and** on n100, with exact match not lower.
- **FP8:** compare `lp_fp8` vs `lp_bf16` on n100 with the chosen parser.

## Run (Oct 13, one 2-GPU boot, ≈ 0.8 card-hours)

- **First launch: HTTP 404 on every request.** `emotion_logprobs.py` sent `config["api_model"]`, which is `local-model` in `inference_config.json` (start.sh overrides it through `EVALUATION_API_MODEL`), while the session serves `--served-model-name model`. The stub server accepted any model name, so the tests missed it. Fix: the payload now always sends `"model"` (Migu copy patched with sed, repo copy too). GPUs were idle and clean (1 MiB each) when it failed.
- **Relaunch guard bug:** `pgrep -af exp022_session || nohup ...` matched our own `tail -f …exp022_session.log`, so the relaunch was skipped. Use `pgrep -af "vllm serve|exp022_session\.sh"` as a separate step.
- bf16 on t10 (600 points, 8 at once): label logprobs on 592/600 (8 have no label token in the top 20); top label = parsed label on 589/592; label position found on 600/600. FP8 on n100: 100/100 and 100/100.
- The bf16 no-bias baseline on n100 is 0.83 (T9: 0.82), so the logprobs run represents the submission.

## Results

### Bias on bf16 (b tuned on tune500)

| b | 0 | 0.25 | **0.5** | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| tune500 accuracy | 0.782 | 0.782 | **0.796** | 0.794 | 0.782 | 0.780 | 0.764 | 0.712 |

Anxiety predicted on tune500: 39.2% → 35.2% at b = 0.5 (gold 32.4%).

| Slice | Emotion acc b 0 → 0.5 | Labels switched | Profile micro-F1 filter on / off | Exact match on / off |
|---|---|---|---|---|
| tune500 (b picked here) | 0.782 → 0.796 | 20 | 0.4561 / 0.4616 | 0.544 / 0.543 |
| n50 | 0.80 → 0.80 | 0 | 0.4866 / 0.4831 | 0.453 / 0.453 |
| n50b | 0.86 → 0.86 | 0 | 0.4768 / 0.4768 | 0.520 / 0.520 |
| n100 | 0.83 → 0.83 | 0 | 0.4806 / 0.4788 | 0.487 / 0.487 |

### FP8 on n100 (filter on)

| | bf16 | FP8 b 0 | FP8 b 0.5 (bf16's b, fixed before) | FP8 b 1.5 (not pre-registered) |
|---|---|---|---|---|
| Emotion accuracy | 0.83 | 0.79 | 0.82 (3 switched, all to the gold label) | 0.84 (7 switched) |
| Profile micro-F1 | 0.481 | 0.467 | 0.467 | 0.467 |
| ROUGE-L / BLEU-4 | 0.2827 / 0.1403 | 0.2743 / 0.1319 | same | same |
| Reply median (chars) | 121.5 | 119.5 | | |

The b = 1.5 run was a typo (an example value from the run instructions). It is reported but not used for any decision: choosing between 0.5 and 1.5 on n100 would tune on the only FP8 check slice.

### Latency of logprobs (bf16, 20 points of t9, one request at a time)

| Request | Mean |
|---|---|
| plain (= submission) | 1669 ms |
| + `logprobs`, `top_logprobs: 20` | 1805 ms (**+136 ms, +8%**) |

Offline check on all 700 points: keeping only the **top 5** alternatives gives exactly the same switch decisions as the top 20 for every b ≤ 1.0 (1 difference at b = 1.5 on bf16). So a submission would request `top_logprobs: 5`.

## Decision

- **Bias on bf16: rejected.** +1.4 points on tune500 is below the +1.5 bar, and it is in-sample (b was picked on those 500 points). Out of sample (n50, n50b, n100) b = 0.5 changes **no label at all**: on those points no anxiety prediction of the bf16 model has another label within 0.5 logprob, so there is nothing for a small bias to fix.
- **Tag filter off: rejected.** Profile micro-F1 rises on tune500 (+0.006) but falls on n100 (−0.002) and n50 (−0.004); exact match is slightly lower on tune500. The filter stays on.
- **FP8:** the bias recovers part of FP8's emotion loss on n100 (0.79 → 0.82) but not its profile loss (−0.014). Judging that on 100 points is too thin → [EXP-023](../exp023-fp8-bias/README.md) re-runs FP8 on all 600 points with its own b and a gate fixed in advance.
- **Submission #2 stays.**

## Files

- Laptop: `outputs/exp022/` (raw answers + logprobs `lp_bf16.jsonl`, `lp_fp8.jsonl`, latency pair, vLLM logs; tuner outputs in `bf16/`, `fp8/` (b 1.5), `fp8_b05/`), `outputs/exp022_session.log`. Archive: `migu-backups/exp022.tgz`.
- Migu: `/mnt/storage/out/exp022/`.
