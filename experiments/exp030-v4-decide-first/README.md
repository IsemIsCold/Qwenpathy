# EXP-030: V4 = QLoRA redo of Qwen3-8B, decide-first, rare classes repeated, dev loss logged

| | |
|---|---|
| **Date** | 2026-11-09 |
| **Author** | Ismail F. |
| **Goal** | Fix what the first QLoRA run (T7 / EXP-019) got wrong, one day before the deadline, without new tooling: emotion decided before the reply, the classes #3/#4 lose to "anxiety" seen more often, a dev loss to watch. |
| **Status** | Built and tested offline (`tests/test_v4.py` 10/10). Two Migu boots pending: training (≈ 2 h 10 min, ≈ 4.4 card-hours) and evaluation (≈ 35 min, ≈ 1.2 card-hours). |

## What changes against T7 (everything else identical: same 14,942 points, prompt, LoRA r16/α32 all-linear, 4-bit base, lr 1e-4 cosine, warmup 3%, 1 epoch, global batch 8, cutoff 6144, template `qwen3` + `enable_thinking: false`)

1. **Decide-first targets** (`train/make_v4_data.py`): the answer JSON is `emotion_label → response_text → user_profile → memory_refs`. The model commits to the emotion before it writes the reply: System One / System Two in a single pass. #4's stop after `emotion_label` still works (generation stops where the profile would start, after the reply).
2. **Rare classes repeated:** one extra copy of every fear (1,044), mixed (799) and shame (593) point and of a seeded half of the helplessness points (585) → 17,963 examples; anxiety 32.1% → 26.7% of the targets. These are the classes #3/#4 turn into anxiety most often on the 600 val points (fear → anxiety 15 of 29).
3. **Dev loss:** 200 seeded points from the held-out dev conversations, every 250 steps (T7 logged none).
4. **Base = `/root/base_qwen3_8b`** explicitly (`/root/participant/model` has held the merged T7 model since Oct 9).
5. Checkpoints every 250 steps and `overwrite_output_dir: false`: a dead boot resumes instead of restarting.

Not changed on purpose: the 350-character reply filter (EXP-018 found it a good match to val), the base model (Qwen3.5-9B needs a new training stack: moved to the national final).

## Pass rules (fixed Nov 9, before training; `scripts/v4_gate.py`)

Control = #4 on the same 600 val points (same code, same CPU profile classifier; emotion 473/600, ROUGE-L 0.2946):

- validate.py 0 errors, 0 parse failures, complete; full-output fallbacks ≤ 6
- emotion ≥ #4 + 12 correct (+2 points) **and** 95% bootstrap CI of the difference (resampling conversations) above 0
- reply ROUGE-L ≥ #4 − 0.005
- latency mean (first 100) ≤ 1.10 × 0.923 s (#4's check run; margin for a different boot)

PASS → the model folders are swapped (T7 weights kept in `/root/model_t7`) and a check run (plain `start.sh` on t9) must be valid, else swapped back. FAIL → #4 stays installed untouched.

## Run

Boot 1 (2 GPUs), after uploading `v4_upload.tgz` and `v4_data.tgz`:
```bash
mkdir -p /mnt/storage/v4_upload /mnt/storage/out && tar -xzf /v4_upload.tgz -C /mnt/storage/v4_upload && tar -xzf /v4_data.tgz -C /mnt/storage
nohup bash /mnt/storage/v4_upload/scripts/v4_train_session.sh > /mnt/storage/out/v4_train_session.log 2>&1 &
tail -f /mnt/storage/out/v4_train_session.log
```
At `DONE`: shut down (GPUs break after training). Boot 2 (fresh, 2 GPUs):
```bash
bash /mnt/storage/v4_upload/scripts/v4_eval_session.sh 2>&1 | tee /mnt/storage/out/v4_eval_session.log
```

## Results (Nov 10): REJECTED by the pre-registered rules. #4 stays the final entry.

**Training** (2 GPUs, 2,246 steps, same recipe as T7): dev loss on the 200 held-out points 1.147 (step 250) → 1.083 → 1.053 → 1.027 → 1.006 (1,250) → 0.979 (2,000). Three interruptions, each resumed from a checkpoint: the instance was restarted at step 663 (preemptible scheduling), a dropped JupyterLab terminal sent SIGHUP at step ~1,830 (`nohup` does not protect `torchrun`, which handles SIGHUP itself → launch with `setsid nohup … < /dev/null &`), and a boot that ended at step 1,881.

**Evaluation on the 600 val points** (V4 merged, served in FP8 with #4's code and config, same boot type as #4's check run):

| | V4 | #4 (control) |
|---|---|---|
| Emotion | 476/600 = 0.793 | 473/600 = 0.788 |
| Only this one right | 39 | 36 |
| Difference, 95% CI (conversations resampled) | −0.023 … +0.032 | |
| ROUGE-L | 0.2957 | 0.2946 |
| Latency, first 100 (mean / median) | 0.890 / 0.739 s | 0.923 / 0.743 s |
| Predicted `anxiety` (gold 195) | 203 | 235 |
| Fallbacks / parse failures | 0 / 0 | 0 / 0 |

Recall by class, V4 vs #4: anxiety 0.88 vs 0.93, relaxed 0.82 vs 0.92, mixed **0.68 vs 0.45**, anger 0.86 vs 0.86, shame 0.89 vs 0.81, loneliness 0.76 vs 0.79, sadness 0.70 vs 0.70, gratitude 0.86 vs 0.79.

Rules: format, fallbacks, ROUGE-L and latency pass; **emotion fails** (needs ≥ 473 + 12 and a CI above 0; got +3, CI spanning 0).

**Reading:** the retrain did what it was built to do (the anxiety over-prediction is gone, mixed and shame are found much more often), but it paid for it one-for-one in anxiety and relaxed. The two models disagree on 75 of 600 points and each wins about half. This matches the Oct 11 error review: the remaining errors sit where the training labels themselves are split, so re-weighting moves errors between classes instead of removing them. For the final: the gain has to come from a better reader or cleaner labels (EXP-029, appraisal questions), or from combining the two models' label probabilities on the points where they disagree, not from another re-weighted run.

Merged V4: `/mnt/storage/merged_v4`; adapter: `/mnt/storage/runs/v4`. `/root/participant` was not touched.
