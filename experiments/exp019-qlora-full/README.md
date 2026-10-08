# EXP-019: Full QLoRA fine-tune of Qwen3-8B (T5–T7)

| | |
|---|---|
| **Date** | 2026-10-07 / 08 |
| **Author** | Ismail F. (data rebuild, Migu setup, pilot, full run, merge, eval) |
| **Goal** | Fine-tune Qwen3-8B on the training set with the frozen V1.8 prompt and beat V1.8 on **both** val slices (TASKS rule for submission #2). |
| **Status** | **Done. PASS.** The full model beats V1.8 and the pilot on every metric on n50 and n50b. It is the candidate for submission #2 (after T9). |

## Setup

- **Base model:** Qwen3-8B (`/root/participant/model`), non-thinking.
- **Data:** `train/build_sft.py --max-reply-chars 350` → **14,942** training examples (`migu_train`). Val conversations excluded; checked by `train/check_sft.py` (PASS, cutoff 6144). Prompt = exactly the V1.8 `SYSTEM_PROMPT` + `conversation_text()` from `participant/run_inference.py`, so training and inference see identical inputs.
- **Method:** QLoRA. 4-bit NF4 base (bitsandbytes), LoRA r=16, α=32, dropout 0.05, on all linear layers (43.6M trainable params, 0.53%). lr 1e-4 cosine, warmup 3%, 1 epoch, `cutoff_len 6144`, bf16.
- **Template:** `template: qwen3` + `enable_thinking: false`. This puts the empty `<think>\n\n</think>\n\n` block in the prompt exactly as vLLM does at inference; the loss is only on the JSON answer. (`qwen3_nothink` in LLaMA-Factory 0.9.5 does **not** add that block.)
- **Hardware:** Migu, 2× RTX 4090, DDP (global batch 8 = 1 per GPU × accum 4 × 2 GPUs, 1,868 steps). Training env = venv `/opt/train_venv` (LLaMA-Factory 0.9.5, torch 2.10.0+cu128); the submission env `conda_env` stays untouched.
- **Eval:** merged bf16 weights served by vLLM through the submission's own `start.sh` (TP2, greedy), scored with `scripts/eval.py`. Same path as the T4 baseline.

Exact configs: [configs/full.yaml](configs/full.yaml), [configs/merge_full.yaml](configs/merge_full.yaml), pilot [configs/pilot_ddp.yaml](configs/pilot_ddp.yaml).

### Commands (on Migu)

```bash
# train (2-GPU boot)
export PATH=/opt/train_venv/bin:$PATH        # torchrun shim -> the venv's python
FORCE_TORCHRUN=1 nohup llamafactory-cli train /mnt/storage/configs/full.yaml > /mnt/storage/runs/full.log 2>&1 &

# merge LoRA into bf16 weights (CPU) + use the base model's tokenizer files
llamafactory-cli export /mnt/storage/configs/merge_full.yaml
cd /root/merged/full
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json added_tokens.json special_tokens_map.json; do
  if [ -f /root/participant/model/$f ]; then cp /root/participant/model/$f .; fi
done
rm -f chat_template.jinja

# eval (FRESH 2-GPU boot, see "What went wrong")
for s in n50 n50b; do
  env VLLM_MODEL_DIR=/root/merged/full VLLM_SERVED_MODEL_NAME=model bash /root/participant/start.sh /mnt/storage/val/val_input_$s.jsonl /mnt/storage/out/full_$s
  /opt/train_venv/bin/python /mnt/storage/scripts/eval.py /mnt/storage/out/full_$s/submission.jsonl /mnt/storage/val/val_gold_$s.jsonl --out /mnt/storage/out/full_$s/metrics.json --history "" --force
done
```

## Training

| | Pilot (T5) | **Full (T7)** |
|---|---|---|
| Examples | 2,000 (`migu_pilot2k`) | **14,942** (`migu_train`) |
| Steps (global batch 8) | 250 | 1,868 |
| Runtime | 892 s | **6,477 s (1 h 48 min)** |
| Throughput | 2.24 samples/s | 2.31 samples/s |
| Mean train loss | 1.315 | **1.130** |

![Training loss of the full run](loss_curve.svg)

Same data as a table ([train/trainer_log.jsonl](train/trainer_log.jsonl), mean per 200 steps):

| Steps | 1–200 | 201–400 | 401–600 | 601–800 | 801–1000 | 1001–1200 | 1201–1400 | 1401–1600 | 1601–1800 | 1801–1868 |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 1.348 | 1.171 | 1.143 | 1.127 | 1.091 | 1.095 | 1.063 | 1.069 | 1.046 | 1.056 |

Fast drop in the first 200 steps (≈2.2 → ≈1.2: the model learns the JSON format and the label set), then a slow, steady decline with no divergence. Still slightly falling at the end, so one epoch did not overfit. The log starts at step 9: steps 1–8 were overwritten when the run was accidentally launched twice (see below).

## Results

Same harness, same val slices, greedy. V1.8 = the base model with the frozen prompt on Migu vLLM (T4).

| Metric | n50: V1.8 | n50: Pilot | **n50: Full** | n50b: V1.8 | n50b: Pilot | **n50b: Full** |
|---|---|---|---|---|---|---|
| Emotion accuracy (official) | 0.54 | 0.68 | **0.80** | 0.62 | 0.78 | **0.86** |
| Emotion macro-F1 | 0.238 | 0.339 | **0.510** | 0.336 | 0.449 | **0.597** |
| Profile micro-F1 (avg) | 0.352 | 0.410 | **0.489** | 0.323 | 0.340 | **0.477** |
| Profile exact match (avg) | 0.287 | 0.420 | **0.453** | 0.287 | 0.380 | **0.500** |
| ROUGE-L (char) | 0.174 | 0.237 | **0.269** | 0.197 | 0.259 | **0.293** |
| BLEU-4 (char) | 0.021 | 0.104 | **0.130** | 0.068 | 0.133 | **0.151** |
| Reply median, chars (gold) | 85 (146) | 128 | 150 (146) | 96 (100.5) | 110 | 106 (100.5) |
| Latency mean / median | 1.32 s | 1.62 s | 1.66 / 1.47 s | 1.52 s | 1.55 s | 1.50 / 1.30 s |
| Parse failures | 0 | 0 | 0 | 0 | 0 | 0 |

For reference, the DeepSeek V4.1-Flash "teacher" via API (EXP-017) scored 0.62 / 0.72 emotion accuracy and 0.231 / 0.249 ROUGE-L. The fine-tuned 8B model beats it on both.

**What improved most:**
- **Emotion:** +26 / +24 points over V1.8. Rare classes went from mostly wrong to mostly right: fear F1 1.00 on n50 (pilot 0.57), shame and disgust 1.00, sadness and loneliness 1.00 on n50b.
- **Replies:** ROUGE-L +0.10 and BLEU-4 ×6 / ×2 over V1.8. Reply length now matches the gold length (median 150 vs 146, 106 vs 100.5); the base model wrote replies that were too short.
- **Profile:** +0.14 / +0.15 micro-F1, and exact match is above V1.8 on both slices.

**Remaining errors:** fear → anxiety (n50b ×3), relaxed → anxiety (n50 ×3). The model still leans towards `anxiety`, the majority class (32% of the data). Target for P2 (class balance / rare-emotion data), if it runs.

**Latency:** +0.34 s on n50 versus V1.8 because the replies are longer (closer to gold); unchanged on n50b. The official latency formula is unknown (organiser question O1b); a small latency cost for +25 points of emotion accuracy is the right trade.

## What went wrong (and the fixes)

1. **GPUs break after a training process exits.** Twice (pilot and full run), vLLM failed within a minute of training ending with `Failed to initialize NVML: Unknown Error` / `Failed to infer device type`. The chained script ([configs/run_t7.sh](configs/run_t7.sh): train → merge → eval) therefore died at the eval step. **Rule now: train in one boot, eval in a fresh boot.** Asked Migu whether this is a known platform issue.
2. **Pilot stopped by accident** at step 179/500 on Oct 7 (1-GPU run). Rerun on 2 GPUs on Oct 8 instead of resuming, which also proved 2-GPU DDP works (E1).
3. **Run launched twice.** The second copy crashed (`CUBLAS_STATUS_ALLOC_FAILED`, GPU memory already taken) and truncated the log files; the first copy trained normally. Lesson: launch once; if nothing prints, it is the `sleep` waiting.
4. **Storage incident.** Overnight, while the instance was off, `merges.txt` files in checkpoint folders on `/mnt/storage` were replaced by OpenSSL-encrypted `merges.txt.enc`. Nothing else was touched. Merged models are kept on `/root`, and tokenizer files are always copied from the base model.

## Cost

Training 1 h 48 min on 2 GPUs ≈ 3.6 card-hours, plus ≈ 0.4 for the eval boot. The exact total for Oct 7–8 comes from the Migu page (E2, LOG.md).

## Decision

- **Submission #2 = this model** (`/root/merged/full`), after the T9 submission check (plain `start.sh` with the weights in `participant/model`, validate, latency on the first 100).
- **T8 (checkpoint sweep) dropped:** the final checkpoint is clearly best; differences between checkpoints on 50 samples would be noise.
- The merged pilot (`/root/merged/pilot_ddp`) is the fallback; V1.8 is the last resort.

## Files

- `configs/`: `full.yaml`, `merge_full.yaml`, `pilot_ddp.yaml`, `run_t7.sh` (exact files used on Migu)
- `train/`: `train_results.json`, `trainer_log.jsonl` (loss per step)
- `loss_curve.svg`: the chart above, generated from `train/trainer_log.jsonl` (grey = per-step loss, blue = 50-step moving average, grid lines = checkpoints)
- `n50/`, `n50b/`: `metrics.json` (eval.py), `performance_report.json` (latency), `submission.jsonl` (the 50 predictions)
- Not in git (weights): LoRA adapter + checkpoints at steps 467 / 934 / 1401 in `/mnt/storage/runs/full`; merged model in `/root/merged/full` on Migu.
