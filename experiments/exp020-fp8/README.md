# EXP-020: FP8 weights at inference (latency test)

| | |
|---|---|
| **Date** | 2026-10-12 |
| **Author** | Ismail F. |
| **Goal** | Cut latency (10 points, first 100 samples) without retraining. The organisers confirmed that quantised models are allowed. |
| **Status** | **Done. Gate failed on quality.** Faster by 25%, but accuracy and profile drop. Submission #2 (bf16) stays unless the latency formula makes the speed worth it. |

## Setup

- **Model:** the merged full QLoRA model from EXP-019 (`/root/participant/model`), unchanged.
- **Change:** one vLLM flag, `--quantization fp8`. vLLM quantises the bf16 weights to FP8 when it loads them ("online" FP8; log: `Selected CutlassFP8ScaledMMLinearKernel for Fp8OnlineLinearMethod`). The RTX 4090 (Ada, sm89) supports FP8 natively. No re-quantisation step, no new weights on disk.
- **Script:** `participant/start_fp8.sh` = `start.sh` with the flag added on the 2-GPU line, made on Migu with:
  ```bash
  sed 's/--gpu-memory-utilization 0.90)/--gpu-memory-utilization 0.90 --quantization fp8)/' start.sh > start_fp8.sh
  ```
  The submission's own `start.sh` was not touched.
- **Run:** one fresh 2-GPU boot, `bash /root/participant/start_fp8.sh /mnt/storage/val/t9_input.jsonl /mnt/storage/out/fp8` (100 points: n50, then n50b; same input file as T9, sha256 `0b99b85f…8b8969`). Greedy, same prompt (V1.8), TP2, 32k context. Eval and validate on a no-GPU boot.

## Results (T9 = the same model in bf16, run the same way on Oct 9)

| | n50: T9 → FP8 | n50b: T9 → FP8 |
|---|---|---|
| Emotion accuracy | 0.80 → **0.76** (−2 labels) | 0.84 → 0.84 |
| Emotion macro-F1 | 0.509 → 0.493 | 0.574 → 0.574 |
| Profile micro-F1 / exact | 0.490 / 0.447 → **0.447 / 0.393** | 0.495 / 0.507 → **0.434 / 0.487** |
| ROUGE-L | 0.267 → 0.258 | 0.292 → 0.289 |
| BLEU-4 | 0.130 → 0.126 | 0.147 → 0.150 |

| Latency, first 100 | T9 (bf16) | **FP8** |
|---|---|---|
| Mean | 1.55 s | **1.16 s** (−25%) |
| Median | 1.35 s | 1.00 s |
| p95 | 3.59 s | 2.32 s |
| Max | 4.85 s | 4.13 s |

`validate.py`: 0 errors. Parse failures: 0.

## Reading

- **The quality loss is real, not noise.** Two bf16 runs of the same model (T7 on Oct 8, T9 on Oct 9) differed by at most 1 label and 0.02 profile micro-F1. FP8 loses 2 labels on n50 and about 0.05 profile micro-F1 on **both** slices. The multi-label profile tags look most sensitive to the lower weight precision.
- **Estimated cost:** emotion −0.02 average accuracy × 25 ≈ −0.5, profile ≈ −0.04 to −0.05 × 15 ≈ −0.6 to −0.75, replies ≈ −0.1. Total **≈ −1.2 to −1.4 points**.
- **Gain:** depends on the latency formula, which is still unknown. If latency is scored relative to other teams, a 25% faster mean could outweigh the loss. If it is a threshold we already pass at 1.55 s, FP8 only costs points.

## Decision

- **Gate failed** (rule: accuracy within 1 label and ROUGE-L within 0.01 on both slices).
- **Submission #2 stays** unless the organisers' answer on the latency formula makes FP8 worth about 1.3 points. If it does: copy line 21 of `start_fp8.sh` into `start.sh`, re-save the image, run the T9 checks, submit #3 by Oct 16.

## Files

Outputs on Migu: `/mnt/storage/out/fp8/` (`submission.jsonl`, `performance_report.json`, `eval_n50.json`, `eval_n50b.json`, `vllm_server.log`).
