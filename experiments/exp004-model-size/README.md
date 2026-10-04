# EXP-004: Qwen3-8B vs Qwen3-14B (non-thinking)

| | |
|---|---|
| **Date** | 2026-10-04 |
| **Author** | Ismail F. |
| **Goal** | Is 14B worth its cost over 8B, with the official prompt? |
| **Status** | ✅ Done on 50 val points. **Decision: Qwen3-8B, non-thinking.** |

## Setup

Same script, data, provider and decoding as [EXP-003](../exp003-qwen3-14b-thinking/README.md); only the model changes. The 14B row is EXP-003's non-thinking run.

```powershell
python scripts\exp3_compare.py --participant <package>\participant `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl `
  --scripts scripts --out experiments\exp004-model-size\qwen3-8b `
  --soft-switch --key-env HF_TOKEN --modes nothink `
  --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

A Qwen2.5-7B row was planned and dropped (the `qwen2.5-7b/` run never started).

## Results (50 val points)

| Metric | Qwen3-8B | Qwen3-14B | Baseline |
|---|---|---|---|
| Emotion accuracy | 0.46 | 0.46 | 0.32 (always anxiety) |
| Emotion macro-F1 | 0.15 | 0.19 | |
| Profile micro-F1 (avg) | 0.29 | 0.30 | 0 |
| Profile exact match (avg) | 0.06 | 0.11 | 0.37 (always empty) |
| Style micro-F1 | **0.21** | 0.11 | |
| Avg latency (API) | 2.0 s | 2.7 s | |
| Reply length median (chars) | 96 | 102 | gold 146 |
| Blank predictions | 0 | 1 | |

Files: `qwen3-8b/out_nothink/` (`submission.jsonl`, `metrics.json`, `raw_outputs.jsonl`, `performance_report.json`), `qwen3-8b/comparison.md`.

## Observations

1. **A statistical tie.** Same emotion accuracy; the macro-F1 and exact-match gaps come from a handful of samples (one sample = 2 accuracy points; standard error ≈ ±7 points). The two models agree on 33 of 50 emotion labels.
2. **Both share the same weaknesses:** profile fields over-filled, short generic replies, fear→anxiety confusion. These are what the prompt (B5) and fine-tuning (I2) fix, not model size.

## Decision: Qwen3-8B

| | 8B | 14B |
|---|---|---|
| Latency (10 pts) | ~1.7× faster on the same GPUs (decoding speed scales with size) | slower |
| QLoRA fine-tuning | ~half the training time → more attempts before Oct 18 | each failed run costs ~a day |
| GPU memory (bf16) | ~16 GB: fits one 4090 with room for long contexts | ~30 GB: needs both GPUs |
| Docker image | ~16 GB of weights | ~30 GB |

Fine-tuning on 17.7k examples will matter far more than the base size, and the smaller model lets us iterate more. 14B stays a fallback: if the 8B fine-tune is submitted by ~Oct 12, the same pipeline can be rerun on 14B (only the model name changes) and submitted only if it clearly wins on val with acceptable latency.
