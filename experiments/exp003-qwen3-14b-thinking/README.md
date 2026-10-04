# EXP-003: Qwen3-14B, thinking vs non-thinking

| | |
|---|---|
| **Date** | 2026-10-04 |
| **Author** | Ismail F. |
| **Goal** | Does Qwen3's thinking mode give better labels than non-thinking mode with the official prompt, and how much slower is it? |
| **Status** | ✅ Done on 50 val points. **Decision: no thinking mode.** The 14B vs 8B question continues in [EXP-004](../exp004-model-size/README.md). |

## Setup

| Item | Value |
|---|---|
| Script | [`scripts/exp3_compare.py`](../../scripts/exp3_compare.py): official prompt, history format and parser; validate + eval per mode |
| Model / provider | `Qwen/Qwen3-14B:nscale` via the Hugging Face router, streaming |
| Thinking switch | `--soft-switch`: Qwen3's `/think` / `/no_think` tag appended to the user message |
| Decoding | Qwen's recommended sampling per mode, seed 42 |
| Data | `data/val_input_n50.jsonl` / `val_gold_n50.jsonl` (first 50 val points) |
| Cost | part of the $5 Hugging Face credit; 0 Migu card-hours |

```powershell
python scripts\exp3_compare.py --participant <package>\participant `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl `
  --scripts scripts --out experiments\exp003-qwen3-14b-thinking `
  --soft-switch --key-env HF_TOKEN --modes nothink think `
  --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-14B:nscale"
```

## Results (50 val points)

| Metric | Non-thinking | Thinking | Baseline |
|---|---|---|---|
| Emotion accuracy | **0.46** | 0.42 | 0.32 (always anxiety) |
| Emotion macro-F1 | **0.19** | 0.15 | |
| Profile micro-F1 (avg) | 0.30 | **0.37** | 0 |
| Profile exact match (avg) | 0.11 | **0.18** | 0.37 (always empty) |
| Style micro-F1 | 0.11 | 0.16 | |
| Avg latency (API) | **2.7 s** | 5.9 s | |
| Reasoning length | 0 | ~835 chars avg (max ~1,900) | |
| Reply length median (chars) | 102 | 98 | gold 146 |
| Blank / invalid predictions | 1 | 0 | |

Files: `out_nothink/` and `out_think/` (`submission.jsonl`, `metrics.json`, `performance_report.json`; `out_think/raw_outputs.jsonl` has the reasoning). `comparison.md` is the script's auto-generated table.

## Observations

1. **Thinking doesn't help emotion.** It's 4 points lower on accuracy (within noise: on 50 points the standard error is about ±7 points) and lower on macro-F1.
2. **Thinking helps the profile a little** (exact match 0.18 vs 0.11), but both are far below the "always empty" baseline of 0.37. Both modes fill nearly every field; gold leaves many empty. That's a prompt/fine-tuning problem, not a reasoning problem.
3. **Thinking costs 2.2× the latency**, and latency is 10 official points. Its reasoning (~835 characters per sample) would also have to fit in the token budget.
4. **Common emotion mistakes in both modes:** fear → anxiety, relaxed ↔ gratitude, helplessness → sadness.
5. **Replies are too short and generic** (median ~100 characters vs 146 for gold), which costs BLEU/ROUGE points.
6. Non-thinking had one blank prediction (`val_conv_000003_t7`): the validator refused the file, so `exp3_compare.py` now scores with `--force` (blanks count as wrong, as in the official scoring) and reports blanks in the table.

## Conclusion

**Drop thinking mode.** It's slower with no gain on the 25 emotion points. The profile gain is better pursued with the prompt (B5) and fine-tuning (I2).

## History

- Oct 4 (morning): first attempt stopped after 9 points when the free API credit ran out (0.44 accuracy on 9 points). Superseded by this run.
