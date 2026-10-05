# EXP-012: V1.5 = V1.4 + emotion-first field order

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Does choosing emotion_label before writing the long reply recover accuracy? |
| **Part of** | [EXP-005 prompt series](../exp005-prompt/README.md) (version table + all prompts there) |
| **Status** | Done. Rejected, hypothesis refuted: the length instruction hurts emotion wherever the reply sits. Length rule dropped for good. |

## Setup

`Qwen/Qwen3-8B:nscale` via the Hugging Face router, non-thinking, greedy (temperature 0), val n50. Prompt = `participant_v1_x/run_inference.py` (only `SYSTEM_PROMPT` / `normalize_prediction` differ from `participant/`; the exact text is in the EXP-005 README).

```powershell
python scripts\exp3_compare.py --participant participant_v1_5 `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v1_5 --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

Run outputs (submission, raw outputs, performance report): `experiments/exp005-prompt/v1_5/`

## Results

| Metric | V0 | V1.4 | V1.5 |
|---|---|---|---|
| Emotion accuracy | **0.50** | 0.44 | 0.42 |
| Emotion macro-F1 | **0.194** | 0.195 | 0.150 |
| Profile micro-F1 (avg) | **0.275** | 0.224 | 0.243 |
| BLEU-4 | 0.021 | 0.051 | 0.048 |
| ROUGE-L | 0.176 | 0.174 | 0.177 |
| Reply median (chars) | 94 | 164 | 154 |
| Latency (s, API) | 2.42 | — | 2.61 |

**Rejected; hypothesis refuted.** Putting emotion_label first did not recover accuracy (0.42). The length instruction itself hurts emotion, wherever the reply sits in the JSON. Long replies also cost latency (+~60% output tokens) and do not move ROUGE-L. **The length rule is dropped.**

Summary of EXP-005: 5 prompt variants (V1, V1.3, V1.4, V1.5, EXP-009), none beat V0 on emotion accuracy. Differences are within the ±7-point noise of 50 points. Further emotion gains need fine-tuning, not prompt wording.

## Files

- `metrics.json`: copy of `experiments/exp005-prompt/v1_5/out_nothink/metrics.json`
