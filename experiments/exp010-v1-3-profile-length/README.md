# EXP-010: V1.3 = V0 + EXP-009 profile rules + length rule

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Does V0 + Ismail M.'s profile and length rules beat V0? |
| **Part of** | [EXP-005 prompt series](../exp005-prompt/README.md) (version table + all prompts there) |
| **Status** | Done. Rejected: emotion −8 points; profile rules help interests only; length rule is the prime suspect (tested in EXP-011). |

## Setup

`Qwen/Qwen3-8B:nscale` via the Hugging Face router, non-thinking, greedy (temperature 0), val n50. Prompt = `participant_v1_x/run_inference.py` (only `SYSTEM_PROMPT` / `normalize_prediction` differ from `participant/`; the exact text is in the EXP-005 README).

```powershell
python scripts\exp3_compare.py --participant participant_v1_3 `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v1_3 --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

Run outputs (submission, raw outputs, performance report): `experiments/exp005-prompt/v1_3/`

## Results

| Metric | V0 | V1.3 |
|---|---|---|
| Emotion accuracy | **0.50** | 0.42 |
| Emotion macro-F1 | **0.194** | 0.153 |
| Profile micro-F1 (avg) | **0.275** | 0.267 |
| Profile exact match (avg) | 0.067 | **0.140** |
| — personality / interests / style micro-F1 | 0.27 / 0.36 / 0.19 | 0.23 / **0.47** / 0.11 |
| BLEU-4 | 0.021 | **0.055** |
| ROUGE-L | 0.176 | **0.184** |
| Reply length median (chars) | 94 | **157** (gold 146) |
| Latency (s, API) | 2.42 | 2.34 |

Reference: empty-profile exact match on these 50 = 0.373 (beats both).
Top confusions V1.3: anxiety→loneliness ×4 (V0 ×2), gratitude→relaxed ×3, fear→anxiety ×2.

EXP-009, Ismail M.'s own run (reported, not re-run here): acc 0.46, macro-F1 0.196, profile micro-F1 0.329, exact 0.273, style micro-F1 0.252, reply median 131.5, latency 3.57 s.

**V1.3 fails the rule** (needs ≥ +6 accuracy, or equal accuracy with clearly higher macro-F1): −8 accuracy, lower macro-F1. The emotion text did not change, so the drop comes from the added instructions (or noise: 8 points = 4 samples, at the edge of ±7). The length rule is the clear win: BLEU-4 ×2.6, reply length matches gold. The profile rules help interests but hurt style and personality.

## Files

- `metrics.json`: copy of `experiments/exp005-prompt/v1_3/out_nothink/metrics.json`
