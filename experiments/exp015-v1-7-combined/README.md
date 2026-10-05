# EXP-015: V1.7 = V0 + V1.6 profile rules + tag filter

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Combine V0's emotion prompt with the profile wins. |
| **Part of** | [EXP-005 prompt series](../exp005-prompt/README.md) (version table + all prompts there) |
| **Status** | Done. Best overall on n50; emotion within noise of V0. Needed confirmation on an untouched slice → EXP-016. |

## Setup

`Qwen/Qwen3-8B:nscale` via the Hugging Face router, non-thinking, greedy (temperature 0), val n50. Prompt = `participant_v1_x/run_inference.py` (only `SYSTEM_PROMPT` / `normalize_prediction` differ from `participant/`; the exact text is in the EXP-005 README).

```powershell
python scripts\exp3_compare.py --participant participant_v1_7 `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v1_7 --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

Run outputs (submission, raw outputs, performance report): `experiments/exp005-prompt/v1_7/`

## Results

| Metric | V0 | V1.6 | V1.7 |
|---|---|---|---|
| Emotion accuracy | **0.50** | 0.44 | 0.46 |
| Emotion macro-F1 | 0.194 | 0.201 | **0.204** |
| Profile micro-F1 (avg) | 0.275 | 0.335 | **0.373** |
| Profile exact match (avg) | 0.067 | 0.313 | **0.320** |
| — personality / interests / style micro-F1 | 0.27 / 0.36 / 0.19 | 0.29 / 0.46 / 0.25 | 0.36 / **0.62** / 0.14 |
| BLEU-4 | 0.021 | 0.023 | **0.025** |
| ROUGE-L | 0.176 | 0.175 | **0.178** |
| Reply median (chars) | 94 | 97 | 95 |
| Latency (s, API — unreliable) | 2.42 | 2.95 | 1.65 |

Empty predictions (personality/interests/style, of 50): V0 2/4/10, V1.6 17/30/28, V1.7 3/20/9.
Top confusions V1.7: gratitude→relaxed ×4, fear→anxiety ×4, anxiety→loneliness ×3. No `care` problem.

The emotion part of the prompt is identical to V0, yet 10 of 50 labels changed (1 wrong→right, 3 right→wrong, 6 wrong→a different wrong; checked with analyze.py --compare): changing other instructions shifts greedy emotion output. −4 points = 2 samples, inside the ±7 noise. Estimated contest points vs V0: emotion −1.0 (if real), profile +1.5 (F1) or +3.8 (exact match), replies ≈ 0 → net positive. Profile rules and tag filter were chosen on n50, so V1.7 was re-tested on an untouched slice (EXP-016).

## Files

- `metrics.json`: copy of `experiments/exp005-prompt/v1_7/out_nothink/metrics.json`
