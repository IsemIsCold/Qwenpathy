# EXP-011: V1.4 = V0 + length rule only

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Isolate the length rule ("around 200 chars, not shorter than 180"). |
| **Part of** | [EXP-005 prompt series](../exp005-prompt/README.md) (version table + all prompts there) |
| **Status** | Done. Rejected: the length rule alone costs 6 accuracy points and does not move ROUGE-L. |

## Setup

`Qwen/Qwen3-8B:nscale` via the Hugging Face router, non-thinking, greedy (temperature 0), val n50. Prompt = `participant_v1_x/run_inference.py` (only `SYSTEM_PROMPT` / `normalize_prediction` differ from `participant/`; the exact text is in the EXP-005 README).

```powershell
python scripts\exp3_compare.py --participant participant_v1_4 `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v1_4 --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

Run outputs (submission, raw outputs, performance report): `experiments/exp005-prompt/v1_4/`

## Results

| Metric | V0 | V1.4 |
|---|---|---|
| Emotion accuracy | **0.50** | 0.44 |
| Emotion macro-F1 | 0.194 | 0.195 |
| Profile micro-F1 (avg) | **0.275** | 0.224 |
| Profile exact match (avg) | **0.067** | 0.040 |
| BLEU-4 | 0.021 | **0.051** |
| ROUGE-L | **0.176** | 0.174 |
| Reply length median (chars) | 94 | 164 (gold 146) |

Top confusions V1.4: anxiety→loneliness ×4, gratitude→relaxed ×2, relaxed→loneliness ×2, fear→anxiety ×2.

**Rejected.** The length rule alone costs 6 accuracy points and profile, and ROUGE-L does not move; only BLEU-4 improves. Same drop and same new anxiety→loneliness confusion as V1.3, so the cause is the long reply, not the profile rules. Hypothesis: the model writes ~160 comforting characters *before* choosing emotion_label and then labels its own reply ("loneliness"). EXP-012 tests this by putting emotion_label first.

## Files

- `metrics.json`: copy of `experiments/exp005-prompt/v1_4/out_nothink/metrics.json`
