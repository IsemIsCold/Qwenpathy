# EXP-014: V1.6 = Ismail M.'s 2nd prompt + 3 fixes

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Test Ismail M.'s second prompt after fixing an empty emotion option, the missing profile vocabulary and memory_refs. |
| **Part of** | [EXP-005 prompt series](../exp005-prompt/README.md) (version table + all prompts there) |
| **Status** | Done. Emotion rejected (care definition → care over-predicted); profile rules kept for V1.7. |

## Setup

`Qwen/Qwen3-8B:nscale` via the Hugging Face router, non-thinking, greedy (temperature 0), val n50. Prompt = `participant_v1_x/run_inference.py` (only `SYSTEM_PROMPT` / `normalize_prediction` differ from `participant/`; the exact text is in the EXP-005 README).

```powershell
python scripts\exp3_compare.py --participant participant_v1_6 `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v1_6 --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

Run outputs (submission, raw outputs, performance report): `experiments/exp005-prompt/v1_6/`

## Results

V1.6 = Ismail M.'s prompt with three fixes: removed `""` from the emotion list and "empty string allowed for emotion_label"; added the official profile vocabulary ({personality}/{interests}/{styles}; the original had none, so most tags would have been dropped as invalid); memory_refs fixed to []. Definitions for 5 emotions (anxiety, fear, gratitude, relaxed, care), "Leave [] if insufficient information" per profile field, no length rule.

| Metric | V0 | V1.6 | V1.6 + tag filter (offline) |
|---|---|---|---|
| Emotion accuracy | **0.50** | 0.44 | 0.44 |
| Emotion macro-F1 | 0.194 | **0.201** | 0.201 |
| Profile micro-F1 (avg) | 0.275 | 0.335 | **0.383** |
| Profile exact match (avg) | 0.067 | 0.313 | **0.393** (all-empty: 0.373) |
| BLEU-4 | 0.021 | 0.023 | 0.023 |
| ROUGE-L | 0.176 | 0.175 | 0.175 |
| Reply median (chars) | 94 | 97 | 97 |
| Latency (s, API) | 2.42 | 2.95 | 2.95 |

Empty predictions (of 50), V0 → V1.6: personality 2 → 17, interests 4 → 30, style 10 → 28.
Top confusions V1.6: gratitude→care ×4, fear→anxiety ×4, anxiety→care ×3. V1.6 predicts `care` 10× (gold 1×).

**Emotion: rejected** (−6 accuracy). The `care` definition ("warm concern for another person") pulls gratitude and anxiety into care: 7 errors.
**Profile: big win.** The "Leave [] if insufficient information" rules triple the empty predictions and raise exact match from 0.07 to 0.31; with the tag filter it is the first variant to beat the all-empty baseline on exact match (0.393 vs 0.373) while also having the best micro-F1.

V1.7 (EXP-015) = V0 prompt (emotion part unchanged, no definitions) + V1.6's four profile-rule lines + tag filter in `normalize_prediction` (`OVERPREDICTED_TAGS`). `participant_v1_7/`, output `v1_7/`.

## Files

- `metrics.json`: copy of `experiments/exp005-prompt/v1_6/out_nothink/metrics.json`
