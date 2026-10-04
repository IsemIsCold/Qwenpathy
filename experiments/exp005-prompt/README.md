# EXP-005: Prompt versions (B5)

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Does a better system prompt beat the official one on Qwen3-8B (non-thinking)? |
| **Status** | V0 and V1 done. **Decision: submission #1 uses V0 (official prompt).** V1.2 next. |

## Setup

Same for every version: `Qwen/Qwen3-8B:nscale` via the Hugging Face router, `--greedy` (temperature 0, so differences come from the prompt, not randomness), `--soft-switch`, 50 val points. Each version is a copy of `participant/` with only `SYSTEM_PROMPT` changed (`participant_vN/`, ignored by git; the prompt text is in this README).

```powershell
python scripts\exp3_compare.py --participant participant_v1 `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v1 --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

## Versions

| Version | Change vs the official prompt |
|---|---|
| V0 | none (official prompt, sorted emotion list) |
| V1 | JSON order emotion_label → user_profile → response_text → memory_refs; "decide emotion_label FIRST from the latest user message"; base-rate line "most messages are anxiety or relaxed; choose a rarer label only when it clearly fits"; one-line definitions for anxiety, fear, helplessness, sadness, loneliness, relaxed, gratitude, joy, neutral, mixed |

## Results

V0 on all 50 points: emotion accuracy **0.50**, macro-F1 0.19, profile micro-F1 0.28, exact match 0.07, 0 parse failures, 2.4 s/sample (API). Greedy decoding is +4 points over the sampled EXP-004 run (0.46).

V1 stopped after 40 points (API error). Compared on the same 40 points:

| Metric | V0 | V1 |
|---|---|---|
| Emotion accuracy | 0.475 | 0.475 |
| Emotion macro-F1 | **0.198** | 0.152 |
| Profile micro-F1 | **0.274** | 0.254 |
| Profile exact match | **0.067** | 0.050 |
| ROUGE-L | 0.179 | 0.180 |
| Reply median length (chars) | 92 | 82 (gold 146) |

Top confusions: V0 gratitude→relaxed ×3, helplessness→sadness ×2, fear→anxiety ×2. V1 gratitude→relaxed ×4, **fear→anxiety ×4**, **relaxed→anxiety ×3**.

## Conclusion

V1 is no better on accuracy and worse on rare emotions. The base-rate line pushed the model *towards* anxiety (fear→anxiety doubled, relaxed→anxiety appeared). Two changes were tested together, so the definitions' own effect is unknown.

**Submission #1 = V0.** Next: V1.2 = V1 without the base-rate line, all 50 points.
