# EXP-007: Ismail M.'s prompt ablation (non-thinking)

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail M. (recorded by Ismail F. from his report) |
| **Goal** | Test a modified English prompt against V0 |
| **Status** | Done. Not adopted. |

## Setup

`Qwen/Qwen3-8B:nscale`, HF router, `--modes nothink`, n50. Prompt: "modified prompt variant", English, starting "You are an emotional-companion assistant. Output ONLY a single JSON object…" with the emotion list including `""`. Greedy or sampled: not stated.

## Results (n50)

| Metric | V0 (EXP-005) | EXP-007 |
|---|---|---|
| Valid | ✅ | ✅ |
| Emotion accuracy | **0.50** | 0.48 |
| Emotion macro-F1 | 0.194 | **0.214** |
| Emotion weighted-F1 | **0.497** | 0.457 |
| Profile micro-F1 | **0.275** | 0.234 |
| Profile macro-F1 | 0.328 | 0.275 |
| Profile exact match | 0.08 | **0.34** |

Top confusions: gratitude→relaxed ×3, fear→anxiety ×3, relaxed→sadness ×2, neutral→relaxed ×2, helplessness→sadness ×2. Always-anxiety baseline 0.32 (delta +0.16).

## Conclusion

Emotion −1 sample vs V0 (noise). Profile exact match up because it leaves more fields empty, but micro-F1 down. Superseded by V1.7/V1.8, which keeps V0's emotion and gets micro-F1 0.373 / exact 0.32 on n50.

**Archive note (Oct 12):** this run was done on Ismail M.'s machine and the exact prompt text and run folder were never pushed. The results above are recorded from his report; they did not feed into any adopted version, so the gap doesn't affect any decision.
