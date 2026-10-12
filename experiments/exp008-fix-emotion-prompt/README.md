# EXP-008: Ismail M.'s emotion-focused prompt (non-thinking)

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail M. (recorded by Ismail F. from his report) |
| **Goal** | A prompt focused on fixing emotion accuracy |
| **Status** | Done. Not adopted; useful as the empty-profile reference. |

## Setup

`Qwen/Qwen3-8B:nscale`, HF router, `--modes nothink`, n50. Prompt: "emotion-focused modified prompt", English (same opening as EXP-007). Greedy or sampled: not stated.

## Results (n50)

| Metric | V0 (EXP-005) | EXP-008 |
|---|---|---|
| Valid | ✅ | ✅ |
| Emotion accuracy | 0.50 | **0.52** |
| Emotion macro-F1 | **0.194** | 0.141 |
| Emotion weighted-F1 | **0.497** | 0.424 |
| Profile micro-F1 | **0.275** | 0.000 |
| Profile macro-F1 | **0.328** | 0.000 |
| Profile exact match | 0.08 | **0.373** |

Top confusions: **fear→anxiety ×5, anger→anxiety ×3, relaxed→anxiety ×2, helplessness→anxiety ×2**, anxiety→relaxed ×2. Always-anxiety baseline 0.32 (delta +0.20).

## Conclusion

- Emotion +1 sample vs V0 (inside the ±7-point noise), reached by pushing labels toward anxiety: macro-F1 drops to the lowest of all variants. Same failure mode as V1's base-rate line.
- Profile micro-F1 0.000 and exact 0.373 = **every profile is empty** (0.373 is exactly the all-empty baseline on n50). This is the cleanest real-run evidence for the open question: if the official profile metric is exact match, empty profiles (0.373) beat V1.7/V1.8 (0.320 on n50); if F1, V1.8 (0.373 micro-F1) wins by a mile.
- Not adopted: V1.8 stays the prompt. Emotion gains have to come from fine-tuning (V2).

**Archive note (Oct 12):** this run was done on Ismail M.'s machine and the exact prompt text and run folder were never pushed. The results above are recorded from his report; they did not feed into any adopted version, so the gap doesn't affect any decision.
