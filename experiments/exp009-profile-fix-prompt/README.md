# EXP-009: Ismail M.'s profile-fix prompt (non-thinking)

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail M. (recorded by Ismail F. from his report and his `Run inference.docx`) |
| **Goal** | Fix the empty/wrong user profiles with a profile-focused English prompt |
| **Status** | Done. Not adopted as a whole; its profile idea led to V1.6 → V1.7 → **V1.8**. |

## Setup

`Qwen/Qwen3-8B:nscale` via the HF router, `--modes nothink`, n50. Greedy or sampled: not stated.

Prompt (from `Run inference.docx`, reviewed Oct 5): English, replaces the whole official prompt.
- definitions for all 16 emotions + "Select ONE emotion label… Do NOT default to anxiety"
- profile rules: tags only with explicit evidence, "Only use empty list [] when there is ZERO supporting evidence", **"DO NOT output empty lists by default"**
- length rule: "around 200 Chinese characters… Do NOT make it shorter than 180"
- `response_text` first in the JSON; `{{ }}` correctly escaped; `SYSTEM_PROMPT = SYSTEM_PROMPT = """` (harmless double assignment)

**TODO (Ismail M.): push the run folder (submission, metrics, raw outputs).**

## Results (n50)

| Metric | V0 (EXP-005) | EXP-009 |
|---|---|---|
| Valid | ✅ | ✅ |
| Emotion accuracy | **0.50** | 0.46 |
| Emotion macro-F1 | 0.194 | **0.196** |
| Emotion weighted-F1 | **0.497** | 0.464 |
| Profile micro-F1 | 0.275 | **0.329** |
| Profile macro-F1 | **0.328** | 0.303 |
| Profile exact match | 0.08 | **0.273** |
| Style micro-F1 | 0.21 | **0.25** |
| Reply median (chars) | 94 | 131.5 (gold 146) |
| Latency (API) | 2.4 s | 3.57 s |

Top confusions: fear→anxiety ×3, relaxed→sadness ×2, helplessness→sadness ×2, gratitude→relaxed ×2, anxiety→loneliness ×2. Always-anxiety baseline 0.32 (delta +0.14).

## Conclusion

- Emotion −2 samples vs V0 (noise), not better.
- Profile clearly better (micro-F1 +5, exact +19 points) — the first sign that explicit profile rules help.
- The length rule makes replies longer (closer to gold) but costs latency, and EXP-011/012 later showed a length instruction lowers emotion accuracy → dropped.
- Follow-ups: EXP-010 (V1.3 = V0 + these profile + length rules) and EXP-014 (V1.6, his second prompt). The profile idea survives in **V1.8** in the softer form "Leave [] if insufficient information", plus a tag filter.
