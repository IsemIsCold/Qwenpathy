# EXP-006: Qwen3-8B thinking mode (V0-style prompt)

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail M. (run on his machine; recorded by Ismail F. from his report) |
| **Goal** | Re-check thinking vs non-thinking on Qwen3-8B (EXP-003 checked it on 14B) |
| **Status** | Done. **Run invalid** — thinking mode stays off. |

## Setup

`Qwen/Qwen3-8B:nscale` via the Hugging Face router, `--modes think`, 50 val points (n50). Prompt: reported as "identical to EXP-005 V0", but the snippet in the report starts "You are an emotional-companion assistant…" (English), while the official V0 prompt is Chinese. Greedy or sampled: not stated.

## Results

Validation **FAILED**: 50 lines, 0 JSON errors, 1 bad label, 1 structural violation, 1 parse failure.
- line 1: `response_text` must be a non-empty string
- line 1: invalid emotion_label `''`

eval.py refuses invalid submissions, so there are no scores.

## Conclusion

Thinking mode produced an empty answer on the first sample, and EXP-003 already showed no accuracy gain (0.42 vs 0.46 on 14B) at 2.2× the latency. **Thinking stays off. Not worth a rerun.** (V1.8's emotion alias map would turn a blank label into `anxiety`, but a blank reply still scores 0.)

**Archive note (Oct 12):** this run was done on Ismail M.'s machine and the exact prompt text and run folder were never pushed. The results above are recorded from his report; they did not feed into any adopted version, so the gap doesn't affect any decision.
