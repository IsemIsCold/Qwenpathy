# EXP-017 (part 1): thinking-switch smoke test

| | |
|---|---|
| **Date** | 2026-10-07 |
| **Author** | Ismail F. |
| **Part of** | [EXP-017 teacher test](../exp017-teacher/README.md). This folder is its first 9-sample check, kept separate because it ran before the main sweep. |

**Goal:** before spending API credit on 50-point runs, check on 9 val points (`data/val_input_n9.jsonl`) whether the Hugging Face router respects the "thinking off" switch for DeepSeek-V4-Pro-0813.

**Result** ([comparison.md](comparison.md)): 7 of 9 answers failed to parse (the output did not start with the JSON object), emotion accuracy 0.11, 6.4 s per sample. V4-Pro on the HF router is not usable with our parser. The provider sweep in `exp017-teacher/switch/` confirmed this on every provider, and V4.1-Flash on novita was chosen instead.
