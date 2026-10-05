# EXP-005: Prompt versions (B5)

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Does a better system prompt beat the official one on Qwen3-8B (non-thinking)? This folder is the overview + all run outputs; each later experiment also has its own folder (EXP-006 to EXP-016). |
| **Status** | V0, V1, V1.3–V1.7 done (V1.2 built, not run). Emotion prompt work stopped (no variant beats V0). **V1.7 confirmed on n50b (EXP-016). Submission #2 = V1.8 (= V1.7 + emotion alias map).** Next: V2 (fine-tuning). |

## Setup

Same for every version: `Qwen/Qwen3-8B:nscale` via the Hugging Face router, `--greedy` (temperature 0, so differences come from the prompt, not randomness), `--soft-switch`, 50 val points. Each version is a copy of `participant/` with only `SYSTEM_PROMPT` changed (`participant_vN/`, ignored by git; the prompt text is in this README).

```powershell
python scripts\exp3_compare.py --participant participant_v1 `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v1 --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

## Versions

| Version | Change vs the official prompt | Run |
|---|---|---|
| V0 | none (official Chinese prompt, sorted emotion list) | EXP-005 (re-run by Ismail M., same acc/macro-F1) |
| V1 | JSON order emotion_label → user_profile → response_text → memory_refs; "decide emotion_label FIRST from the latest user message"; base-rate line "most messages are anxiety or relaxed…"; one-line definitions for 10 emotions | EXP-005 (40 pts) |
| V1.2 | V1 without the base-rate line (`participant_v1_2/`) | built, **not run** |
| — | Ismail M.'s own prompts: thinking mode, ablation, emotion-focused | [EXP-006](../exp006-think-8b/README.md), [EXP-007](../exp007-prompt-ablation/README.md), [EXP-008](../exp008-fix-emotion-prompt/README.md) |
| — | Ismail M.'s English prompt: 16 emotion definitions, "do NOT default to anxiety", "DO NOT output empty lists by default", length rule (~200 chars, ≥180) | [EXP-009](../exp009-profile-fix-prompt/README.md) (his run) |
| V1.3 | V0 + EXP-009's profile rules + length rule, verbatim; V0's "prefer empty arrays" line removed | [EXP-010](../exp010-v1-3-profile-length/README.md) |
| V1.4 | V0 + length rule only | [EXP-011](../exp011-v1-4-length-rule/README.md) |
| V1.5 | V1.4 + V1's field order | [EXP-012](../exp012-v1-5-emotion-first/README.md) |
| — | offline: drop over-predicted tags from V0's output | [EXP-013](../exp013-profile-tag-filter/README.md) |
| V1.6 | Ismail M.'s 2nd prompt + 3 fixes (no `""` emotion, profile vocab added, memory_refs []) | [EXP-014](../exp014-v1-6-ismail-m-prompt/README.md) |
| V1.7 | V0 + V1.6's "Leave [] if insufficient information" profile lines + over-predicted tag filter in `normalize_prediction` | [EXP-015](../exp015-v1-7-combined/README.md) (n50), [EXP-016](../exp016-n50b-confirmation/README.md) (n50b) |
| **V1.8** | V1.7 + `EMOTION_ALIASES` (invented labels → closest official label, fallback anxiety). Parser-only, prompt = V1.7. `participant_v1_8/` | **frozen prompt; submission #2 fallback** |

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

### EXP-010: V1.3 vs V0 (all 50 points, greedy)

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

### EXP-011: V1.4 vs V0 (all 50 points, greedy)

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

### EXP-012: V1.5 vs V0 (all 50 points, greedy)

| Metric | V0 | V1.4 | V1.5 |
|---|---|---|---|
| Emotion accuracy | **0.50** | 0.44 | 0.42 |
| Emotion macro-F1 | **0.194** | 0.195 | 0.150 |
| Profile micro-F1 (avg) | **0.275** | 0.224 | 0.243 |
| BLEU-4 | 0.021 | 0.051 | 0.048 |
| ROUGE-L | 0.176 | 0.174 | 0.177 |
| Reply median (chars) | 94 | 164 | 154 |
| Latency (s, API) | 2.42 | — | 2.61 |

**Rejected; hypothesis refuted.** Putting emotion_label first did not recover accuracy (0.42). The length instruction itself hurts emotion, wherever the reply sits in the JSON. Long replies also cost latency (+~60% output tokens) and do not move ROUGE-L. **The length rule is dropped.**

Summary of EXP-005: 5 prompt variants (V1, V1.3, V1.4, V1.5, EXP-009), none beat V0 on emotion accuracy. Differences are within the ±7-point noise of 50 points. Further emotion gains need fine-tuning, not prompt wording.

### EXP-013 (offline, no API): profile tag filter on V0's output

analyze.py showed V0 over-predicts three tags (V0 count vs gold count on n50): personality `emotionally_stable` 47 vs 2, interests `study_exam` 36 vs 0, style `colloquial` 38 vs 3. Removing them from V0's existing predictions, scored with eval.py:

| | micro-F1 avg | exact match avg | personality / interests / style micro-F1 |
|---|---|---|---|
| V0 as is | 0.275 | 0.067 | 0.27 / 0.36 / 0.19 |
| **drop 3 tags** | **0.360** | **0.240** | 0.36 / 0.51 / 0.21 |
| all-empty profile | 0.000 | 0.373 | — |

+8.5 points micro-F1 and +17 points exact match, with no API call, no emotion risk and no latency cost. **Caveat:** found and measured on the same 50 points, so it must be confirmed on a different val slice before shipping.

### EXP-014: V1.6 (Ismail M.'s English prompt, 3 fixes) vs V0

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

### EXP-015: V1.7 vs V0 vs V1.6

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

### EXP-016: V0 vs V1.7 on n50b (independent slice)

n50b = 50 random val points (seed 42) from conversations not used in n50 (`convert_val.py --exclude data/val_gold_n50.jsonl --sample 50 --seed 42`). Nothing was tuned on it.

| Metric | V0 | V1.7 |
|---|---|---|
| Emotion accuracy | 0.60 | **0.62** |
| Emotion macro-F1 | 0.366 | **0.374** |
| Profile micro-F1 (avg) | 0.175 | **0.300** |
| Profile exact match (avg) | 0.107 | **0.247** (all-empty: 0.593) |
| — personality / interests / style micro-F1 | 0.14 / 0.24 / 0.15 | 0.20 / 0.43 / 0.27 |
| BLEU-4 | 0.068 | 0.068 |
| ROUGE-L | **0.205** | 0.200 |
| Reply median (chars) | 108 | 99 (gold 101) |
| Invalid emotion labels (blank) | 3 | 3 |

**V1.7 confirmed.** Emotion equal (31 vs 30 correct), the profile gain holds on unseen data (+12.5 micro-F1, +14 exact match), replies equal. n50b is easier than n50 for both (V0 0.60 vs 0.50): always compare versions on the same slice.

Both versions lose 3/50 samples to invented emotion labels (`frustration`, `annoyance`, `hopefulness`) that the parser turns into a blank = always wrong. Fix = **V1.8** (`participant_v1_8/`) = V1.7 + `EMOTION_ALIASES` in `normalize_prediction`, which maps common inventions to the closest official label, otherwise falls back to `anxiety`. Re-parsing V0's n50b raw outputs with it: 30 → 31 correct. It cannot make a valid label worse.

Open: an all-empty profile still has the highest exact match (0.593 on n50b). If the official profile metric is exact match, empty profiles win; if F1, V1.7 wins. The organisers' answer decides.

## Conclusion

V1 is no better on accuracy and worse on rare emotions. The base-rate line pushed the model *towards* anxiety (fear→anxiety doubled, relaxed→anxiety appeared). Two changes were tested together, so the definitions' own effect is unknown.

**Submission #1 = V0. Submission #2 = V1.8** = V1.7 (V0 emotion prompt + "Leave [] if insufficient" profile rules + over-predicted tag filter) + emotion alias map. Profile decision waits on the organisers' answer (F1 vs exact match).
