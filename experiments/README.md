# Experiments

One folder per experiment, each with a README (goal, setup, results, decision). Every score comes from `scripts/eval.py` on the fixed val slices (`data/val_*_n50.jsonl`, `data/val_*_n50b.jsonl`). The full results with numbers are in [LOG.md](../LOG.md) → *Experiments*.

## Phase 1: base model and prompt (API runs, Oct 3–5)

| # | Folder | Question | Outcome |
|---|---|---|---|
| EXP-001 | [exp001-baseline-qwen0.5b-cpu](exp001-baseline-qwen0.5b-cpu/README.md) | Does the official baseline run end to end? | Pipeline works; output unusable |
| EXP-002 | [exp002-qwen7b-api](exp002-qwen7b-api/README.md) | What does a 7B model give with the official prompt? | Correct format, generic replies |
| EXP-003 | [exp003-qwen3-14b-thinking](exp003-qwen3-14b-thinking/README.md) | Thinking vs non-thinking mode | Thinking no better, 2.2× slower → **off** |
| EXP-004 | [exp004-model-size](exp004-model-size/README.md) | Qwen3-8B vs 14B | Tie on emotion → **Qwen3-8B** |
| EXP-005 | [exp005-prompt](exp005-prompt/README.md) | Prompt versions V0–V1.8 (overview of EXP-005 to EXP-016; all prompt files in `versions/`) | No prompt beats V0 on emotion; V1.8 wins on profile |
| EXP-006 | [exp006-think-8b](exp006-think-8b/README.md) | Thinking mode on 8B | Invalid run → thinking stays off |
| EXP-007 | [exp007-prompt-ablation](exp007-prompt-ablation/README.md) | English prompt variant | Not adopted |
| EXP-008 | [exp008-fix-emotion-prompt](exp008-fix-emotion-prompt/README.md) | Emotion-focused prompt | Anxiety-biased, empty profiles → not adopted |
| EXP-009 | [exp009-profile-fix-prompt](exp009-profile-fix-prompt/README.md) | Profile-focused prompt | Profile idea kept (→ V1.8) |
| EXP-010 | [exp010-v1-3-profile-length](exp010-v1-3-profile-length/README.md) | V1.3: profile + length rules | Emotion worse → rejected |
| EXP-011 | [exp011-v1-4-length-rule](exp011-v1-4-length-rule/README.md) | V1.4: length rule only | Emotion worse → rejected |
| EXP-012 | [exp012-v1-5-emotion-first](exp012-v1-5-emotion-first/README.md) | V1.5: emotion field first | No help → rejected |
| EXP-013 | [exp013-profile-tag-filter](exp013-profile-tag-filter/README.md) | Drop 3 over-predicted profile tags (offline) | Profile micro-F1 0.275 → 0.360 → kept |
| EXP-014 | [exp014-v1-6-ismail-m-prompt](exp014-v1-6-ismail-m-prompt/README.md) | V1.6: "leave [] if unsure" profile rules | Profile rules kept, emotion part rejected |
| EXP-015 | [exp015-v1-7-combined](exp015-v1-7-combined/README.md) | V1.7 = V0 + profile rules + tag filter | Candidate |
| EXP-016 | [exp016-n50b-confirmation](exp016-n50b-confirmation/README.md) | V0 vs V1.7 on the untouched n50b slice | **V1.7 confirmed → V1.8 frozen** |

## Phase 2: fine-tuning and serving (Migu, Oct 7–12)

| # | Folder | Question | Outcome |
|---|---|---|---|
| EXP-017 | [exp017-teacher](exp017-teacher/README.md) (+ [exp017-switchtest](exp017-switchtest/README.md)) | Could a stronger API model (DeepSeek) be a data tool? | V4.1-Flash works; **not used** for training |
| EXP-018 | [exp018-data-research](exp018-data-research/README.md) | Data research for a second training run (Z4): reply length, the 350-char filter, augmentation | No training lever ≥ 1 point; found the profile tag filter deleting tags the model was trained to output |
| EXP-019 | [exp019-qlora-full](exp019-qlora-full/README.md) | QLoRA fine-tune: pilot (T5/T6) and full run (T7), plus the submission check (T9) | Emotion 0.54 → 0.80 / 0.62 → 0.84 → **submission #2** |
| EXP-020 | [exp020-fp8](exp020-fp8/README.md) | FP8 weights at inference for latency | 25% faster, but quality gate failed |
| EXP-021 | [exp021-ngram-spec](exp021-ngram-spec/README.md) | n-gram speculative decoding (lossless) for latency | Same quality, but 23% slower → rejected |
| EXP-022 | [exp022-emotion-bias](exp022-emotion-bias/README.md) | Anxiety-prior correction from label logprobs + tag filter off (no training) | Scripts ready and tested on a stub; GPU run Oct 13 |

Task-level steps without their own folder (T1 data build, T2 environment, T4 baseline on vLLM, T9 submission check, P1 error review) are logged in [LOG.md](../LOG.md) and [TASKS.md](../TASKS.md). The P1 review is in [docs/p1-error-review.md](../docs/p1-error-review.md).
