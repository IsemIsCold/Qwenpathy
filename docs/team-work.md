# Who built what

> Last updated: **2026-10-12**. What each person delivered and where it lives. Commands for every tool: [how-to.md](how-to.md). Task history with owners: [TASKS.md](../TASKS.md).

## How the pieces fit

```
organiser data ──► convert_val.py (Zakaria) ──► fixed val slices n50 / n50b ──► validate.py + eval.py (Yahya) ──► every decision
       │
       └──► build_sft.py + check_sft.py ──► QLoRA on Migu ──► merged weights ──► participant/ (start.sh + run_inference.py, V1.8 prompt) ──► submissions
                     ▲                         (Ismail F.)                                  ▲
                     └──────────── one prompt, shared by training and inference ────────────┘
                                   (EXP-005–016: Ismail F. + Ismail M.)
```

---

## Zakaria: dataset

| Delivered | File | Status |
|---|---|---|
| Val set → test-format input + gold labels (`--n`, `--exclude`, `--sample`) | [`scripts/convert_val.py`](../scripts/convert_val.py) | done (B2) |
| Every JSONL format we use | [`docs/data-format.md`](data-format.md) | done; §4, §6, §7 updated Oct 12 (Ismail F.) |
| Dataset statistics: labels, profiles, lengths | [`scripts/compute_stats.py`](../scripts/compute_stats.py) → [`docs/data-stats.md`](data-stats.md) | done (B6); rewritten Oct 12 with real token counts (Ismail F.) |
| Training-data rules (split by conversation, history truncation) | spec for `train/build_sft.py` | done (I1) |

Later tasks Z1–Z4 were not delivered. Z2 (stats script with real tokens) and Z3 (format doc update) were done by Ismail F. on Oct 12. Z1 (inference truncation) turned out not to be needed, since submission #2 ran on the real test set without errors, and Z4 (augmentation research) became moot with no second training run. Details in TASKS.md.

---

## Yahya: evaluation

| Delivered | File | Status |
|---|---|---|
| Strict format checker (fails closed) | [`scripts/validate.py`](../scripts/validate.py) | done (B3) |
| Scorer: emotion accuracy, macro-F1, per-class table, confusions, profile micro/macro-F1 + exact match, baselines, reply length, char-level BLEU-4 / ROUGE-L, BERTScore | [`scripts/eval.py`](../scripts/eval.py) | done (B4, I4) |
| Test suite for validate + eval | [`tests/run_tests.py`](../tests/run_tests.py) | done: 73/73 |
| Design notes and how to read the output | [`docs/eval-design.md`](eval-design.md), [`docs/interpreting-eval-output.md`](interpreting-eval-output.md) | done |
| Error-analysis tool, v2.2 and v3 ("Error Genome"), the base of the current v4.1 | [`scripts/analyze.py`](../scripts/analyze.py), [`tests/test_analyze.py`](../tests/test_analyze.py) | done; v4.1 = v3 + review fixes, 102/102 tests |
| Pre-flight gate for pilot runs, with 10 tests | `pilot_gate.py` (Oct 9, not merged) | reviewed Oct 10: needs fixes before use (one submission per slice, fail when a check can't run, validity check) |
| Research report on class imbalance (Oct 9) | — | reviewed and corrected against our data: [`docs/p2-imbalance-plan.md`](p2-imbalance-plan.md) |

Every score in this repo comes from Yahya's `eval.py`.

---

## Ismail F.: captain, training lead, Migu infrastructure, prompt

| Delivered | Where | Status |
|---|---|---|
| Base model and mode: official baseline, Qwen2.5-7B, Qwen3-14B thinking vs not, 8B vs 14B → **Qwen3-8B non-thinking** | EXP-001 to EXP-004 | done |
| API experiment runner (official prompt/parser + validate + eval in one command) and Windows setup | [`scripts/exp3_compare.py`](../scripts/exp3_compare.py), [`scripts/setup-api-env.ps1`](../scripts/setup-api-env.ps1) | done |
| Prompt work → **V1.8, frozen** | EXP-005, EXP-010 to EXP-016 | done (B5) |
| Migu setup, official vLLM image, **submission #1** | `participant/`, LOG.md | done (B7, B9) |
| Teacher test: DeepSeek V4.1-Flash | EXP-017 | done |
| SFT data rebuild with the frozen prompt + `check_sft.py` sign-off | [`train/build_sft.py`](../train/build_sft.py), [`train/check_sft.py`](../train/check_sft.py) | done (T1) |
| Migu training environment, eval harness, V1.8 baseline on vLLM | TASKS T2–T4 | done |
| Pilot QLoRA, first 2-GPU DDP run, pilot gate | TASKS T5, T6, E1 | done |
| **Full QLoRA run + merge**: emotion 0.54 → 0.80 / 0.62 → 0.84 | [EXP-019](../experiments/exp019-qlora-full/README.md) | done (T7) |
| Submission check (the exact organiser run path) → **submission #2** | TASKS T9, T10 | done (Oct 9) |
| Error review of all 17 remaining emotion errors → no second training run | [`docs/p1-error-review.md`](p1-error-review.md), [`docs/p2-imbalance-plan.md`](p2-imbalance-plan.md) | done (P1) |
| FP8 latency test | [EXP-020](../experiments/exp020-fp8/README.md) | done |
| Card-hour ledger, expert-review docs (README, integrity note, experiment index) | LOG.md, [`docs/integrity.md`](integrity.md), [`experiments/README.md`](../experiments/README.md) | done (Oct 12) |

---

## Ismail M.: prompts and training support

| Delivered | Where | Status |
|---|---|---|
| Base-model check: thinking mode on 8B | EXP-006 | done (invalid run → thinking stays off) |
| Four prompt variants; the profile-rule prompt is the origin of V1.8's "leave [] if insufficient information" rules | EXP-007, EXP-008, EXP-009, EXP-014 (V1.6) | done |
| Contribution to the T1 `build_sft.py` rebuild | `train/build_sft.py` | done (T1, with Ismail F.) |

The second training run (P2) he owned was dropped on Oct 11 after the error review.

---

## Elly: infrastructure support (joined Oct 8)

Owned E1–E5, the T9 check and organiser questions O1b. E1 (2-GPU DDP) was completed by Ismail F.'s pilot rerun on Oct 8; T9, the card-hour ledger, the latency test and the organiser questions were done by Ismail F. by Oct 12, and the remaining rows were closed (see TASKS.md).

---

## Where everything stands (Oct 12)

| Area | State |
|---|---|
| Data | Official train only; val used only for evaluation; fixed slices n50 / n50b / n100 tracked in git |
| Evaluation | validate + eval (73 tests), analyze v4.1 (102 tests) |
| Model | Qwen3-8B non-thinking + full QLoRA, merged bf16 |
| Prompt | V1.8, frozen, shared by training and inference |
| Submissions | 2 of 5 used. **#2 (Oct 9) = the full QLoRA model.** FP8 variant tested (EXP-020), submitted only if the latency formula makes it worth the quality loss |
| Compute | 4.95 of 30 card-hours left (Oct 12) |
