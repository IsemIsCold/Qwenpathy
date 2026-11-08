# Who built what

> Last updated: **2026-11-08**. What each person delivered and where it lives. Commands for every tool: [how-to.md](how-to.md). Task history with owners: [TASKS.md](../TASKS.md).

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
| Every JSONL format we use | [`docs/data-format.md`](data-format.md) | done; §4, §6, §7 updated Oct 12 and §6 (tune500, t10, EXP-029 splits) Nov 8 (Ismail F.) |
| Dataset statistics: labels, profiles, lengths | [`scripts/compute_stats.py`](../scripts/compute_stats.py) → [`docs/data-stats.md`](data-stats.md) | done (B6); rewritten Oct 12 with real token counts (Ismail F.) |
| Training-data rules (split by conversation, history truncation) | spec for `train/build_sft.py` | done (I1) |

Later tasks Z1–Z5 were not delivered by Zakaria. Ismail F. did Z1 (inference truncation, opt-in and off in every submission), Z2 (stats script with real tokens), Z3 (format doc update) and Z4 (as EXP-018) on Oct 12, and Z5 (slice docs) on Nov 8. Details in TASKS.md.

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

Final-week tasks Y5 (eval report on the 600 val points) and E6a (latency table) were done by Ismail F. on Nov 8 ([`scripts/eval_600.py`](../scripts/eval_600.py) → [`docs/eval-600.md`](eval-600.md), [`docs/latency.md`](latency.md)); both score with `eval.py`'s functions.

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
| n-gram speculative decoding test (rejected) | [EXP-021](../experiments/exp021-ngram-spec/README.md) | done |
| Anxiety-bias correction from label logprobs (rejected), FP8 + bias (NO-GO) | [EXP-022](../experiments/exp022-emotion-bias/README.md), [EXP-023](../experiments/exp023-fp8-bias/README.md) | done |
| CPU profile classifier + blend → **submission #3** (FP8 + blend, Oct 16) | [EXP-024](../experiments/exp024-cpu-classifier/README.md), [EXP-025](../experiments/exp025-profile-blend/README.md) | done |
| Paper ideas checked offline (nothing adopted) | [EXP-026](../experiments/exp026-papers-check/README.md) | done |
| #4 candidate: stop after `emotion_label` + exact-match profile thresholds | EXP-027 | built, check run pending |
| Depth read-out probes (Attention Residuals idea) | [EXP-029](../experiments/exp029-depth-readout/README.md) | built + tested, GPU run pending |
| Final-week docs: eval report on 600 points, latency table, data slices, how-to, training-config labels | [`docs/eval-600.md`](eval-600.md), [`docs/latency.md`](latency.md), [`docs/data-format.md`](data-format.md), [`docs/how-to.md`](how-to.md) | done (Nov 8: Y5, E6, Z5, M2) |

---

## Ismail M.: prompts and training support

| Delivered | Where | Status |
|---|---|---|
| Base-model check: thinking mode on 8B | EXP-006 | done (invalid run → thinking stays off) |
| Four prompt variants; the profile-rule prompt is the origin of V1.8's "leave [] if insufficient information" rules | EXP-007, EXP-008, EXP-009, EXP-014 (V1.6) | done |
| Contribution to the T1 `build_sft.py` rebuild | `train/build_sft.py` | done (T1, with Ismail F.) |

The second training run (P2) he owned was dropped on Oct 11 after the error review. The final-week check of the training write-up (M2) was done by Ismail F. on Nov 8: the three files in `train/configs/` are the Oct 5 planning drafts and are now labelled as such; the configs that ran are in `experiments/exp019-qlora-full/configs/`.

---

## Elly: infrastructure support (Oct 8 – Nov 5)

Elly joined on Oct 8 and left the team on Nov 5 (captain's decision). No code or document in the repo is his. The tasks assigned to him and what happened to each:

| Task | Outcome |
|---|---|
| E1 2-GPU DDP launch | done by Ismail F. (pilot rerun, Oct 8) |
| E2 card-hour ledger + shutdown checks | done by Ismail F. (LOG.md *Compute*) |
| E3 private images | submission image saved and checked by Ismail F. in T9; training image dropped |
| E4 latency levers | done by Ismail F. as EXP-020 (FP8) and EXP-021 (n-gram, rejected) |
| E5 offline BERTScore weights | dropped (every comparison used BLEU / ROUGE-L) |
| T9 submission check, O1b organiser questions | done by Ismail F. |
| E6 latency table + how-to (Nov 5) | done by Ismail F. on Nov 8 |

---

## Where everything stands (Nov 8)

| Area | State |
|---|---|
| Data | Official train only; val used only for evaluation; fixed slices n50 / n50b / n100 / tune500 / t10 (the 600 points) tracked in git |
| Evaluation | validate + eval, analyze v4.1; offline report on the 600 points (`scripts/eval_600.py`) |
| Model | Qwen3-8B non-thinking + full QLoRA, merged, **served in FP8** |
| Prompt | V1.8, frozen, shared by training and inference |
| Profile | CPU TF-IDF + logistic-regression classifier blended with the model's tags (#3); classifier alone with exact-match thresholds (#4 candidate) |
| Submissions | 3 of 5 used. **#3 (Oct 16) = FP8 + profile blend counts now.** #4 candidate (EXP-027) waits for its check run |
| Compute | 3.93 card-hours read on Oct 15 (see LOG.md *Compute*) |
