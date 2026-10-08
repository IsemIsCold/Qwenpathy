# Who built what, and how it helps you

> Last updated: **2026-10-08**. Each section: what the person delivered, where it lives, and which teammates depend on it. Commands for every tool are in [how-to.md](how-to.md).

## The dependency map

```
Zakaria ──val files──► Yahya (scores them)      ──scores──► everyone's decisions
   │                   Ismail F. (API runs)
   │                   Elly (Migu runs)
   │
   └──SFT data spec──► build_sft.py ──training data──► Ismail M. (QLoRA) ──weights──► Elly (image)
                            ▲
Ismail F. ──prompt (B5)─────┘ (training and inference share one prompt)
Elly ──image + latency──► Ismail F. (submissions)
```

---

## Zakaria: dataset

| Delivered | File | Status |
|---|---|---|
| Val → test-format input + gold labels, with `--n` | [`scripts/convert_val.py`](../scripts/convert_val.py) | done (B2) |
| Every JSONL format we use, documented | [`docs/data-format.md`](data-format.md) | done |
| Dataset statistics (labels, profiles, lengths, tokens) | [`scripts/compute_stats.py`](../scripts/compute_stats.py) → [`docs/data-stats.md`](data-stats.md) | done (B6, v3) |
| Training-data rules (split by conversation, truncation) | spec for [`train/build_sft.py`](../train/build_sft.py) | done (I1, coded by Ismail F.) |

**How it helps you**
- **Everyone:** `val_input_n50.jsonl` / `val_gold_n50.jsonl` are the shared test set. Every experiment uses the same 50 points, so results are comparable.
- **Ismail F. (prompt):** [data-stats.md](data-stats.md) gives the base rates to put in the prompt (anxiety 32% and relaxed 17% of val points; 37% of profiles completely empty) and the gold reply length to target.
- **Ismail M. (training):** the token-length table decides the max sequence length (we use 3,500 tokens of history).
- **Elly (inference):** the same table says how many test histories will need truncating at inference.

---

## Yahya: evaluation

| Delivered | File | Status |
|---|---|---|
| Strict format checker (fails closed) | [`scripts/validate.py`](../scripts/validate.py) | done (B3) |
| Scorer: emotion accuracy + macro-F1 + per-class table + confusions, profile micro/macro-F1 + exact match, baselines, reply length | [`scripts/eval.py`](../scripts/eval.py) | done (B4) |
| Test suite (attacks the validator, checks the scorer's maths) | [`tests/run_tests.py`](../tests/run_tests.py) | done: 64/64 |
| Design + how to read the output | [`docs/eval-design.md`](eval-design.md), [`docs/interpreting-eval-output.md`](interpreting-eval-output.md) | done |
| Character-level BLEU-4, ROUGE-L, BERTScore in `eval.py` (the 20 reply-quality points) | next version of `eval.py` | in review: BLEU/ROUGE verified against sacrebleu; 2 BERTScore fixes pending (layer 8, exclude [CLS]/[SEP]) |

**How it helps you**
- **Everyone:** run `validate.py` before believing any result, and `eval.py` to compare anything. No "it looks better" decisions.
- **Ismail F.:** the confusion pairs in `eval.py`'s output (fear→anxiety, relaxed↔gratitude, helplessness→sadness) are exactly what B5 targets.
- **Ismail M.:** a fine-tuned model only "wins" when `eval.py` on val beats the untuned one.
- **Elly:** run `validate.py` on the Migu output before any submission. It catches the format errors that would score 0.

---

## Ismail F.: captain, training lead (QLoRA), Migu & infrastructure, classification & prompt

| Delivered | File | Status |
|---|---|---|
| Official baseline run, exact commands (did Elly's B1) | [`experiments/exp001-*`](../experiments/exp001-baseline-qwen0.5b-cpu/README.md) | done |
| Qwen2.5-7B via API | [`experiments/exp002-*`](../experiments/exp002-qwen7b-api/README.md) | done |
| API experiment runner: official prompt/parser + validate + eval in one command | [`scripts/exp3_compare.py`](../scripts/exp3_compare.py) | done |
| Windows setup for API runs | [`scripts/setup-api-env.ps1`](../scripts/setup-api-env.ps1) | done |
| Qwen3-14B thinking vs non-thinking, 50 pts | [`experiments/exp003-*`](../experiments/exp003-qwen3-14b-thinking/README.md) | done |
| Qwen3-14B vs 8B, 50 pts → **decision: 8B non-thinking** | [`experiments/exp004-*`](../experiments/exp004-model-size/README.md) | done |
| Training-data builder (imports the inference prompt) | [`train/build_sft.py`](../train/build_sft.py) | done (I1) |
| Technical plan, organiser questions | [`docs/technical-plan.md`](technical-plan.md), [`docs/organiser-answers.md`](organiser-answers.md) | maintained |
| Better prompt (emotion first, label definitions, strict profile, reply length) | EXP-005 to EXP-016 → V1.8 | done (B5) |
| Migu setup, val50 run, private image, **submission #1** (did Elly's B7 while she was out) | `participant/`, LOG.md | done (B7, B9) |
| Teacher test: DeepSeek V4.1-Flash as P2 data tool | `experiments/exp017-teacher/` | done (EXP-017) |
| **SFT data rebuild** with the frozen V1.8 prompt + `check_sft.py` sign-off (17,745 → 14,942 train after the 350-char filter) | [`train/build_sft.py`](../train/build_sft.py), [`train/check_sft.py`](../train/check_sft.py) | done (T1, Oct 6–7) |
| **Migu training environment**: venv `/opt/train_venv` on `conda_env`, smoke test, `qwen3` + `enable_thinking: false` template fix, `torchrun` shim | `/mnt/storage/configs`, TASKS T2 | done (T2, Oct 7) |
| **vLLM eval harness on Migu** (start.sh + validate + eval, model override by env) | TASKS T3 | done (T3, Oct 7) |
| **V1.8 baseline on Migu vLLM** = the bar for fine-tuning (acc 0.54 / 0.62) | TASKS T4, LOG.md | done (T4, Oct 7) |
| **Pilot QLoRA** (2k ex) + **first 2-GPU DDP run** | `configs/pilot_ddp.yaml`, TASKS T5 / E1 | done (T5, E1, Oct 8) |
| **Pilot gate**: acc 0.68 / 0.78, ROUGE-L 0.237 / 0.259 | TASKS T6, LOG.md | done (T6, Oct 8) |
| **Full QLoRA run + merge** (14,942 ex, 1 h 48 min): **acc 0.80 / 0.86, ROUGE-L 0.269 / 0.293** = submission #2 candidate | `configs/full.yaml`, `/root/merged/full`, TASKS T7, LOG.md | done (T7, Oct 8) |

**How it helps you**
- **Ismail M.:** the base model is decided (Qwen3-8B, non-thinking), and `build_sft.py` gives you training data in LLaMA-Factory format that matches inference byte for byte.
- **Elly:** `exp3_compare.py` results are the target. Your Migu run of the same 50 points should give similar scores. If not, something differs between API and vLLM.
- **Everyone:** `exp3_compare.py` lets anyone test an idea for cents without touching the GPU budget.

---

## Ismail M.: training support (P2)

| Delivered | File | Status |
|---|---|---|
| Base model choice | [technical-plan.md §2](technical-plan.md) | done: Qwen3-8B non-thinking (B8) |
| API setup on his laptop | `scripts/setup-api-env.ps1` | done |
| LoRA config, SFT data rebuild, smoke test, training runs | Migu | done by Ismail F. (T1–T7) |

**What he needs from others:** the SFT data from `build_sft.py` (ready), the final prompt from B5 (rebuild the data when it changes), and `eval.py` to judge each checkpoint.
**What others need from him:** merged bf16 weights that load offline in `participant/model`. Agree the format with Elly.

---

## Elly: infrastructure support (joins Oct 8; the Migu setup and all training runs were built by Ismail F.)

| Delivered | File | Status |
|---|---|---|
| Migu instance + official image running Qwen3-8B on vLLM, latency measured | `docs/submission-howto.md` (to write) | todo (B7) |
| vLLM backend, truncation at inference (same rule as `build_sft.py`), `sorted(EMOTIONS)` patch | `participant/` | todo (I5) |

**What she needs from others:** the patched `participant/run_inference.py` (sorted emotions, now in git), `validate.py`/`eval.py` to check the Migu output, the 50 val points, and later Ismail M.'s weights.
**What others need from her:** proof the pipeline runs in the official image, the real latency on 2×4090, and the step-by-step submission process. **Nobody can submit until B7 is done.**

---

## Where everything stands

| Area | Done | Next |
|---|---|---|
| Data | val files, stats, formats, SFT data | 200-point val file for confirming B5 |
| Evaluation | validate, eval, 64 tests | BLEU/ROUGE/BERTScore merged (I4) |
| Model choice | Qwen3-8B non-thinking | — |
| Prompt | baseline measured | B5: V0–V3 on 50 pts, winner on 200 |
| Training | T1–T7 by Ismail F. (data rebuild, env, pilot, full run, merges): full model acc 0.80 / 0.86 | T9 submission check, P2 only if P1 finds a clear fix |
| Infrastructure | B7 done by Ismail F.; 2-GPU DDP works (Oct 8) | Elly back Oct 8: card-hour ledger, images, latency, T9 |
| Submissions | 1 of 5 used (V0, Oct 4) | #2 = T7 full model after T9 |
