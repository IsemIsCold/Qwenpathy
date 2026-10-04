# Tasks

Status: `todo` · `doing` · `review` · `done`. Update your own rows, and edit them in your PR.

## Roles

| Area | Owner | Owns |
|---|---|---|
| Infrastructure | **Elly** | `participant/` (start.sh, run_inference.py), the Docker image, the Migu platform, the vLLM backend, latency |
| Dataset | **Zakaria** | Data conversion, val/train splits, SFT data, data statistics, truncating long conversations |
| Training | **Ismail M.** (lead) | Base model choice, LoRA/QLoRA, AutoDL runs, the weights we ship |
| Evaluation | **Yahya** | `validate.py`, `eval.py`, the reply-quality judge, the score history |
| Classification + captain | **Ismail F.** | `emotion_label` + `user_profile`, the system prompt, merging PRs, **all submissions** |

## Phase 2: Baseline (Oct 1–4)

**Goal:** a first successful submission, plus a way to measure ourselves on val. Measurement tools are done (Oct 4); submission #1 moved to **Oct 7** (waiting on the Migu setup).

| # | Task | Owner | Depends on | Due | Status |
|---|---|---|---|---|---|
| B1 | Run the official baseline locally (Qwen2.5-0.5B, CPU, 5 samples) and write down the exact run command in the README | Elly → done by Ismail F. ([EXP-001](experiments/exp001-baseline-qwen0.5b-cpu/README.md); Qwen2.5-7B checked via API in [EXP-002](experiments/exp002-qwen7b-api/README.md)) | — | Oct 2 | done (Oct 3) |
| B2 | `scripts/convert_val.py`: val set → test-format input + gold labels, with an `--n` flag | Zakaria | — | Oct 2 | done (Oct 4) |
| B3 | `scripts/validate.py`: checks the submission format (4 fields, allowed labels, ids match the input) | Yahya | B2 (format) | Oct 2 | done (Oct 4, 58 tests) |
| B4 | `scripts/eval.py`: emotion accuracy + macro-F1, profile F1, majority-class baselines | Yahya | B2 | Oct 3 | done (Oct 4) |
| B5 | Emotion-first prompt: label before reply, guidance for the top 5 emotions, profile "empty unless evidence", measured on 50 val samples | Ismail F. | B1, B4 | Oct 6 | doing |
| B6 | Data stats report: label distribution, turn/char lengths, empty profiles → `docs/data-stats.md` | Zakaria | — | Oct 3 | done (Oct 4, v3) |
| B7 | Migu: set up the instance, run the official image with Qwen3-14B (non-thinking, vLLM TP=2) on val, measure latency, make the pipeline ready to submit | Elly | B1, B3 | Oct 6 | todo |
| B8 | Choose the base model + LoRA config | Ismail M. | B6 | Oct 4 | base model decided: Qwen3-14B non-thinking ([technical plan](docs/technical-plan.md), [EXP-003](experiments/exp003-qwen3-14b-thinking/README.md)); LoRA config todo |
| B9 | **Submission #1** (Qwen3-14B non-thinking + best B5 prompt) | Ismail F. | B7, B4 | Oct 7 | todo |

## Phase 3: Improve (Oct 5–14)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| I1 | `train/build_sft.py`: train set → chat-format SFT data matching inference exactly, split by conversation | Zakaria (implemented by Ismail F.) | Oct 6 | done (Oct 4) |
| I2 | LoRA smoke test (50 examples), then the full run on AutoDL | Ismail M. | Oct 9 | todo |
| I3 | A dedicated classifier for emotion and profile (e.g. a small Chinese BERT): does it beat the LLM's labels? | Ismail F. | Oct 10 | todo |
| I4 | Per-sample mistakes CSV, then character-level BLEU-4 / ROUGE-L in eval.py (20 official points), then the reply-quality judge | Yahya | Oct 8 | todo |
| I5 | vLLM backend + latency of the first 100 samples; truncation at inference (3,500 tokens, same rule as build_sft.py); patch `run_inference.py` to `sorted(EMOTIONS)` | Elly | Oct 10 | todo |
| I6 | Submissions #2–#3 with the best fine-tuned model + classifier | Ismail F. | Oct 14 | todo |
| I7 | Full EXP-003 on Migu: Qwen3-14B thinking vs non-thinking on 50 val points (≈0.3 card-hours) | Elly + Ismail M. | Oct 6 | todo |
| I8 | Ask the organisers again: Qwen3 allowed? profile metric? memory_refs at test time? | Ismail F. | Oct 5 | todo |

## Phase 4: Finalise (Oct 15–20)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| F1 | Pick the final config from the val scores | All | Oct 16 | todo |
| F2 | Final submission(s) | Ismail F. | Oct 18 | todo |
| F3 | Buffer. Fix failed runs only, no new features | Elly + Ismail F. | Oct 20 | todo |
