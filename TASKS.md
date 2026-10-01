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

**Goal:** a first successful submission by **Oct 4**, plus a way to measure ourselves on val.

| # | Task | Owner | Depends on | Due | Status |
|---|---|---|---|---|---|
| B1 | Run the official baseline locally (Qwen2.5-0.5B, CPU, 5 samples) and write down the exact run command in the README | Elly | — | Oct 2 | todo |
| B2 | `scripts/convert_val.py`: val set → test-format input + gold labels, with an `--n` flag | Zakaria | — | Oct 2 | todo |
| B3 | `scripts/validate.py`: checks the submission format (4 fields, allowed labels, ids match the input) | Yahya | B2 (format) | Oct 2 | todo |
| B4 | `scripts/eval.py`: emotion accuracy + macro-F1, profile F1, majority-class baselines | Yahya | B2 | Oct 3 | todo |
| B5 | Emotion-first prompt: label before reply, guidance for the top 5 emotions, measured on 50 val samples | Ismail F. | B1, B4 | Oct 3 | todo |
| B6 | Data stats report: label distribution, turn/char lengths, empty profiles → `docs/data-stats.md` | Zakaria | — | Oct 3 | todo |
| B7 | Migu: set up the instance, run the official image with a bigger Qwen on full val, and make the pipeline ready to submit | Elly | B1, B3 | Oct 4 | todo |
| B8 | Choose the base model + LoRA config, and write the plan in `docs/training-plan.md` | Ismail M. | B6 | Oct 4 | todo |
| B9 | **Submission #1** (baseline) | Ismail F. | B7, B4 | Oct 4 | todo |

## Phase 3: Improve (Oct 5–14)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| I1 | `train/build_sft.py`: train set → chat-format SFT data, with a split by conversation | Zakaria | Oct 6 | todo |
| I2 | LoRA smoke test (50 examples), then the full run on AutoDL | Ismail M. | Oct 9 | todo |
| I3 | A dedicated classifier for emotion and profile (e.g. a small Chinese BERT): does it beat the LLM's labels? | Ismail F. | Oct 10 | todo |
| I4 | Reply-quality judge: score `response_text` on val (empathy, coherence, use of memory) | Yahya | Oct 8 | todo |
| I5 | vLLM backend + latency of the first 100 samples; long-history truncation at inference time | Elly | Oct 10 | todo |
| I6 | Submissions #2–#3 with the best fine-tuned model + classifier | Ismail F. | Oct 14 | todo |

## Phase 4: Finalise (Oct 15–20)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| F1 | Pick the final config from the val scores | All | Oct 16 | todo |
| F2 | Final submission(s) | Ismail F. | Oct 18 | todo |
| F3 | Buffer. Fix failed runs only, no new features | Elly + Ismail F. | Oct 20 | todo |
