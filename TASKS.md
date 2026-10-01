# Tasks

Status: `todo` · `doing` · `review` · `done`. Update your own rows, and edit them in your PR.

## Phase 2: Baseline (Oct 1–4)

**Goal:** a first successful submission by **Oct 4**, plus a way to measure ourselves on val.

| # | Task | Owner | Depends on | Due | Status |
|---|---|---|---|---|---|
| B1 | Run the official baseline locally (Qwen2.5-0.5B, CPU, 5 samples) and write down the exact `run_inference.py` command | Ismail F. | — | Oct 2 | doing |
| B2 | `convert_val.py`: val set → test-format input file + gold-label file | Yahya | — | Oct 2 | todo |
| B3 | `validate.py`: checks a `submission.jsonl` (all 4 fields, allowed labels, one line per input id, valid JSON, UTF-8) | Yahya | — | Oct 2 | todo |
| B4 | `eval.py`: emotion accuracy + macro-F1, profile F1 per sub-field, and a per-emotion confusion table | Zakaria | B2 (format) | Oct 3 | todo |
| B5 | Emotion-first prompt: `emotion_label` comes before `response_text`, with guidance for the top 5 emotions | Elly | B1 | Oct 3 | todo |
| B6 | `build_sft.py`: train set → chat-format SFT JSONL | Ismail M. | — | Oct 4 | todo |
| B7 | Build the image on Migu with a bigger Qwen, run it on full val, and **submit #1** | Ismail F. | B1, B3, B4 | Oct 4 | todo |

## Phase 3: Improve (Oct 5–14)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| I1 | LoRA fine-tune on the train set (AutoDL), starting with a 50-sample smoke test | Ismail M. | Oct 9 | todo |
| I2 | Prompt iterations, each scored with `eval.py` | Elly | ongoing | todo |
| I3 | Profile prediction: does a separate pass or a classifier beat the LLM? | Zakaria | Oct 10 | todo |
| I4 | Long-conversation handling: truncation/summary for 40k-char histories | Ismail F. | Oct 10 | todo |
| I5 | Latency: switch to the vLLM backend and measure the first 100 samples | Yahya | Oct 12 | todo |
| I6 | Submissions #2–#3 with the best fine-tuned model | Ismail F. | Oct 14 | todo |

## Phase 4: Finalise (Oct 15–20)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| F1 | Pick the final config based on val scores | All | Oct 16 | todo |
| F2 | Final submission(s) | Ismail F. | Oct 18 | todo |
| F3 | Buffer. Fix failed runs only, no new features | Ismail F. | Oct 20 | todo |
