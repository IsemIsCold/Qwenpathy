# Tasks

Status: `todo` · `doing` · `review` · `done` · `dropped`. Update your own rows, and edit them in your PR.
**Last full update: Oct 5 (evening).** Deadline **Oct 20, 23:59**. Code freeze **Oct 18**.

## Where we are (Oct 5)

- **Submission #1 (V0) is in** (Oct 4). Official score pending.
- **Prompt phase is finished.** 16 experiments (EXP-005 to EXP-016): no prompt beats V0 on emotion beyond noise, but **V1.8** wins clearly on profile, confirmed on an untouched val slice (n50b). V1.8 is frozen and is now `participant/run_inference.py` on `main` (V0 = commit `e5209d2`).
- **Next phase: V2 = fine-tuning** (QLoRA on Qwen3-8B). Emotion (25 pts) can only improve through training now.
- **Submission #2 waits for the first training results.** Rule: a fine-tuned model is submitted only if it beats V1.8 on **both** n50 and n50b. Fallback: if nothing beats V1.8 by **Oct 16**, submit V1.8.
- Remember: the **last** successful submission counts. Never submit something worse after something better.

### Baseline to beat (V1.8 = V1.7 prompt, HF API, greedy)

| | n50 | n50b |
|---|---|---|
| Emotion accuracy | 0.46 | 0.62 |
| Profile micro-F1 / exact | 0.373 / 0.320 | 0.300 / 0.247 |
| ROUGE-L | 0.178 | 0.200 |

These are API numbers. The fine-tuned model runs on vLLM, so **T4 below re-measures V1.8 on the same vLLM setup** before any comparison.

## Roles

| Area | Owner | Owns |
|---|---|---|
| Captain, prompt, Migu, submissions | **Ismail F.** | `participant/`, the Migu image, all submissions, merging PRs. Took over infrastructure from Elly (unavailable). |
| Training | **Ismail M.** (lead) | SFT data build, LLaMA-Factory configs, training runs on Migu, merged weights |
| Dataset | **Zakaria** | val/train splits, data stats, truncation of long conversations |
| Evaluation | **Yahya** | `validate.py`, `eval.py`, `analyze.py`, latency measurement |
| Infrastructure | ~~Elly~~ | unavailable; tasks moved to Ismail F. / Yahya |

## Phase 3a: V2 — fine-tuning on Migu (Oct 6–11)

**No AutoDL: everything (training + evals + submissions) runs on the Migu instance (2× RTX 4090).** The 28.67 card-hours left are the hard budget. Log the estimate in LOG.md **before** every GPU job and the actual after.

| Job | Card-hours (est.) |
|---|---|
| T4 V1.8 baseline on vLLM (n50 + n50b) | ~0.3 |
| T5 pilot QLoRA (2k × 1 epoch) | ~1 |
| T6 pilot eval | ~0.3 |
| T7 full QLoRA (18.6k × 1 epoch) | ~5–6 |
| T8 checkpoint evals | ~0.6 |
| T9 + submissions | ~1.5 |
| P3 AWQ test | ~1 |
| **Planned total** | **~10–11** (P2 second run: +~6 if needed) |
| **Reserve, never touch** | **≥ 5** |

Rules: installs, downloads, file copying and data prep with 无卡开机 (*wú kǎ kāijī*, no GPU). Turn the GPU instance off the moment a job ends. Train in a **separate conda env** (`train_env`) so the submission env (`conda_env`) and the saved submission image stay untouched; clone the instance or save a separate private image for training.

| # | Task | Owner | Depends on | Due | Status |
|---|---|---|---|---|---|
| T1 | **Rebuild SFT data with the frozen prompt.** `build_sft.py` must `import` SYSTEM_PROMPT + the history formatter from `participant/run_inference.py` (= V1.8 since Oct 5; no copy-paste), set `memory_refs: []` in every target, keep the field order the parser expects, and exclude all val conversations. Output: LLaMA-Factory sharegpt + `dataset_info.json`, template `qwen3_nothink`. Report: #examples, token-length stats (p50/p95/max), how many are truncated. | Ismail M. (review: Ismail F.) | — | Oct 6 | todo |
| T2 | **Migu training setup** (无卡开机, no GPU): new conda env `train_env` with LLaMA-Factory + bitsandbytes; reuse the Qwen3-8B weights already on the instance; copy SFT data + val n50/n50b. Check pip/ModelScope access works from the instance. Save it as a private image (`npu-ic-train`). | Ismail M. (+ Ismail F. for Migu access) | — | Oct 6 | todo |
| T3 | **vLLM eval harness on Migu:** one script that serves a model (base or merged) with vLLM in `conda_env` and runs `participant/run_inference.py` (V1.8) on n50 and n50b, then `eval.py` + `analyze.py`. Same config as the submission (non-thinking, greedy). Basically `start.sh` + eval, so it also proves the submission path. | Ismail F. | T2 | Oct 7 | todo |
| T4 | **V1.8 baseline on vLLM** (base Qwen3-8B, no training) on n50 + n50b with T3. These numbers replace the API numbers as the bar to beat. | Ismail F. | T3 | Oct 7 | todo |
| T5 | **Pilot QLoRA:** 2,000 training examples × 1 epoch on Migu (≈40–60 min on one 4090, or ~half with both GPUs via DDP; ~1 card-hour either way). LoRA r=16, α=32, lr 1e-4, cutoff 4096, loss on the assistant turn only. | Ismail M. | T1, T2 | Oct 7 | todo |
| T6 | **Pilot gate** (run T3 on the pilot adapter, merged): 100% parseable JSON · no emotion collapse (no label > 60% of predictions) · emotion accuracy clearly above T4 on n50 **and** n50b · reply length near gold (median ≈ 100 chars). Fail → fix data/config before any full run. | Ismail F. + Yahya | T4, T5 | Oct 8 | todo |
| T7 | **Full QLoRA run:** full train set × 1 epoch on Migu, both GPUs (DDP): ≈1.5–2.5 h wall, ≈5–6 card-hours; exact estimate from T1's token count and the pilot's speed, written in LOG.md first. Save checkpoints every ~25%. | Ismail M. | T6 pass | Oct 9 | todo |
| T8 | **Merge** the LoRA into FP16 safetensors (no peft at inference), load it in vLLM on Migu, run T3. Pick the best checkpoint on n50 + n50b. | Ismail M. + Ismail F. | T7 | Oct 10 | todo |
| T9 | **Submission check** of the merged model: copy weights into `participant/model` in `conda_env`, run `start.sh` end to end, measure latency (mean of first 100), save the submission image. Budget ≈ 0.5 card-hours. | Ismail F. | T8 | Oct 11 | todo |
| T10 | **Submission #2** = fine-tuned model, only if it beats V1.8 on n50 and n50b (T4 numbers). | Ismail F. | T9 | Oct 11 | todo |

## Phase 3b: polish (Oct 12–16)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| P1 | Error analysis of the fine-tuned model with `analyze.py --compare` against V1.8: which emotions / profile tags got worse, reply length vs latency | Yahya | Oct 12 | todo |
| P2 | Second training run only if P1 shows a clear fix (e.g. class balance, longer/shorter targets, 2 epochs) **and** ≥ 11 card-hours remain (6 for the run + 5 reserve). One run max. | Ismail M. | Oct 14 | todo |
| P3 | **AWQ test** (only if the organisers allow quantised models): quantise the merged model (AutoAWQ / llm-compressor), compare FP16 vs AWQ on Migu: latency + accuracy on val50. Switch only if latency drops clearly and accuracy stays within ~2 points. | Ismail M. + Ismail F. | Oct 15 | todo |
| P4 | Truncation at inference for very long histories (same rule as `build_sft.py`), so no request can exceed the context and kill the run | Zakaria | Oct 13 | todo |
| P5 | **Submission #3** with the best of P2/P3 | Ismail F. | Oct 16 | todo |
| P6 | **Fallback:** if no fine-tuned model beats V1.8 by Oct 16, submit V1.8: `participant/` on `main` is already V1.8 (Oct 5), so rebuild the Migu image from `main` with the base Qwen3-8B weights | Ismail F. | Oct 16 | todo |

## Phase 4: finalise (Oct 17–20)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| F1 | Pick the final config from n50 + n50b + Migu val50 | All | Oct 17 | todo |
| F2 | Final submission (#4, keep #5 spare) | Ismail F. | Oct 18 | todo |
| F3 | Expert-review evidence (30 pts): README, experiment logs, LOG.md, clean code, docs/how-to.md up to date | All | Oct 19 | todo |
| F4 | Buffer: fix failed runs only, no new features | Ismail F. | Oct 20 | todo |

## Ongoing

| # | Task | Owner | Status |
|---|---|---|---|
| O1 | Ask the organisers (WeChat group): profile metric (F1 vs exact match), latency formula, are quantised models allowed, memory bank at test time? | Ismail F. | doing |
| O2 | `analyze.py` follow-ups: `--force` passthrough for submissions with blank labels; drop "increase temperature" advice (line 402) | Yahya | todo |
| O3 | EXP-006/007/008/009: push the full prompt text + run folders into `experiments/` | Ismail M. | todo |
| O4 | Commit everything from Oct 5: convert_val (`--exclude/--sample`), exp3_compare (retries), analyze.py + tests, participant V1.7/V1.8 into git history, READMEs, LOG | Ismail F. | todo |

## Done (archive)

| # | Task | Owner | Done |
|---|---|---|---|
| B1 | Official baseline run locally (EXP-001), Qwen2.5-7B via API (EXP-002) | Ismail F. | Oct 3 |
| B2 | `convert_val.py` (+ `--exclude` / `--sample` for n50b, Oct 5) | Zakaria / Ismail F. | Oct 4 |
| B3 | `validate.py` (58 tests) | Yahya | Oct 4 |
| B4 | `eval.py`: emotion, profile, BLEU-4 / ROUGE-L / BERTScore | Yahya | Oct 4 |
| B5 | Prompt work: EXP-005 to EXP-016 → **V1.8** frozen | Ismail F. + Ismail M. | Oct 5 |
| B6 | Data stats → `docs/data-stats.md` | Zakaria | Oct 4 |
| B7 | Migu setup, val50 run, private image (done by Ismail F. for Elly) | Ismail F. | Oct 4 |
| B8 | Base model: **Qwen3-8B non-thinking** (EXP-003, EXP-004, EXP-006) | Ismail M. / Ismail F. | Oct 4 |
| B9 | **Submission #1** (V0) | Ismail F. | Oct 4 |
| I1 | `train/build_sft.py` first version (rebuilt in T1) | Ismail F. | Oct 4 |
| I4 | BLEU/ROUGE-L/BERTScore in eval.py; per-sample CSV → `analyze.py` v2.2 merged | Yahya | Oct 5 |
| I7 | Thinking vs model size via API instead of Migu (0 card-hours) | Ismail F. | Oct 4 |
| — | n50b independent eval slice | Ismail F. | Oct 5 |
| I3 | Dedicated emotion/profile classifier (small Chinese BERT) | Ismail F. | **dropped**: no time next to fine-tuning; SFT learns the labels directly |
| I5 | vLLM backend + `sorted(EMOTIONS)` patch (done in submission #1); latency script → Yahya; truncation → P4 | — | split |
