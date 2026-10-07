# Tasks

Status: `todo` · `doing` · `review` · `done` · `dropped`. Update your own rows, and edit them in your PR.
**Last full update: Oct 7 (night).** Deadline **Oct 20, 23:59**. Code freeze **Oct 18**.

## Where we are (Oct 7)

- **Submission #1 (V0) is in** (Oct 4). Official score pending.
- **Prompt phase is finished.** 16 experiments (EXP-005 to EXP-016): no prompt beats V0 on emotion beyond noise, but **V1.8** wins clearly on profile, confirmed on an untouched val slice (n50b). V1.8 is frozen and is now `participant/run_inference.py` on `main` (V0 = commit `e5209d2`).
- **Next phase: V2 = fine-tuning** (QLoRA on Qwen3-8B). Emotion (25 pts) can only improve through training now.
- **Submission #2 waits for the first training results.** Rule: a fine-tuned model is submitted only if it beats V1.8 on **both** n50 and n50b. Fallback: if nothing beats V1.8 by **Oct 16**, submit V1.8.
- **Oct 6: T1 done** — SFT data verified (`check_sft.py` PASS). Training uses `cutoff_len 6144`. Note for eval: val gold `val_conv_000331` contains leaked `<think>` reasoning (not in n50/n50b).
- **Oct 7: EXP-017 PASS** — DeepSeek V4.1-Flash (HF router `:novita`, thinking off) beats V1.8 on n50 and n50b (+~0.05 ROUGE-L, +10–16 acc). Not submittable (offline); it becomes the P2 data tool (rewrite long replies, rare-emotion augmentation).
- **Oct 7: training data filtered** to gold replies ≤ 350 chars (`build_sft.py --max-reply-chars 350`, now default): 14,942 train / 768 dev, 18.7M train tokens.
- **Oct 7: T2, T3, T4 done; T5 pilot running.** Training env = venv `/opt/train_venv` on top of `conda_env` (conda_env untouched). Template = `qwen3` + `enable_thinking: false` (matches vLLM exactly; `qwen3_nothink` did NOT add the empty think block). **New bar to beat = T4 (V1.8 on Migu vLLM): n50 acc 0.54 / ROUGE-L 0.174, n50b acc 0.62 / ROUGE-L 0.197.**
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
| T5 pilot QLoRA (2k × 1 epoch) | ~1 (actual ≈0.8: ran on 1 GPU while 2 were booked) |
| T6 pilot eval | ~0.3 |
| T7 full QLoRA (14.9k × 1 epoch, 18.7M tokens) | **~3–3.5** (pilot speed ≈3.1 s/step at 4 samples/step on 1 GPU; use both GPUs) |
| T8 checkpoint evals | ~0.6 |
| T9 + submissions | ~1.5 |
| P3 AWQ test | ~1 |
| **Planned total** | **~10–11** (P2 second run: +~6 if needed) |
| **Reserve, never touch** | **≥ 5** |

Rules: installs, downloads, file copying and data prep with 无卡开机 (*wú kǎ kāijī*, no GPU). Turn the GPU instance off the moment a job ends. Train in the **separate venv `/opt/train_venv`** (built on top of `conda_env` with `--system-site-packages`; launch with `/opt/train_venv/bin/llamafactory-cli`) so the submission env (`conda_env`) and the saved submission image stay untouched; clone the instance or save a separate private image for training.

| # | Task | Owner | Depends on | Due | Status |
|---|---|---|---|---|---|
| T1 | **Rebuild SFT data with the frozen prompt.** `train/build_sft.py` imports SYSTEM_PROMPT + `conversation_text()` from `participant/run_inference.py` (V1.8), `memory_refs: []`, prompt field order, val excluded; skips 7 gold replies with leaked `<think>` reasoning. `train/check_sft.py` = sign-off (val leakage, targets, prompt identity, exact token lengths, pilot subset). **Result (Oct 6, PASS):** 17,745 train + 905 dev (5% of train conversations); 660 histories truncated (`--max-user-tokens 3500`); total tokens p50 1,110 / p95 3,597 / max 5,009; train sum 25.2M; pilot = `migu_pilot2k` (2,000, seed 42). Output: `data/sft/` sharegpt + `dataset_info.json`. | Ismail M. (review: Ismail F.) | — | Oct 6 | done |
| T2 | **Migu training setup** — venv `/opt/train_venv` on `conda_env` (torch 2.10.0+cu128 kept via constraint; LLaMA-Factory 0.9.5, bitsandbytes 0.50.2); model at `/root/participant/model`; data in `/mnt/storage/sft` + `/mnt/storage/val`; configs in `/mnt/storage/configs`. Smoke test (10 steps, 1 GPU) PASS: 4-bit load, loss 2.04→1.30, 1.36 samples/s. Template fixed to `qwen3` + `enable_thinking: false` (empty think block in prompt, masked). `torchrun` shim added to the venv for 2-GPU DDP. Lessons: conda clone broke torch (pip pulled cu130) → venv instead; `FORCE_TORCHRUN` used conda_env's torchrun → shim. | Ismail F. (+ Ismail M.) | — | Oct 7 | done |
| T3 | **vLLM eval harness on Migu** = `bash /root/participant/start.sh <input> <outdir>` (V1.8 `run_inference.py` installed, V0 kept as `run_inference_v0.py`) + `scripts/validate.py` + `scripts/eval.py` in `/mnt/storage/scripts` (run with `/opt/train_venv/bin/python`). BERTScore skipped (no `bert-base-chinese` on the instance yet). | Ismail F. | T2 | Oct 7 | done |
| T4 | **V1.8 baseline on vLLM** (base Qwen3-8B, no training). n50: acc **0.54**, macro-F1 0.238, profile micro-F1 0.352, exact 0.287, ROUGE-L **0.174**, BLEU-4 0.021, latency mean 1.32 s. n50b: acc **0.62**, macro-F1 0.336, profile 0.323, exact 0.287, ROUGE-L **0.197**, BLEU-4 0.068, latency 1.52 s. 0 format errors. **This is the bar for T6/T10.** | Ismail F. | T3 | Oct 7 | done |
| T5 | **Pilot QLoRA:** `configs/pilot.yaml` = 2,000 ex (`migu_pilot2k`) × 1 epoch, LoRA r=16 α=32 all-linear, lr 1e-4 cosine, batch 1 × accum 4, `cutoff_len 6144`, template `qwen3` + `enable_thinking: false`, output `/mnt/storage/runs/pilot`. Running on 1 GPU (500 steps, ~3.1 s/step, ~26 min); loss 2.54 → ~1.25 by step 100. Next: read `train_runtime` and loss curve. | Ismail F. | T1, T2 | Oct 7 | doing |
| T6 | **Pilot gate:** merge on CPU (无卡开机) with `llamafactory-cli export configs/merge_pilot.yaml` → `/mnt/storage/merged/pilot`, copy the base model's tokenizer files over it (same chat template as V1.8), run T3 on n50 + n50b. Pass: 100% parseable JSON · no label > 60% of predictions · emotion acc clearly above T4 (0.54 / 0.62) on **both** slices · reply median ≈ 100 chars · latency not worse than ~1.5 s. Fail → fix data/config before any full run. | Ismail F. + Yahya | T4, T5 | Oct 8 | todo |
| T7 | **Full QLoRA run:** 14,942 ex × 1 epoch (`migu_train`), **both GPUs**: `export PATH=/opt/train_venv/bin:$PATH; FORCE_TORCHRUN=1 nohup llamafactory-cli train configs/full.yaml &`. Estimate ≈1.6 h wall, **≈3–3.5 card-hours** (log before starting). Save checkpoints every ~25%. Never leave the instance on after the run. | Ismail F. (+ Ismail M.) | T6 pass | Oct 9 | todo |
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
