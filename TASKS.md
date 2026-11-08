# Tasks

Status: `todo` · `doing` · `review` · `done` · `dropped`. Update your own rows, and edit them in your PR.
**Last full update: Nov 8.** Deadline **Nov 10, 23:59** (moved from Oct 20 by the organisers on Oct 17). Code freeze **Nov 8, 23:59**: after that, docs only.

## Where we are (Nov 5)

- **Submissions: 3 of 5 used. #3 (FP8 + CPU profile classifier, EXP-025) is the one that counts now. 2 left.** The last successful submission counts, so we never submit something worse after something better.
- **#4 candidate = EXP-027** (stop generating after `emotion_label`, profile from the CPU classifier with exact-match thresholds, lenient JSON parser): ≈ +1.3 contest points over #3 offline. It still needs its Migu check run before it can be submitted. The organisers confirmed the profile score is **exact match**, so the `exact` variant is the one we use.
- **EXP-029 (new, offline study):** can a read-out over all 36 layers of our model (the Attention Residuals idea) predict emotion and profile better than the label the model writes? Built and tested (`tests/test_exp029.py` 38/38); it needs one Migu boot. Pass rules are fixed in `experiments/exp029-depth-readout/README.md`. It ships nothing by itself; a PASS could only become #5.
- **GPU rule unchanged: nobody boots Migu without asking Ismail F.** Every task below for the team is CPU-only.
- **Nov 5: Elly is no longer on the team** (captain's decision). His E6 is split between Yahya (latency table) and Zakaria (how-to.md); Migu and infrastructure stay with Ismail F., who already did E1–E5.
- The expert review is worth **30 points**: from now on, the most useful work for everyone is making the repo, the docs and the experiment logs correct and easy to check.

## Phase 5: final week (Nov 5–10)

One branch + PR per task, command output pasted in the PR. CPU only; no edits to `participant/` (Ismail F. owns it until the final submission).

| # | Task | Owner | Pass when | Due | Status |
|---|---|---|---|---|---|
| N1 | **Migu status:** AIQ balance, card-hours, instance + `/mnt/storage` still there, is `/mnt/storage/exp027_upload/` there. If storage is gone: rebuild EXP-027 + restore the model from `migu-backups`. | Ismail F. | Numbers written in LOG.md *Compute* | Nov 6 | todo |
| N2 | **One 2-GPU boot:** EXP-027 check run (`VARIANT=exact`) and EXP-029 in the same session (`scripts/exp029_session.sh`, ≈ 0.7–1.2 card-hours + ≈ 0.7 for EXP-027). PASS on EXP-027 → submit **#4**. | Ismail F. | `DONE` line in both logs; #4 accepted on the platform | Nov 7 | todo |
| N3 | **#5 decision:** only if something beats #4 by its pre-registered rule (EXP-029 A2 path or nothing). Default: **#4 stays final and #5 is not used.** | Ismail F. | Decision written in LOG.md | Nov 9 | todo |
| N4 | **Commit EXP-027, EXP-028 and EXP-029 into the repo** (scripts, tests, READMEs, LOG rows), repo `participant/` = exactly what was submitted (FP8 `start.sh` `a7698a02` + #4's `run_inference.py`). | Ismail F. | `git status` clean; sha256 of `participant/` files = Migu | Nov 9 | todo |
| Y5 | **Nov 8: done by Ismail F.** `scripts/eval_600.py --check` reproduces 474 / 472 / 473 and #4 profile exact 0.652 → [docs/eval-600.md](docs/eval-600.md). **Final eval report on the 600 val points from the saved answers** (`outputs/exp022/lp_bf16.jsonl` = #2, `outputs/exp023/lp_fp8_t10.jsonl` = #3/#4's model, gold = `data/val_gold_n100.jsonl` + `data/val_gold_tune500.jsonl`): emotion accuracy, macro-F1, per-class recall, the 16×16 confusion matrix and the top-10 confusions, with `scripts/eval.py`'s own functions. New file `docs/eval-600.md` + the script that writes it (`scripts/eval_600.py`). | Yahya | Numbers reproduce LOG.md: FP8 **472/600** (0.787) with the current parser, bf16 **474/600** (0.790); top FP8 confusion fear → anxiety (15); 2nd run leaves `git diff` clean | Nov 8 | done |
| Z5 | **Nov 8: done by Ismail F.** Every slice command re-checked record for record (laptop files are CRLF, Migu files LF); `t10_input.jsonl` + the saved 600-point answers now tracked via `.gitignore` exceptions. **Data docs for the slices added since Oct 12:** `tune500`, `n100`, `t10` (= the 600 points = n100 + tune500), how each was made (exact `convert_val.py` command), plus the EXP-029 splits (seeded 6,000-point sample of SFT train, the 768 SFT dev points held out from fine-tuning). Update `docs/data-format.md` §6. | Zakaria | Each command reproduces the tracked file byte for byte (paste `sha256sum`) | Nov 8 | done |
| E6 | **Nov 8: done by Ismail F.** (a) [docs/latency.md](docs/latency.md); (b) how-to.md §3b FP8 note, §9 test table, new §11 (eval_600) and §12 (GPU sessions). **Nov 5: Elly left the team → split.** (a) **Latency table** for submissions #1–#3 (+ #4 when known) from `performance_report.json` / LOG.md (mean, median, p95, hardware) → **Yahya** (he owned latency before). (b) **`docs/how-to.md` up to date**: FP8 `start.sh`, `exp025`/`exp027`/`exp029` session commands, `tests/` commands → **Zakaria**. | Yahya (a), Zakaria (b) | (a) every number has a LOG.md row or a report file behind it; (b) every command in how-to.md was run once (CPU ones) or matches the script's own usage text (GPU ones) | Nov 8 | done |
| M2 | **Nov 8: done by Ismail F.** Write-up matches `experiments/exp019-qlora-full/configs/` and LOG.md. Found: the 3 files in `train/configs/` are the Oct 5 drafts (`qwen3_nothink`, cutoff 4096, accum 8) → now labelled PLANNING DRAFT with a pointer to the real configs; same note on `docs/training-plan.md`. **Training write-up check:** read `experiments/exp019-qlora-full/README.md` and the README training section against `train/configs/*.yaml` and LOG.md (template `qwen3` + `enable_thinking: false`, 350-char reply filter, LoRA r16/α32 all-linear, NF4, lr 1e-4 cosine, 1 epoch, 2-GPU DDP, merge step, GPUs-break-after-training rule). Fix anything wrong; O3 still welcome if you have the EXP-006–009 prompt files. | Ismail M. | PR lists each claim checked and its source file | Nov 8 | done |
| O6 | **Nov 8: Elly's section rewritten by Ismail F.** (task → who did it). Others: still send corrections. **Everyone:** read your own section of `docs/team-work.md` (last updated Oct 12) and send Ismail F. what is missing or wrong. Ismail F. rewrites Elly's section to show who actually did each infrastructure task (E1–E5 were done or closed by Ismail F.). | All | Message sent | Nov 8 | doing |
| F5 | **README pass for the expert review:** results table up to #4, what each submission changed and why, links to every experiment folder. Then review by one teammate. | Ismail F. (review: Yahya) | Every number in the README has a LOG.md row or an experiment README behind it | Nov 10 | todo |

## History: where we were (Oct 13)

- **Oct 13: no submission #3. Submission #2 stays our entry.** Two last no-training ideas were tested against rules fixed before each run: the anxiety-bias correction (EXP-022: +1.4 points in-sample, no label changed out of sample) and FP8 + that correction (EXP-023: profile micro-F1 −2.1 points on 500 points → NO-GO). The tag filter stays on. **No more GPU work is planned** (≈ 3.1 card-hours left): nobody boots Migu without asking Ismail F. Details: `experiments/exp022-emotion-bias/`, `experiments/exp023-fp8-bias/`.
- **What's left for the team (Oct 13–19):** O5 (everyone: check your own section of `docs/team-work.md` by Oct 15), O3 (Ismail M., optional: EXP-006–009 prompt files by Oct 16), F3 (expert-review evidence, Ismail F.).

- **Oct 12 (late): EXP-022 ready, no GPU used.** Organisers say generated training data is allowed (answer #11), but P2 stays parked: rare classes are 4.3% of val and a run would use all the card-hours left. First, the no-training levers: an anxiety-prior correction from label logprobs (b tuned on a new 500-point slice `tune500`) + the tag filter off + FP8 vs bf16, all from **one 2-GPU session** (`scripts/exp022_session.sh`, ~0.8 card-hours). Scripts tested on a stub server. **Oct 13: run it, then pick submission #3.**
- **Oct 12: team tasks closed out; Ismail F. finishes the rest himself** (FP8 test, card-hour ledger, expert-review package F3). Open E/Z/O rows are marked dropped or reassigned with the reason.
- **Oct 12: EXP-020 FP8 tested** (organisers allow quantisation). Latency **1.55 → 1.16 s** (−25%), but n50 accuracy 0.80 → 0.76 and profile micro-F1 ≈ −0.05 on both slices: a certain cost of ≈1.3 points vs an unknown latency gain. **Submission #2 stays unless the latency formula makes FP8 worth it** (asked the organisers). FP8 = `participant/start_fp8.sh` on Migu, ready.

- **Oct 11: P1 error review done → no emotion-targeted P2.** The 17 emotion errors of the full model on n50 + n50b: 8 model wrong, 6 arguable, 3 gold wrong (`docs/p1-error-review.md`). The real errors are worried-then-relieved messages, where the training gold itself is split (anxiety 38% vs relaxed 24%), so retraining would trade them for new errors. Details and data: `docs/p2-imbalance-plan.md`. **Submission #2 stays the model to beat; card-hours go to latency (E4/P3).**

- **Oct 9: Submission #2 is in = the T7 full fine-tuned model** (T9 PASS, T10 done). Plain `start.sh` on 100 val points: n50 acc **0.80**, n50b **0.84**, ROUGE-L 0.267 / 0.292 (same as T7 within noise); validate.py 0 errors; latency mean **1.55 s** (sub #1: 1.41 s). On Migu now: `/root/participant/model` = full model, `/root/base_qwen3_8b` = base (all training configs point there), `/root/merged/pilot_ddp` = fallback. Official score pending. **Next: P1 (no GPU), then decide on P2.**

- **Oct 8: Elly is back** and takes back infrastructure (from Ismail F.) and latency (from Yahya). His tasks: see **Infrastructure (Elly)** below. T2/T3/T4 were finished while he was out; details in the T rows and LOG.md.
- **Oct 8: the whole fine-tuning chain is done (T5 pilot on 2 GPUs, T6 gate PASS, T7 full run), all run by Ismail F.** Rule learned: **train in one boot, eval in a fresh boot** (GPUs break when a training process exits).
- **Oct 8: T7 done — full model is the new best.** vs pilot / V1.8 (T4): n50 acc **0.80** (0.68 / 0.54), macro-F1 0.510, profile micro-F1 0.489 / exact 0.453, ROUGE-L **0.269** (0.237 / 0.174), BLEU-4 0.130; n50b acc **0.86** (0.78 / 0.62), macro-F1 0.597, profile 0.477 / 0.500, ROUGE-L **0.293** (0.259 / 0.197), BLEU-4 0.151. Reply median 150 / 106 (gold 146 / 100.5). Train 1 h 48 min, 2.31 samples/s. Merged: `/root/merged/full`. **GPUs break after a training process exits** (`Failed to initialize NVML: Unknown Error`, twice) → eval in a fresh boot. Latency mean 1.66 / 1.50 s (median 1.47 / 1.30; T4 1.32 / 1.52), 0 parse failures. **→ Submission #2 candidate.**
- **Oct 8: Zakaria's B2/B6 reviewed.** Data is correct (val numbers match an independent recount), but the stats doc is hand-edited, real token counts are never printed, and one chart is blank. Real Qwen3 ratio on val ≈ **1.6 chars/token** (not 1.4, not 2+). His tasks are now spelled out step by step in **Dataset (Zakaria)** below + [docs/briefs/zakaria.md](docs/briefs/zakaria.md). **P4 is moved up to Oct 10 (= Z1)** so truncation is in before T9/submission #2. New research task **Z4** (due Oct 12) picks the long-reply filter + augmentation recipe for P2.

- **Submission #1 (V0) is in** (Oct 4). Official score pending.
- **Prompt phase is finished.** 12 experiments (EXP-005 to EXP-016): no prompt beats V0 on emotion beyond noise, but **V1.8** wins clearly on profile, confirmed on an untouched val slice (n50b). V1.8 is frozen and is now `participant/run_inference.py` on `main` (V0 = commit `e5209d2`).
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
| **Captain · training lead (QLoRA) · Migu & infrastructure · classification & prompt** | **Ismail F.** | Captain: all submissions, merging PRs, organiser questions. Migu & infrastructure: built the whole Migu setup: instance, official image, vLLM serving, submission #1 (B7, B9), training venv, eval harness, 2-GPU DDP, merges, private images (T2, T3, E1). Training lead: SFT data rebuild + sign-off (T1), V1.8 vLLM baseline (T4), the full QLoRA pipeline: pilot, gate, full run (T5–T7). Classification & prompt: model choice (EXP-003/004), EXP-005–016 → V1.8, `participant/`. |
| Training | **Ismail F.** (lead, T1–T7) · Ismail M. (P2) | Ismail M.: P2 second run, P3 with Elly. |
| Dataset | **Zakaria** | val/train splits, data stats, truncation of long conversations |
| Evaluation | **Yahya** | `validate.py`, `eval.py`, `analyze.py` |
| Infrastructure + latency | ~~Elly~~ **Nov 5: Elly left the team** → infrastructure: Ismail F.; latency docs: Yahya | Migu instances + private images, 2-GPU launch, card-hour ledger, submission check (T9), latency measurement + P3 |

## Phase 3a: V2 — fine-tuning on Migu (Oct 6–11)

**No AutoDL: everything (training + evals + submissions) runs on the Migu instance (2× RTX 4090).** The 28.67 card-hours left are the hard budget. Log the estimate in LOG.md **before** every GPU job and the actual after.

| Job | Card-hours (est.) |
|---|---|
| T4 V1.8 baseline on vLLM (n50 + n50b) | ~0.3 |
| T5 pilot QLoRA (2k × 1 epoch) | ~1 (actual: ≈0.8 for the stopped 1-GPU run on Oct 7 + ≈0.5 for the 2-GPU rerun on Oct 8) |
| T6 pilot eval | ~0.3 (actual ≈0.6: ran twice after an ssh drop) |
| T7 full QLoRA (14.9k × 1 epoch, 18.7M tokens) | ~3–3.5 (actual: 1 h 48 min on 2 GPUs ≈ 3.6, + ~0.5 h with dead GPUs before shutdown) |
| T8 checkpoint evals | dropped (0) |
| T9 + submissions | ~1.5 |
| P3 latency test (FP8, EXP-020) | ~0.5 (one 2-GPU boot, Oct 12) |
| **Planned total** | **~10–11** (P2 second run: +~6 if needed) |
| **Reserve, never touch** | **≥ 5** |

Rules: installs, downloads, file copying and data prep with 无卡开机 (*wú kǎ kāijī*, no GPU). Turn the GPU instance off the moment a job ends. Train in the **separate venv `/opt/train_venv`** (built on top of `conda_env` with `--system-site-packages`; launch with `/opt/train_venv/bin/llamafactory-cli`) so the submission env (`conda_env`) and the saved submission image stay untouched; clone the instance or save a separate private image for training.

| # | Task | Owner | Depends on | Due | Status |
|---|---|---|---|---|---|
| T1 | **Rebuild SFT data with the frozen prompt.** `train/build_sft.py` imports SYSTEM_PROMPT + `conversation_text()` from `participant/run_inference.py` (V1.8), `memory_refs: []`, prompt field order, val excluded; skips 7 gold replies with leaked `<think>` reasoning. `train/check_sft.py` = sign-off (val leakage, targets, prompt identity, exact token lengths, pilot subset). **Result (Oct 6, PASS):** 17,745 train + 905 dev (5% of train conversations); 660 histories truncated (`--max-user-tokens 3500`); total tokens p50 1,110 / p95 3,597 / max 5,009; train sum 25.2M; pilot = `migu_pilot2k` (2,000, seed 42). Output: `data/sft/` sharegpt + `dataset_info.json`. | Ismail F. | — | Oct 6 | done |
| T2 | **Migu training setup** — venv `/opt/train_venv` on `conda_env` (torch 2.10.0+cu128 kept via constraint; LLaMA-Factory 0.9.5, bitsandbytes 0.50.2); model at `/root/participant/model`; data in `/mnt/storage/sft` + `/mnt/storage/val`; configs in `/mnt/storage/configs`. Smoke test (10 steps, 1 GPU) PASS: 4-bit load, loss 2.04→1.30, 1.36 samples/s. Template fixed to `qwen3` + `enable_thinking: false` (empty think block in prompt, masked). `torchrun` shim added to the venv for 2-GPU DDP. Lessons: conda clone broke torch (pip pulled cu130) → venv instead; `FORCE_TORCHRUN` used conda_env's torchrun → shim. | Ismail F. | — | Oct 7 | done |
| T3 | **vLLM eval harness on Migu** = `bash /root/participant/start.sh <input> <outdir>` (V1.8 `run_inference.py` installed, V0 kept as `run_inference_v0.py`) + `scripts/validate.py` + `scripts/eval.py` in `/mnt/storage/scripts` (run with `/opt/train_venv/bin/python`). BERTScore skipped (no `bert-base-chinese` on the instance yet). | Ismail F. | T2 | Oct 7 | done |
| T4 | **V1.8 baseline on vLLM** (base Qwen3-8B, no training). n50: acc **0.54**, macro-F1 0.238, profile micro-F1 0.352, exact 0.287, ROUGE-L **0.174**, BLEU-4 0.021, latency mean 1.32 s. n50b: acc **0.62**, macro-F1 0.336, profile 0.323, exact 0.287, ROUGE-L **0.197**, BLEU-4 0.068, latency 1.52 s. 0 format errors. **This is the bar for T6/T10.** | Ismail F. | T3 | Oct 7 | done |
| T5 | **Pilot QLoRA:** 2,000 ex (`migu_pilot2k`) × 1 epoch, LoRA r=16 α=32 all-linear, lr 1e-4 cosine, `cutoff_len 6144`, template `qwen3` + `enable_thinking: false`. Oct 7 1-GPU run stopped by accident at step 179 (checkpoint-125 kept). **Oct 8 rerun on 2 GPUs** (`configs/pilot_ddp.yaml`, DDP, global batch 8, 250 steps): train_runtime 892 s, 2.24 samples/s, train_loss 1.315. Merged → `/root/merged/pilot_ddp`. T7 estimate ≈ 1.85 h wall ≈ 3.7 card-hours. | Ismail F. | T1, T2 | Oct 8 | done |
| T6 | **Pilot gate: PASS (Oct 8).** Merged pilot (`/root/merged/pilot_ddp`) via `env VLLM_MODEL_DIR=... VLLM_SERVED_MODEL_NAME=model bash /root/participant/start.sh` + `eval.py`. n50: acc **0.68** (T4 0.54), macro-F1 0.339, profile micro-F1 0.410 / exact 0.420, ROUGE-L **0.237** (0.174), BLEU-4 0.104, reply median 128 (gold 146), latency 1.62 s. n50b: acc **0.78** (0.62), macro-F1 0.449, profile 0.340 / 0.380, ROUGE-L **0.259** (0.197), BLEU-4 0.133, median 110 (gold 100.5), latency 1.55 s. 100% valid JSON, anxiety ≈46% of predictions. | Ismail F. | T4, T5 | Oct 8 | done |
| T7 | **Full QLoRA run: done Oct 8.** 14,942 ex × 1 epoch (`migu_train`), 2 GPUs DDP (`configs/full.yaml`, global batch 8, 1,868 steps, checkpoints every 467), 1 h 48 min, 2.31 samples/s. Merged → `/root/merged/full`. n50: acc **0.80**, macro-F1 0.510, profile 0.489 / 0.453, ROUGE-L **0.269**, BLEU-4 0.130, latency 1.66 s. n50b: acc **0.86**, macro-F1 0.597, profile 0.477 / 0.500, ROUGE-L **0.293**, BLEU-4 0.151, latency 1.50 s. The chained eval in `configs/run_t7.sh` died (GPUs lost after training) → rerun on a fresh boot. | Ismail F. | T6 pass | Oct 8 | done |
| T8 | ~~Checkpoint sweep~~ **Dropped (Oct 8):** the final checkpoint is clearly best; differences between checkpoints on 50 samples would be noise. Merge already done in T7. | Ismail F. | T7 | — | dropped |
| T9 | **Submission check: PASS (Oct 9).** No-GPU boot: base → `/root/base_qwen3_8b`, `/root/merged/full` → `/root/participant/model`, 7 training configs repointed to the base. Fresh 2-GPU boot: plain `start.sh` (no env overrides) on n50 + n50b in one 100-line file. n50 acc 0.80 / ROUGE-L 0.267, n50b acc 0.84 / ROUGE-L 0.292 (= T7 within noise); 100/100 ids in order; validate.py 0 errors; 0 parse failures; latency mean **1.55 s**, median 1.35, p95 3.59. vLLM 0.19.0 and run_inference sha = main checked before saving. Organiser sample test + long5 stress run skipped (see LOG). Image saved. | Ismail F. | T7 | Oct 10 | done |
| T10 | **Submission #2 = T7 full model: sent Oct 9, successful.** Official score pending. | Ismail F. | T9 | Oct 10 | done |

### Infrastructure (Elly)

| # | Task | Depends on | Due | Status |
|---|---|---|---|---|
| E1 | **2-GPU DDP launch.** Test the `torchrun` shim (`/opt/train_venv/bin/torchrun` → `python -m torch.distributed.run`, then `export PATH=/opt/train_venv/bin:$PATH`; check `which torchrun`) with a short smoke run on 2 GPUs before T7. The pilot fell back to 1 GPU after two failed launches; 2 GPUs ≈ 1.6 h vs ≈ 3 h for T7. **Done by Ismail F. on Oct 8: the pilot rerun (T5) ran DDP on 2 GPUs.** | — | Oct 9 | done |
| E2 | **Oct 12, closed by Ismail F.: Ismail F. reads the Migu page himself (task list #2).** **Card-hour ledger + shutdown check.** Fill in the Oct 7 actuals in LOG.md from the Migu page; confirm the GPU instance is off after every session. | — | Oct 8 | reassigned |
| E3 | **Oct 12, closed by Ismail F.: the submission image is saved and verified in T9 (Oct 9); no separate training image needed (no P2).** **Private images:** save the training image `npu-ic-train`; keep a clean submission image with `conda_env` verified (vLLM 0.19.0, transformers 4.57.6). | — | Oct 9 | dropped |
| E4 | **Oct 12, closed by Ismail F.: done as EXP-020 (FP8, see P3).** **Latency levers** (T9 numbers: mean 1.55 s, median 1.35, p95 3.59 on the full model; sub #1 1.41 s): `max_new_tokens` 1200, TP2, `max_model_len`. Then P3. | T9 | Oct 12 | done |
| E5 | **Oct 12, closed by Ismail F.: BERTScore stays skipped; it scores replies only, and every comparison so far used BLEU/ROUGE-L.** Put `bert-base-chinese` on the instance (offline, no GPU) so BERTScore runs in `eval.py`. | — | Oct 10 | dropped |

### Dataset (Zakaria)

**Full step-by-step specs, tests and pass criteria: [docs/briefs/zakaria.md](docs/briefs/zakaria.md).** Working with Kimi: paste it the brief's *Rules for every task* + the whole task section, not just the row below. One branch + PR per task (`zakaria/z1-truncation`, …), command output pasted in the PR, no zips / HTML / PNGs. No card-hours needed for any of these.

| # | Task | Touch (only these) | Pass when | Due | Status |
|---|---|---|---|---|---|
| Z1 | **Oct 12, done by Ismail F. (opt-in, switched OFF in the submission):** `fit_history()` + optional config key `max_user_tokens` in run_inference.py, `tests/test_truncation.py` ALL PASS: identical text to build_sft.py on 18,657/18,657 train points; val 24 / n50 0 / n50b 2 truncated; overhead mean 2 ms (counted in latency). With the key absent the output is byte-identical to the old file (stub-server end-to-end check). **= P4. Inference truncation, same rule as training.** Port `build_sft.py` `fit_history()` into `participant/run_inference.py` (copy, don't import): drop oldest turns until `conversation_text()` ≤ `max_user_tokens` Qwen3 tokens (`add_special_tokens=False`), never start on an assistant turn, always keep the last user turn. Opt-in config key `max_user_tokens` (absent = V1.8 byte-identical, no tokenizer loaded); tokenizer from `VLLM_MODEL_DIR` → `EVALUATION_MODEL_PATH` → config `model_path`, loaded once at start-up. Count goes into `performance_report.json`. Must use the same `max_user_tokens` + token counting as `data/sft/build_report.json`. **Don't** add the key to the config (Ismail F. turns it on for T9) and don't touch prompt / parser / `EMOTION_ALIASES` / start.sh. | `participant/run_inference.py`, new `tests/test_truncation.py` | CPU test passes: (a) huge budget = no change on all val; (b) budget 3500 → ≤ 3500 tokens or 1 turn, suffix of original, last user turn kept; (c) **identical text to build_sft.py `fit_history()` on 100% of train points**; (d) counts printed: val ≈ 24 / 2,354 truncated, n50 = 0; (e) input not mutated; (f) overhead ms/sample printed. GPU check is done by Elly in T9. | **Oct 10** | done |
| Z2 | **Oct 12, done by Ismail F.:** compute_stats.py rewritten: real Qwen3-8B tokens on `conversation_text()`, system prompt 355 tokens, thresholds 2048/3500/4096/8192, per-field empty rates, memory_refs coverage, `<think>`-leak ids, out-of-vocabulary row + sum check, no date; two runs give an identical file. All pass values match (37.2% / 61.9 / 74.9 / 51.7%, 98.3%, 1 val + 7 train leaks, reply p50/p90/p99 = 98/336/847; 24 val and 660 train histories > 3500 tokens, as in build_sft.py). **Fix B6: `compute_stats.py` must generate `docs/data-stats.md` by itself with real tokens.** Tokenizer `models/qwen3-8b-tok` required (exit if missing, delete the 1.4 estimate); history tokens = tokens of `conversation_text()` imported from run_inference.py; system-prompt tokens printed (≈ 355); thresholds 2048/**3500**/4096/8192; add per-field empty rates, memory_refs coverage, points per conversation, `<think>`-leak ids, emotion out-of-vocab row + sum check; remove vocab warning, hardcoded date, 4K/500-token note; old PNGs out. | `scripts/compute_stats.py`, `docs/data-stats.md` | 2nd run leaves `git diff` clean; val matches: 37.2% all-empty, empty personality 61.9 / interests 74.9 / style 51.7%, memory_refs 98.3%, 1 val leak (`val_conv_000331_t17`), 7 train leaks, reply p50/p90/p99 = 98/336/847. | Oct 13 | done |
| Z3 | **Oct 12, done by Ismail F.:** §4 = what build_sft.py / check_sft.py write now (with examples), §6 = which files are in git + the command for each (n50/n50b commands checked: they reproduce the tracked slices exactly), val ids in the examples, new §7 known data issues. **Update `docs/data-format.md`:** §4 = what build_sft.py writes now (`data/sft/*.jsonl`, `*_sharegpt.json`, `*_text.jsonl`, `dataset_info.json`, `build_report.json`), §6 = `data/` is gitignored + regenerate command per file (incl. n50b `--exclude … --sample 50 --seed 42`), val ids in val examples, known `<think>` leaks. | `docs/data-format.md` | every path/field exists in current code; PR says what each was checked against | Oct 13 | done |
| Z4 | **Oct 12, done by Ismail F. as EXP-018:** no training change worth ≥ 1 point (reply length already matches gold; cuts only lower BLEU; 350 filter is a good match). Found the tag filter deleting tags the model was trained on → filter-off parser to test at inference. **Research for P2: long-reply filter + augmentation recipe.** Part A, offline: R1 cost of each reply cutoff 150–500/none (emotion counts, length + emotion-mix distance to val); R2 what long replies look like (lists, markdown, paragraphs), train vs val; R3 oracle test: ROUGE-L/BLEU-4 of gold cut to 80/100/150/200/300 chars vs full gold, using eval.py's own functions; R4 trim at sentence boundary vs drop vs LLM rewrite. Part B: R5 one-page methods review (oversampling, paraphrase, LLM generation, public datasets CPED / SMILE / CPsyCoun / ESConv: link, license, label mapping, or "unverified"); R6 Flash prompts + `check_synthetic.py` (format, length, dedup, val-leak, label-agreement hook). No API calls, no GPU, no edits to build_sft.py; Ismail F. runs the 40-call pilot. | new `scripts/reply_length_study.py`, `scripts/check_synthetic.py`, `experiments/exp018-data-research/`, 1 LOG.md row | Memo ends in a **5-line recommendation** (filter, rare-emotion targets, synthetic share ≤ 15%, train tokens → card-hours, fallback if generated/external data is not allowed); every number from script output; scripts run offline. Feeds P1/P2. | **Oct 12** | done |

## Phase 3b: polish (Oct 12–16)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| P1 | Error analysis of the fine-tuned model. **Done Oct 11 (Ismail F.):** review of the 17 emotion errors → 8 model wrong / 6 arguable / 3 gold wrong, no emotion-targeted P2 (`docs/p1-error-review.md`, `docs/p2-imbalance-plan.md`). Tag-filter re-score **dropped**: needs a GPU re-run and the 3 tags are rare in gold (≤ 4% of val points). **Still open (Yahya):** `analyze.py` v4.1 `--compare` T9 vs V1.8 for profile / reply regressions. | Yahya (+ Ismail F.) | Oct 12 | doing |
| P2 | ~~Second training run~~ **Dropped for emotion (Oct 11, P1):** the remaining errors sit where the training labels are split, so a rebalanced run can't fix them without new errors. Reopens only for rare-class generated data if the organisers allow it (O1), with the gate in `docs/p2-imbalance-plan.md` §6. | Ismail M. | Oct 14 | dropped |
| P3 | **Oct 12, closed by Ismail F.: done as EXP-020 = FP8 instead of AWQ (one vLLM flag, no re-quantisation; organisers allow quantisation). Latency 1.55 → 1.16 s, but n50 acc −2 labels and profile ≈ −0.05 on both slices. Submit only if the latency formula makes the speed worth ≈1.3 points.** **AWQ test** (only if the organisers allow quantised models): quantise the merged model (AutoAWQ / llm-compressor), compare FP16 vs AWQ on Migu: latency + accuracy on val50. Switch only if latency drops clearly and accuracy stays within ~2 points. Wait for the latency formula (O1). | Elly + Ismail M. | Oct 15 | done |
| P4 | **Oct 12: done as Z1 (opt-in, off).** Truncation at inference for very long histories (same rule as `build_sft.py`): **moved up, now Z1** (see *Dataset (Zakaria)*) | Zakaria | **Oct 10** | done |
| P5 | **Oct 13: no submission #3.** EXP-022 (bias on bf16) rejected, EXP-023 (FP8 + bias) NO-GO on profile micro-F1. **Submission #3** only if P3 (latency) gives a clear win without losing accuracy | Ismail F. | Oct 16 | dropped |
| P6 | **Fallback:** if T7 fails T9 or scores worse officially, submit the merged pilot (`/root/merged/pilot_ddp`, also beats V1.8 on both slices). V1.8 with base weights stays the last resort. | Ismail F. | Oct 16 | todo |

## Phase 4: finalise (Oct 17–20)

| # | Task | Owner | Due | Status |
|---|---|---|---|---|
| F1 | Pick the final config from n50 + n50b + Migu val50 | All | Oct 17 | dropped (Nov 5: superseded by Phase 5) |
| F2 | Final submission (#4, keep #5 spare) | Ismail F. | Oct 18 | dropped (Nov 5: superseded by Phase 5) |
| F3 | Expert-review evidence (30 pts): README, experiment logs, LOG.md, clean code, docs/how-to.md up to date. **Nov 5: split into Y5, Z5, E6, M2, O6, F5 (Phase 5).** | All | Nov 10 | doing |
| F4 | Buffer: fix failed runs only, no new features | Ismail F. | Oct 20 | dropped (Nov 5: superseded by Phase 5) |

## Ongoing

| # | Task | Owner | Status |
|---|---|---|---|
| O1 | **Answered: quantised models are allowed; generated training data is allowed (answer #11); profile metric = exact match (confirmed after Oct 16).** Ask the organisers (WeChat group): profile metric (F1 vs exact match), are quantised models allowed, memory bank at test time? **+ (Oct 8, decides P2/Z4) may we train on LLM-generated/rewritten data and on public Chinese emotional-dialogue datasets?** | Ismail F. | done |
| O1b | **Oct 12, closed by Ismail F.: Ismail F. asks the latency formula himself (decides EXP-020 / P5).** Ask the organisers: submission mechanism / eval hardware, how card-hours are counted, latency formula. Captain posts in the official group. | Elly | reassigned |
| O2 | **Oct 12, closed by Ismail F.: not needed before the freeze.** `analyze.py` follow-ups: `--force` passthrough for submissions with blank labels; drop "increase temperature" advice (line 402) | Yahya | dropped |
| O3 | **Oct 13, reopened as optional:** if Ismail M. still has them, send the prompt text + outputs of EXP-006/007/008/009 to Ismail F. (a zip in WeChat is fine) **by Oct 16**; Ismail F. commits them into `experiments/`. If not, the audit note stays ('results only, prompt text not archived'). ~~Oct 12, closed by Ismail F.~~ EXP-006/007/008/009: push the full prompt text + run folders into `experiments/` | Ismail M. | optional |
| O5 | Everyone: read your own section of `docs/team-work.md` and tell Ismail F. what is missing or wrong (credit for the expert review) **by Oct 15** → **Nov 5: repeated as O6 (due Nov 8)** | All | dropped |
| O4 | Commit everything from Oct 5: convert_val (`--exclude/--sample`), exp3_compare (retries), analyze.py + tests, participant V1.7/V1.8 into git history, READMEs, LOG. **Nov 5: check what is still missing as part of N4.** | Ismail F. | todo |

## Done (archive)

| # | Task | Owner | Done |
|---|---|---|---|
| B1 | Official baseline run locally (EXP-001), Qwen2.5-7B via API (EXP-002) | Ismail F. | Oct 3 |
| B2 | `convert_val.py` (+ `--exclude` / `--sample` for n50b, Oct 5) | Zakaria / Ismail F. | Oct 4 |
| B3 | `validate.py` (58 tests) | Yahya | Oct 4 |
| B4 | `eval.py`: emotion, profile, BLEU-4 / ROUGE-L / BERTScore | Yahya | Oct 4 |
| B5 | Prompt work: EXP-005 to EXP-016 → **V1.8** frozen | Ismail F. + Ismail M. | Oct 5 |
| B6 | Data stats → `docs/data-stats.md` (review Oct 8: numbers correct, script/doc fixes → Z2) | Zakaria | Oct 4 |
| B7 | Migu setup, val50 run, private image (done by Ismail F. for Elly) | Ismail F. | Oct 4 |
| B8 | Base model: **Qwen3-8B non-thinking** (EXP-003, EXP-004, EXP-006) | Ismail M. / Ismail F. | Oct 4 |
| B9 | **Submission #1** (V0) | Ismail F. | Oct 4 |
| I1 | `train/build_sft.py` first version (rebuilt in T1) | Ismail F. | Oct 4 |
| I4 | BLEU/ROUGE-L/BERTScore in eval.py; per-sample CSV → `analyze.py` v2.2 merged | Yahya | Oct 5 |
| I7 | Thinking vs model size via API instead of Migu (0 card-hours) | Ismail F. | Oct 4 |
| — | n50b independent eval slice | Ismail F. | Oct 5 |
| I3 | Dedicated emotion/profile classifier (small Chinese BERT) | Ismail F. | **dropped**: no time next to fine-tuning; SFT learns the labels directly |
| I5 | vLLM backend + `sorted(EMOTIONS)` patch (done in submission #1); latency script → Yahya; truncation → P4 | — | split |
