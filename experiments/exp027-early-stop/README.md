# EXP-027: submission #4 candidate = #3 + stop after `emotion_label` + exact-match profile thresholds

| | |
|---|---|
| **Date** | 2026-10-16 (designed, offline numbers); **2026-11-08 rebuilt** (the Oct 16 files were lost with that work session) |
| **Author** | Ismail F. |
| **Goal** | More profile points now that the organisers confirmed **exact-match** profile scoring, plus lower latency, without touching the model or the reply. |
| **Status** | **Check run PASS (Nov 8) → submitted as #4 on Nov 10 (~11:55).** |

## What changes (only `participant/run_inference.py` and two config keys)

1. **`"stop_after_emotion": true`.** The model always writes `response_text → emotion_label → user_profile → memory_refs` (1,200/1,200 saved answers), and about 41% of the output characters come after the label. vLLM gets `stop: ["\"user_profile\""]`, so generation ends right after the label; the cut answer is closed with `}` and parsed. If a cut answer has no reply or no label (the model wrote the profile first), the same sample is generated again in full, inside the timed region. The log prints `stop after emotion: cut n/N, full-output fallbacks k`.
2. **`"profile_blend": {field: [a, t]}`** overrides the blend stored in `profile_clf/profile_clf.json`. #4 uses the classifier alone (`a = 0`) with the exact-match thresholds personality 0.45, interests 0.30, style 0.35 (fitted on the 600 val points Oct 16; 5-fold CV 0.647, in-sample 0.652).
3. **Lenient JSON retry.** Two of the 600 FP8 answers contain a raw line break or a stray backslash inside the reply; #3's parser then skipped to the inner `user_profile` object and lost reply + label (`val_conv_000383_t7`, `val_conv_000755_t7`). The same object is now retried with `strict=False` and invalid escapes doubled before moving on. Emotion on the 600 points: 472 → 473.

Without the two new keys the output equals #3's except those 2 recovered answers (test C1). `performance_report.json` keys are unchanged.

## Offline numbers on the 600 val points (saved FP8 answers, `scripts/eval_600.py`)

| | #3 (submitted) | #4 candidate |
|---|---|---|
| Emotion | 472/600 = 0.787 | 473/600 = 0.788 |
| Profile per-field exact (scored) | 0.562 | **0.652** (in-sample; CV 0.647) |
| Whole-profile exact | 0.180 | 0.353 |
| ROUGE-L / BLEU-4 | 0.294 / 0.144 | same (same reply) |

+0.09 per-field exact × 15 points ≈ **+1.3 contest points**, plus a shorter output (latency measured in the check run).

## Check run (one 2-GPU boot): pass rules fixed Nov 8, before the run

`bash /mnt/storage/exp027_upload/scripts/exp027_session.sh` (or let `exp029_session.sh` call it first). Run A = plain `start.sh` on t9 with #3 as installed; install the candidate (backup `/root/sub3_backup_exp027`); run B = the same plain `start.sh`. The candidate passes if all hold:

- validate.py 0 errors, 0 parse failures, `complete: true`, full-output fallbacks ≤ 2
- emotion n100 ≥ 0.80 and ≥ run A − 0.02; ROUGE-L n100 ≥ run A − 0.005
- profile per-field exact n100 ≥ run A
- latency mean ≤ 1.05 × run A (same boot)

The Oct 16 draft asked for ≥ 10% lower latency, because the profile metric was still unknown then. With exact match confirmed, the profile gain alone justifies #4, so #4 only has to be no slower. The rule was changed before any run. FAIL → #3's files are restored automatically. PASS → shut down, 重新上传 (*chóngxīn shàngchuán*, upload again) → 去提交 (*qù tíjiāo*, submit) = #4.

## Files

`participant/run_inference.py` (EXP-027 version), `scripts/exp027_install.py` (backup / install / `--restore`), `scripts/exp027_session.sh`, `tests/test_exp027.py`, `tests/stub_vllm.py` (now honours `stop`, `STUB_PROFILE_FIRST`, `STUB_SLEEP_PER_CHAR`), `run_inference_sub3.py` here = submission #3's exact file (sha256 `ec3cfdf5…`).

## Results

**Check run, Nov 8 (one 2-GPU boot, t9 = 100 points, plain `start.sh`): PASS on every rule.**

| | run A = #3 (same boot) | run B = #4 |
|---|---|---|
| Emotion | 0.8400 | 0.8400 |
| Profile per-field exact (scored) | 0.4733 | **0.5267** |
| Profile micro-F1 (not scored) | 0.5348 | 0.3591 |
| ROUGE-L | 0.2756 | 0.2756 |
| Latency mean / median / p95 | 1.125 / 0.957 / 2.358 s | **0.923** / 0.743 / 2.141 s (−18%) |

`stop after emotion: cut 100/100, full-output fallbacks 0`; validate.py 0 errors; 0 parse failures. Micro-F1 falls because the exact-match thresholds leave a field empty unless the classifier is confident; exact match is what the organisers score.

**Submission:** the platform upload failed 4 times (Nov 8–9, no reason shown). Organiser answer (Nov 10): the public image (vLLM, 28.29 GB) plus the used system disk must stay under 50 GB; ours was 70 GB. Moved to `/mnt/storage/sysdisk_archive` (nothing deleted): `/root/base_qwen3_8b`, `/root/merged/`, `/opt/train_venv` (symlink kept), old backups, and a torch 2.14 + CUDA 13 install that an Oct 7 `pip` run had put into conda's base Python 3.13 (never used: the submission runs in `conda_env`, Python 3.12). → 17 GB used; `conda_env` checked (torch 2.10.0+cu128, vLLM 0.19.0). Submitted Nov 10 ~11:55.
