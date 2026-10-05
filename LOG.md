# Log

## Submissions (5 successful runs max; the last successful one counts)

| # | Date | Commit | Model + prompt | Val score (eval.py) | Official score | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-10-04 | `e5209d2` + config fix (push pending) | Qwen3-8B non-thinking, vLLM TP2 32k, official prompt V0, greedy. Image `npu-ic-qwen3-8b-v0` / `v20261004225439` | acc 0.48, macro-F1 0.198, ROUGE-L 0.173, BLEU-4 0.021, profile micro-F1 0.262 (API V0: 0.50) | pending | Latency on Migu 2×4090: mean 1.41 s, median 1.40 s, max 2.01 s; vLLM start-up 109 s cold / 51 s warm; 0 parse failures |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |

**Failed runs** don't use up an attempt, but log them anyway, with the error message:

| Date | Commit | Error | Fix |
|---|---|---|---|

## Compute

**Migu: 30 card-hours total (voucher).** A 2-GPU instance probably costs 2 card-hours per hour.

| Date | Who | Job | Estimated | Actual | Remaining |
|---|---|---|---|---|---|
| 2026-10-04 | Ismail F. | B7: create instance, no-GPU setup + model download, GPU smoke (9) + val50 run, save image, upload | ~0.6 | 1.33 (29.99 → 28.67; the voucher was 30, not 20) | 28.67 |

**AutoDL:** not used (decision Oct 5): all training and evals run on Migu.

**API experiments (development only, never in the submission)**

| Date | Who | Provider | Model | Job | Cost |
|---|---|---|---|---|---|
| 2026-10-03 | Ismail F. | SiliconFlow (.com) | Qwen2.5-7B-Instruct | EXP-002, 3 test samples | free |
| 2026-10-04 | Ismail F. | Hugging Face router (Nscale) | Qwen3-14B non-thinking | EXP-003, 9 val samples | free monthly credit ($0.10), used up |
| 2026-10-04 | Ismail F. | Hugging Face router (Nscale) | Qwen3-14B non-thinking + thinking | EXP-003, 50 val points per mode | from the $5 credit added Oct 4 |
| 2026-10-04 | Ismail F. | Hugging Face router (Nscale) | Qwen3-8B non-thinking | EXP-004, 50 val points | from the $5 credit |
| 2026-10-05 | Ismail F. + Ismail M. | Hugging Face router (Nscale) | Qwen3-8B (non-thinking; one thinking run) | EXP-005 to EXP-016, ~16 runs × 50 points | from the $5 credit (cents per run) |

## Experiments

| # | Date | What | Result | Log |
|---|---|---|---|---|
| EXP-001 | 2026-10-03 | Official baseline, Qwen2.5-0.5B, CPU | Pipeline works; output unusable | [README](experiments/exp001-baseline-qwen0.5b-cpu/README.md) |
| EXP-002 | 2026-10-03 | Qwen2.5-7B via API, official prompt | Correct role, all fields filled, generic replies | [README](experiments/exp002-qwen7b-api/README.md) |
| EXP-003 | 2026-10-04 | Qwen3-14B thinking vs non-thinking via API, 50 val pts | Thinking: no better on emotion (0.42 vs 0.46), 2.2× slower (5.9 s vs 2.7 s). **Thinking dropped.** | [README](experiments/exp003-qwen3-14b-thinking/README.md) |
| EXP-004 | 2026-10-04 | Qwen3-8B vs 14B (non-thinking) via API, 50 val pts | Tie on emotion accuracy (0.46); 8B faster, cheaper to train, smaller image. **Base model: Qwen3-8B.** | [README](experiments/exp004-model-size/README.md) |
| EXP-005 | 2026-10-05 | Prompt versions V0–V3 (B5), Qwen3-8B greedy, 50 val pts | V0 acc 0.50 / macro-F1 0.19 = submission #1. V1 no better (base-rate line hurt rare emotions). | `experiments/exp005-prompt/` |
| EXP-006 | 2026-10-05 | Ismail M.: Qwen3-8B **thinking** mode, n50 | **Invalid run**: empty reply + blank label on sample 1. Thinking stays off. | [README](experiments/exp006-think-8b/README.md) |
| EXP-007 | 2026-10-05 | Ismail M.: English prompt ablation, nothink, n50 | acc 0.48, macro-F1 0.214, profile micro-F1 0.234, exact 0.34. Not adopted. | [README](experiments/exp007-prompt-ablation/README.md) |
| EXP-008 | 2026-10-05 | Ismail M.: emotion-focused prompt, nothink, n50 | acc 0.52 (anxiety-biased, macro-F1 0.141), profiles all empty (micro-F1 0, exact 0.373 = empty baseline). Not adopted. | [README](experiments/exp008-fix-emotion-prompt/README.md) |
| EXP-009 | 2026-10-05 | Ismail M.: profile-fix English prompt (16 definitions, profile + length rules), nothink, n50 | acc 0.46, macro-F1 0.196, profile micro-F1 0.329, exact 0.273, reply median 131.5, latency 3.57 s (API). Profile idea kept (→ V1.8), length rule dropped. | [README](experiments/exp009-profile-fix-prompt/README.md) |
| EXP-010 | 2026-10-05 | V1.3 = V0 + EXP-009 profile + length rules, greedy, 50 val pts | acc 0.42, macro-F1 0.153 (worse); BLEU-4 0.021→0.055, ROUGE-L 0.176→0.184, reply median 157. **Rejected; keep V0.** | [README](experiments/exp010-v1-3-profile-length/README.md) |
| EXP-011 | 2026-10-05 | V1.4 = V0 + length rule only, greedy, 50 val pts | acc 0.44, macro-F1 0.195, profile micro-F1 0.224, BLEU-4 0.051, ROUGE-L 0.174, reply median 164. **Rejected.** Long reply written before the label hurts emotion. | [README](experiments/exp011-v1-4-length-rule/README.md) |
| EXP-012 | 2026-10-05 | V1.5 = V1.4 + emotion-first field order | acc 0.42, macro-F1 0.150, ROUGE-L 0.177, BLEU-4 0.048. **Rejected.** Order doesn't help; length rule dropped. Emotion prompt work stops, V0 stays. | [README](experiments/exp012-v1-5-emotion-first/README.md) |
| EXP-013 | 2026-10-05 | Offline: drop over-predicted tags (emotionally_stable, study_exam, colloquial) from V0 output | profile micro-F1 0.275→0.360, exact 0.067→0.240 on n50. Needs confirmation on another val slice. | [README](experiments/exp013-profile-tag-filter/README.md) |
| EXP-014 | 2026-10-05 | V1.6 = Ismail M.'s English prompt (+3 fixes: no empty emotion, profile vocab added, memory_refs []) | acc 0.44 (care over-predicted 10×), profile micro-F1 0.335, exact 0.313 (+filter: 0.383 / 0.393). Emotion rejected, profile rules kept. | [README](experiments/exp014-v1-6-ismail-m-prompt/README.md) |
| EXP-015 | 2026-10-05 | V1.7 = V0 + V1.6 profile rules + tag filter | acc 0.46, macro-F1 0.204, profile micro-F1 0.373, exact 0.320, ROUGE-L 0.178. Candidate; re-tested on n50b. | [README](experiments/exp015-v1-7-combined/README.md) |
| EXP-016 | 2026-10-05 | V0 vs V1.7 on n50b (independent random slice) | acc 0.60 vs 0.62, profile micro-F1 0.175 vs 0.300, exact 0.107 vs 0.247, ROUGE-L 0.205 vs 0.200. **V1.7 confirmed.** Submission #2 = **V1.8** = V1.7 + emotion alias map (parser only). | [README](experiments/exp016-n50b-confirmation/README.md) |
