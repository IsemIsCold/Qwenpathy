# Log

## Submissions (5 successful runs max; the last successful one counts)

| # | Date | Commit | Model + prompt | Val score (eval.py) | Official score | Notes |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |

**Failed runs** don't use up an attempt, but log them anyway, with the error message:

| Date | Commit | Error | Fix |
|---|---|---|---|

## Compute

**Migu: 20 card-hours total.** A 2-GPU instance probably costs 2 card-hours per hour.

| Date | Who | Job | Estimated | Actual | Remaining |
|---|---|---|---|---|---|
| | | | | | 20 (unused as of Oct 4) |

**AutoDL (team-paid)**

| Date | Who | GPU | Job | Hours | Cost (¥) |
|---|---|---|---|---|---|

**API experiments (development only, never in the submission)**

| Date | Who | Provider | Model | Job | Cost |
|---|---|---|---|---|---|
| 2026-10-03 | Ismail F. | SiliconFlow (.com) | Qwen2.5-7B-Instruct | EXP-002, 3 test samples | free |
| 2026-10-04 | Ismail F. | Hugging Face router (Nscale) | Qwen3-14B non-thinking | EXP-003, 9 val samples | free monthly credit ($0.10), used up |

## Experiments

| # | Date | What | Result | Log |
|---|---|---|---|---|
| EXP-001 | 2026-10-03 | Official baseline, Qwen2.5-0.5B, CPU | Pipeline works; output unusable | [README](experiments/exp001-baseline-qwen0.5b-cpu/README.md) |
| EXP-002 | 2026-10-03 | Qwen2.5-7B via API, official prompt | Correct role, all fields filled, generic replies | [README](experiments/exp002-qwen7b-api/README.md) |
| EXP-003 | 2026-10-04 | Qwen3-14B non-thinking via API, 9 val pts (partial) | Emotion acc 0.44 vs 0.22 baseline; profile over-filled. **Working base model.** | [README](experiments/exp003-qwen3-14b-thinking/README.md) |
