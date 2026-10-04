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
| 2026-10-04 | Ismail F. | Hugging Face router (Nscale) | Qwen3-14B non-thinking + thinking | EXP-003, 50 val points per mode | from the $5 credit added Oct 4 |
| 2026-10-04 | Ismail F. | Hugging Face router (Nscale) | Qwen3-8B non-thinking | EXP-004, 50 val points | from the $5 credit |

## Experiments

| # | Date | What | Result | Log |
|---|---|---|---|---|
| EXP-001 | 2026-10-03 | Official baseline, Qwen2.5-0.5B, CPU | Pipeline works; output unusable | [README](experiments/exp001-baseline-qwen0.5b-cpu/README.md) |
| EXP-002 | 2026-10-03 | Qwen2.5-7B via API, official prompt | Correct role, all fields filled, generic replies | [README](experiments/exp002-qwen7b-api/README.md) |
| EXP-003 | 2026-10-04 | Qwen3-14B thinking vs non-thinking via API, 50 val pts | Thinking: no better on emotion (0.42 vs 0.46), 2.2× slower (5.9 s vs 2.7 s). **Thinking dropped.** | [README](experiments/exp003-qwen3-14b-thinking/README.md) |
| EXP-004 | 2026-10-04 | Qwen3-8B vs 14B (non-thinking) via API, 50 val pts | Tie on emotion accuracy (0.46); 8B faster, cheaper to train, smaller image. **Base model: Qwen3-8B.** | [README](experiments/exp004-model-size/README.md) |
| EXP-005 | 2026-10-05 | Prompt versions V0–V3 (B5), Qwen3-8B greedy, 50 val pts | in progress | `experiments/exp005-prompt/` |
