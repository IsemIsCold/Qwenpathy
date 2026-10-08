# M-Zone AI+ Contest: Emotional Companion Dialogue Model

Our entry for the **2026 动感地带AI+高校创智计划 (Dònggǎn Dìdài AI+ Gāoxiào Chuàngzhì Jìhuà)**, technical track, topic **数字人综合情感陪伴对话模型 (shùzìrén zōnghé qínggǎn péibàn duìhuà móxíng)**: a digital-human emotional-companion dialogue model.

**Team:** **Ismail F.** (captain · training lead (QLoRA) · Migu & infrastructure · classification & prompt) · Ismail M. (training support) · Zakaria (data) · Yahya (evaluation) · Elly (infrastructure support, from Oct 8)
**Faculty advisor:** Prof. Wang Fang
**Deadline:** **Oct 20, 2026, 23:59 China time.** Code freeze **Oct 18**.

> **New here? Read in this order:**
> 1. this README (10 min)
> 2. [docs/how-to.md](docs/how-to.md): how to run every tool in the repo, with copy-paste commands
> 3. [docs/team-work.md](docs/team-work.md): what each person built and how it helps your task
> 4. [TASKS.md](TASKS.md): what you should be doing now

---

## The task

For each test sample we get a Chinese conversation that ends on a user message. Our Docker image must output one JSON line with four fields:

| Field | What it is | Points |
|---|---|---|
| `emotion_label` | Emotion of the latest user message, one of 16 labels | 25 (accuracy) |
| `response_text` | The next assistant reply, in Chinese | 20 (BLEU-4 / ROUGE-L / BERTScore vs the gold reply) |
| `user_profile` | `personality_traits`, `interests`, `style`: multi-label, fixed vocabularies | 15 |
| `memory_refs` | Always `[]`: the test set has no memory bank | 0 for everyone |
| Latency | Time per sample on the first 100 samples | 10 |
| Expert review | Integrity, code quality, platform records: **this repo is evidence** | 30 |

The organisers run `bash /root/participant/start.sh TEST_FILE RESULT_DIR` **offline** (no internet, no APIs) on what is probably 2× RTX 4090. We get **5 successful submissions; the last successful one counts.**

## Current approach (Oct 8)

**Qwen3-8B, non-thinking mode, QLoRA fine-tuned on the full training set**, served by vLLM on 2× 4090. Built and run end to end on Migu by Ismail F. (T1–T7 in [TASKS.md](TASKS.md)).

- **Prompt: V1.8, frozen** (`participant_v1_8/`): the official prompt + "leave [] if insufficient information" profile rules + a filter for over-predicted profile tags + a map for invented emotion labels. Chosen over 16 experiments and confirmed on an untouched val slice: [experiments/exp005-prompt](experiments/exp005-prompt/README.md). Training and inference use exactly this prompt.
- **Fine-tuning (V2):** LoRA r=16 / α=32 on all linear layers, 4-bit base (QLoRA), 14,942 training examples × 1 epoch, 2-GPU DDP, 1 h 48 min. Merged into bf16 weights for vLLM.

| Val slice | Metric | V1.8 (base model) | Pilot (2k ex) | **Full QLoRA** |
|---|---|---|---|---|
| n50 | Emotion accuracy | 0.54 | 0.68 | **0.80** |
| n50 | ROUGE-L / BLEU-4 | 0.174 / 0.021 | 0.237 / 0.104 | **0.269 / 0.130** |
| n50 | Profile micro-F1 | 0.352 | 0.410 | **0.489** |
| n50b | Emotion accuracy | 0.62 | 0.78 | **0.86** |
| n50b | ROUGE-L / BLEU-4 | 0.197 / 0.068 | 0.259 / 0.133 | **0.293 / 0.151** |
| n50b | Profile micro-F1 | 0.323 | 0.340 | **0.477** |
| n50 / n50b | Latency mean | 1.32 / 1.52 s | 1.62 / 1.55 s | 1.66 / 1.50 s |

- Submissions: #1 = V0 (Oct 4). **#2 = the full QLoRA model** after the T9 submission check.

Why 8B, non-thinking: [exp004-model-size](experiments/exp004-model-size/README.md), [exp006-think-8b](experiments/exp006-think-8b/README.md), [docs/technical-plan.md](docs/technical-plan.md).

## The pipeline at a glance

```
organiser data (train_public / val_public)
   │
   ├─ scripts/convert_val.py ──► data/val_input*.jsonl + data/val_gold*.jsonl     (Zakaria)
   │                                  │
   │                     participant/run_inference.py  (official, + our prompt)    (Ismail F.)
   │                     or scripts/exp3_compare.py via API, for quick tests
   │                                  │
   │                                  ▼
   │                         submission.jsonl
   │                                  │
   │              scripts/validate.py (format gate) ─► scripts/eval.py (scores)   (Yahya)
   │
   └─ train/build_sft.py ──► SFT data (same prompt as inference) ──► QLoRA on Migu ──► merged weights   (Ismail F.)
```

## Repo layout

```
README.md               you are here
TASKS.md                who does what, by when, status
LOG.md                  submissions, compute spend, experiment index
CONTRIBUTING.md         team rules: branches, PRs, submissions, compute
participant/            the code we submit: start.sh, run_inference.py (patched), configs/   (Ismail F.)
scripts/
  convert_val.py        val set -> test-format input + gold labels          (Zakaria)
  compute_stats.py      dataset statistics -> docs/data-stats.md            (Zakaria)
  validate.py           checks a submission is well-formed                  (Yahya)
  eval.py               scores a submission against gold                    (Yahya)
  analyze.py            error analysis: buckets, points lost, A/B compare   (Yahya)
  exp3_compare.py       runs the official prompt/parser through an API      (Ismail F.)
  setup-api-env.ps1     Windows setup for API runs (venv, key, checks)      (Ismail F.)
train/
  build_sft.py          train set -> SFT data that matches inference        (Ismail F.; spec: Zakaria)
  check_sft.py          sign-off for the SFT data (leakage, targets, token lengths)   (Ismail F.)
tests/
  run_tests.py          tests for validate.py + eval.py                     (Yahya)
  test_analyze.py       42 tests for analyze.py                             (Yahya)
docs/
  how-to.md             how to run everything (start here)
  team-work.md          what each member built, and how it helps the others
  technical-plan.md     scoring, model decision, pipeline, timeline, risks
  data-format.md        every JSONL format we use                           (Zakaria)
  data-stats.md         label distributions, lengths, token counts          (Zakaria)
  eval-design.md        how eval.py is designed                             (Yahya)
  interpreting-eval-output.md   how to read eval.py's numbers               (Yahya)
  organiser-answers.md  questions to the organisers + answers
  briefs/               the original one-page task briefs (Oct 1)
experiments/
  expNNN-name/          one folder per experiment: README + outputs + metrics

NOT in git (see .gitignore):
  data/                 contest data and val files: get them from the team drive / Ismail F.
  participant/model/    model weights
```

## Five-minute setup

You need **Python 3.10+** and **git**. The API path needs no PyTorch.

```powershell
git clone <repo URL>
cd NPU-IC-MLP2026
# put the data/*.jsonl files in place (see docs/how-to.md §1)
Set-ExecutionPolicy -Scope Process Bypass
. .\scripts\setup-api-env.ps1 -Participant "$PWD\participant"
python tests\run_tests.py        # should end with ALL TESTS PASSED
```

Then follow [docs/how-to.md](docs/how-to.md) for whatever you need to run.

## Git in China: "Connection was reset"

Git doesn't use your VPN's system proxy on its own. Point it there (Clash Verge uses 7897, older Clash 7890), or turn on TUN mode:

```bash
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

## Two Windows traps

- **Never create JSONL with PowerShell `>`.** It writes UTF-16 and breaks the Chinese text. Write files from Python with `encoding="utf-8"`.
- **Never paste API keys in chats or commit them.** Keys live in an environment variable for one terminal window only (the setup script does this).
