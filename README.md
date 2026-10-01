# M-Zone AI+ Contest: Emotional Companion Dialogue Model

Our entry for the **2026 动感地带AI+高校创智计划 (Dònggǎn Dìdài AI+ Gāoxiào Chuàngzhì Jìhuà)**, the M-Zone AI+ University Innovation Program. We're in the technical track, topic **数字人综合情感陪伴对话模型 (shùzìrén zōnghé qínggǎn péibàn duìhuà móxíng)**, a digital-human emotional-companion dialogue model.

**Team:** Ismail F. (captain) · Ismail (master's) · Zakaria · Yahya · Elly
**Faculty advisor:** Prof. Wang Fang
**First-round deadline:** **Oct 20, 2026, 23:59 China time.** Code freeze is **Oct 18**.

---

## The task in one paragraph

For each test sample, we get a Chinese conversation that ends on a user message. Our container must output four fields for it:

| Field | What it is |
|---|---|
| `response_text` | The next assistant reply, in Chinese |
| `emotion_label` | The emotion of the latest user message, one of 16: `joy, gratitude, relaxed, care, pride, neutral, surprise, mixed, sadness, loneliness, anxiety, anger, fear, disgust, shame, helplessness` |
| `user_profile` | Multi-label: `personality_traits` (10 options), `interests` (12), `style` (11) |
| `memory_refs` | List of `mem_XXXXXX` IDs. The official baseline outputs `[]` because the test set has no public memory bank. |

A missing or invalid field scores 0 for that item. The latency of the first 100 samples is also scored.

**Input line:**
`{id, conversation_id, target_user_turn_id, history: [{turn_id, role, content}]}`

**Output:** one JSON object per line in `RESULT_DIR/submission.jsonl`

## How the organisers run us

```bash
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

- It runs in the conda env `conda_env`, **offline**: no internet and no external APIs.
- There are two backends. `transformers` loads the model from `participant/model`. `openai_compatible` has `start.sh` launch vLLM.
- The evaluation hardware is probably 2× RTX 4090 (24 GB each). We're waiting for the organisers to confirm.

## Repo layout

```
participant/          official entry point (start.sh, run_inference.py) — what gets shipped
  model/              model weights — NOT in git (see .gitignore)
prompts/              system prompts + per-emotion guidance
scripts/
  convert_val.py      val set -> test-format input + gold labels   (Yahya)
  validate.py         checks a submission.jsonl is well-formed      (Yahya)
  eval.py             scores predictions against gold labels        (Zakaria)
train/
  build_sft.py        train set -> chat-format SFT data             (Ismail M.)
  lora_train.py       LoRA fine-tuning script                       (Ismail M.)
data/                 contest data — NOT in git, everyone downloads it themselves
outputs/              local run outputs — NOT in git
docs/                 rules notes, findings, decisions
```

## Quick start (local, CPU, ~5 samples)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate modelscope
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir participant/model

python scripts/convert_val.py --n 5 --out data/sample.jsonl --gold data/sample_gold.jsonl
python participant/run_inference.py ...   # see the transformers branch of start.sh for exact args
python scripts/validate.py outputs/submission.jsonl
python scripts/eval.py outputs/submission.jsonl data/sample_gold.jsonl
```

> **Windows:** don't create JSONL files with PowerShell `>`. It writes UTF-16 and garbles the Chinese text. Write them from Python with `encoding="utf-8"`.

## Key docs

- [CONTRIBUTING.md](CONTRIBUTING.md): team rules, branches, and how submissions work
- [TASKS.md](TASKS.md): who owns what, and the deadlines
- [LOG.md](LOG.md): submissions, scores and compute spend
- [docs/briefs/](docs/briefs/): a one-page brief for each person
