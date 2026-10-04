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

**Scoring (official rules page):** 25 emotion accuracy + 20 reply quality (BLEU-4 / ROUGE-L / BERTScore vs the gold reply) + 15 profile + 10 latency = 70 automatic, plus 30 expert review.

**Current approach:** Qwen3-14B in non-thinking mode, served by vLLM on 2× RTX 4090, prompt-tuned and then QLoRA fine-tuned on the training set. See [docs/technical-plan.md](docs/technical-plan.md).

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
participant/          official entry point (start.sh, run_inference.py)  (Elly)
  model/              model weights — NOT in git (see .gitignore)
prompts/              system prompts + per-emotion guidance          (Ismail F.)
classifier/           emotion + profile classifier                   (Ismail F.)
scripts/
  convert_val.py      val set -> test-format input + gold labels     (Zakaria)
  compute_stats.py    dataset statistics -> docs/data-stats.md        (Zakaria)
  validate.py         checks a submission.jsonl is well-formed        (Yahya)
  eval.py             scores predictions against gold labels          (Yahya)
  exp3_compare.py     API experiment runner: Qwen3 thinking vs non-thinking
tests/
  run_tests.py        test suite for validate.py and eval.py          (Yahya)
train/
  build_sft.py        train set -> SFT data matching inference exactly (Zakaria / Ismail F.)
  lora_train.py       LoRA fine-tuning script                         (Ismail M.)
data/                 contest data — NOT in git, everyone downloads it themselves
outputs/              local run outputs — NOT in git
docs/                 technical plan, data format and stats, eval design
experiments/          one folder per experiment: README log + outputs
```

## Who owns what

| Infrastructure | Dataset | Training | Evaluation | Classification + captain |
|---|---|---|---|---|
| Elly | Zakaria | Ismail M. | Yahya | Ismail F. |

Full task list: [TASKS.md](TASKS.md).

## Git in China: "Connection was reset"

GitHub is often blocked. Git doesn't use your VPN's system proxy on its own, so point it there. Use your client's port: Clash Verge defaults to 7897, older Clash versions use 7890.

```bash
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

Or turn on TUN mode in your VPN client, which routes all traffic, git included.

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
- [LOG.md](LOG.md): submissions, scores, compute spend and the experiment index
- [docs/technical-plan.md](docs/technical-plan.md): scoring, model decision, pipeline, timeline, risks
- [docs/data-format.md](docs/data-format.md) and [docs/data-stats.md](docs/data-stats.md): data contracts and statistics
- [docs/eval-design.md](docs/eval-design.md) and [docs/interpreting-eval-output.md](docs/interpreting-eval-output.md): how we evaluate
- [docs/briefs/](docs/briefs/): a one-page brief for each person
