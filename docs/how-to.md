# How to use this repo

Copy-paste commands for every tool, in the order you'll need them. Commands are PowerShell (Windows); on Linux/macOS use `/` instead of `\` and `\` instead of `` ` `` for line breaks.

**Contents**
1. [One-time setup](#1-one-time-setup)
2. [Make val files](#2-make-val-files-convert_valpy)
3. [Get predictions](#3-get-predictions)
4. [Check a submission](#4-check-a-submission-validatepy)
5. [Score a submission](#5-score-a-submission-evalpy)
6. [Log an experiment](#6-log-an-experiment)
7. [Build training data](#7-build-training-data-build_sftpy)
8. [Dataset statistics](#8-dataset-statistics-compute_statspy)
9. [Test the tools](#9-test-the-tools-run_testspy)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. One-time setup

### 1.1 What you need that is NOT in git

| Thing | Where it goes | Where to get it |
|---|---|---|
| Organiser data: `train_public.jsonl`, `val_public.jsonl` | `data\` | The organisers' dataset link (in the project notes) |
| Full val files: `val_input.jsonl`, `val_gold.jsonl` | `data\` | Make them yourself (§2) or ask Ismail F. |
| Model weights (only for GPU runs) | `participant\model\` | ModelScope: `modelscope download --model Qwen/Qwen3-8B --local_dir participant/model` |

`data\` and `participant\model\` are in `.gitignore`. Never commit them. **Exception:** the fixed eval slices `data\val_{input,gold}_{n50,n50b,n100}.jsonl` *are* in git (since Oct 12), so everyone scores on identical files. n50 = the first 50 val points; n50b = 50 points sampled once with seed 42 from other conversations; n100 = n50 followed by n50b. They never change: don't regenerate them. The rest of `participant\` (`start.sh`, `run_inference.py`, `configs\`, `templates\`) **is in git**: it's the code we submit, so you get it with `git pull`.

> **Use the patched `run_inference.py`.** The official one builds the emotion list from a Python *set*, so the order changes every run and the prompt is never the same twice. The copy in this repo uses `", ".join(sorted(EMOTIONS))` (line 405). Training data and every submission must use it, so never replace it with the organisers' original.

### 1.2 Python and an API key (for API experiments)

```powershell
cd $HOME\NPU-IC-MLP2026
Set-ExecutionPolicy -Scope Process Bypass             # allows scripts in this window only
. .\scripts\setup-api-env.ps1 -Participant "$PWD\participant"
```

The script (note the `. ` at the start, it keeps the settings in your window):
- finds Python 3.10+, creates and activates a venv at `$HOME\venvs\migu` (no packages needed);
- asks for your Hugging Face token without showing it and keeps it **in this window only**;
- checks the files above exist;
- sends one tiny test request and explains 401/402/403/400 errors;
- prints the command to run.

Options: `-Model "Qwen/Qwen3-14B:nscale"`, `-Provider siliconflow`, `-SkipCheck`.

**Your token:** huggingface.co → Settings → Access Tokens → *Fine-grained* → tick only "Make calls to Inference Providers". Never paste it in a chat or a file.

### 1.3 Tools that need extra packages

| Tool | Needs |
|---|---|
| `validate.py`, `eval.py` (without BERTScore), `convert_val.py`, `exp3_compare.py`, `build_sft.py` (without `--tokenizer`) | Python standard library only |
| `tests\run_tests.py` | `pip install numpy` |
| `eval.py` BERTScore, `build_sft.py --tokenizer`, `compute_stats.py` token counts | `pip install torch transformers` + a local model folder |

---

## 2. Make val files (`convert_val.py`)

**Owner:** Zakaria · **Why:** the organisers only give us conversations; this turns val into exactly what the test input looks like, plus an answer key.

```powershell
# full val set: 2,354 prediction points
python scripts\convert_val.py --in data\val_public.jsonl --out data\val_input.jsonl --gold data\val_gold.jsonl

# first N points, for cheap experiments
python scripts\convert_val.py --in data\val_public.jsonl --out data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --n 50
```

| File | Content |
|---|---|
| `val_input*.jsonl` | `{id, conversation_id, target_user_turn_id, history}`: history ends on the user message we answer |
| `val_gold*.jsonl` | `{id, response_text, emotion_label, user_profile, memory_refs}`: the correct answers |

Full format spec: [data-format.md](data-format.md).

**Which size?** 50 points to compare ideas quickly (each point is worth 2 accuracy points, so only trust gaps of ~10+ points). 200 points to confirm a winner. The full 2,354 for a final check before a submission.

---

## 3. Get predictions

### 3a. Through an API: fast, no GPU (`exp3_compare.py`)

**Owner:** Ismail F. · **Why:** test prompts and models in minutes for a few cents, with the **official** prompt, history format and JSON parser, so results transfer to the real submission.

```powershell
python scripts\exp3_compare.py --participant "$PWD\participant" `
    --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
    --out experiments\exp005-prompt\v0 --modes nothink --greedy `
    --soft-switch --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 `
    --model "Qwen/Qwen3-8B:nscale"
```

What it does: copies `participant\` to a temp folder, runs every sample through the API, then runs `validate.py` and `eval.py` and writes a summary table.

| Flag | Meaning |
|---|---|
| `--key-env HF_TOKEN` | the **name** of the variable holding your key, never the key itself |
| `--modes nothink` / `think` / both | Qwen3 thinking off / on |
| `--greedy` | temperature 0: same output every run. **Use it when comparing prompts**, otherwise randomness hides the difference |
| `--soft-switch` | switches thinking with `/no_think` tags; needed on Hugging Face |
| `--no-switch` | for models without thinking (Qwen2.5) |

Outputs in `--out`:

| File | What |
|---|---|
| `comparison.md` | one table: accuracy, macro-F1, profile, latency, blanks |
| `out_nothink\submission.jsonl` | the predictions |
| `out_nothink\metrics.json` | full `eval.py` result |
| `out_nothink\raw_outputs.jsonl` | the model's raw text per sample (for reading mistakes) |
| `out_nothink\performance_report.json` | latency, parse failures |

**Testing a new prompt (B5):** copy `participant` to `participant_v1` (ignored by git; save the final prompt text in the experiment folder), edit `SYSTEM_PROMPT` in `participant_v1\run_inference.py`, and pass `--participant participant_v1`. Keep the `{emotions}`, `{personality}`, `{interests}`, `{styles}` placeholders, and write any literal `{` `}` as `{{` `}}`.

> API latency measures the provider's servers, not ours. Only latency measured on the Migu 2×4090 counts.

### 3b. The real thing: the official image (`start.sh`)

**Owner:** Elly · This is exactly what the organisers run:

```bash
bash /root/participant/start.sh data/val_input_n50.jsonl outputs/run1
```

Config lives in `participant/configs/inference_config.json`. For Qwen3 on vLLM, thinking is switched off with:

```json
"api_options": {"chat_template_kwargs": {"enable_thinking": false}}
```

Details and the Migu workflow: B7 in [TASKS.md](../TASKS.md) and §3 of [technical-plan.md](technical-plan.md).

---

## 4. Check a submission (`validate.py`)

**Owner:** Yahya · **Why:** a malformed line scores 0, and a broken file can waste one of our 5 submissions.

```powershell
python scripts\validate.py experiments\exp005-prompt\v0\out_nothink\submission.jsonl --input data\val_input_n50.jsonl
```

| Exit code | Meaning |
|---|---|
| 0 | PASS (warnings, e.g. different line order, are fine) |
| 1 | FAIL: bad JSON, missing/extra field, label not in the official lists, missing/duplicate ids... The summary line says what |
| 2 | the input file itself is broken |

Every check and why it exists: [eval-design.md](eval-design.md).

---

## 5. Score a submission (`eval.py`)

**Owner:** Yahya · **Why:** this is how we decide anything. Same numbers for everyone, no guessing.

```powershell
python scripts\eval.py <submission.jsonl> data\val_gold_n50.jsonl --out <folder>\metrics.json --history ""
```

| Flag | Meaning |
|---|---|
| `--out` | where to write `metrics.json` (default: next to the submission) |
| `--history docs\score-history.md` | append a one-line summary to the shared score table (default). `--history ""` skips it |
| `--force` | score even if validation fails; invalid fields count as wrong (`exp3_compare.py` uses this) |
| `--bert-model <folder>` | local BERTScore model; skipped if missing |

**The numbers that matter** (how to read all of them: [interpreting-eval-output.md](interpreting-eval-output.md)):

| Number | Official? | Useless-model baseline (n50 / full val) |
|---|---|---|
| Emotion accuracy | yes, 25 pts | always-anxiety: 0.32 / 0.319 |
| Emotion macro-F1 | no, but shows if rare emotions are ever right | ≈ 0.03 |
| Profile micro-F1 and exact match | 15 pts, exact metric unknown | empty profile: exact match 0.37 / 0.63 |
| BLEU-4, ROUGE-L, BERTScore | 20 pts (in Yahya's next version) | |
| Reply length vs gold | no, but affects BLEU/ROUGE and latency | gold median 146 chars |

---

## 6. Log an experiment

Every run that informs a decision gets a folder and a README. The expert review (30 pts) looks at exactly this.

```
experiments/expNNN-short-name/
  README.md              goal, setup, exact command, results table, conclusion
  out_*/metrics.json     from eval.py
  out_*/submission.jsonl
  comparison.md          from exp3_compare.py
```

Then add one row to the Experiments table in [LOG.md](../LOG.md), and the cost to the Compute table. Copy the structure of [exp004](../experiments/exp004-model-size/README.md).

---

## 7. Build training data (`build_sft.py`)

**Owner:** Zakaria (written by Ismail F.) · **Used by:** Ismail M. for QLoRA.

```powershell
python train\build_sft.py --train data\train_public.jsonl --inference participant\run_inference.py --out-dir train\data
# optional: exact token counts + pre-rendered text (needs transformers and the model folder)
python train\build_sft.py ... --tokenizer C:\models\Qwen3-8B
```

The key idea: it **imports** the system prompt, label lists and history formatting from the `run_inference.py` we will submit. So the model is trained on exactly the prompt it will see at test time. If you change the prompt, rebuild the data.

| Output | For |
|---|---|
| `train_sharegpt.json`, `dev_sharegpt.json`, `dataset_info.json` | LLaMA-Factory (template `qwen3_nothink`) |
| `train.jsonl`, `dev.jsonl` | any chat-format trainer |
| `*_text.jsonl` | TRL / custom loops (only with `--tokenizer`) |
| `build_report.json` | counts: 17,752 train / 905 dev examples, 0 conversation overlap |

Rules it enforces: split by `conversation_id` (no leakage), `memory_refs` always `[]`, histories over 3,500 tokens lose their oldest turns. **Inference must truncate the same way** (I5, Elly). Use `--field-order` if the prompt changes the JSON key order.

---

## 8. Dataset statistics (`compute_stats.py`)

**Owner:** Zakaria · The result is already in [data-stats.md](data-stats.md); rerun only if the data changes.

```powershell
python scripts\compute_stats.py --train data\train_public.jsonl --val data\val_public.jsonl --out docs\data-stats.md
```

---

## 9. Test the tools (`run_tests.py`)

**Owner:** Yahya · Run after changing `validate.py` or `eval.py`. Needs `data\val_input.jsonl`, `val_gold.jsonl` and the `_n50` pair, plus numpy.

```powershell
python tests\run_tests.py          # ends with: ALL TESTS PASSED (64/64)
```

---

## 10. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `The term '.\scripts\...' is not recognized` | you're not in the repo folder, or the file isn't in `scripts\` | `cd $HOME\NPU-IC-MLP2026`; check with `ls scripts` |
| `Set your API key first: $env:XYZ="..."` | `--key-env` got the token instead of the name, or the variable is empty | `$env:HF_TOKEN = Read-Host "token"`, then `--key-env HF_TOKEN` |
| `Not found: ...\configs\inference_config.json` | `participant\` is nested twice, or the file became `.json.txt` | re-extract `participant` into the repo root; `Get-ChildItem participant -Recurse` |
| `Illegal characters in path` | you pasted a placeholder like `<folder>` | use a real path |
| `401` / `402` / `403` | wrong key / no credit / blocked or token lacks the Inference permission | new fine-grained token; check Hugging Face billing |
| `400 model not supported` | model name or provider suffix wrong | check the model page's provider list, e.g. `Qwen/Qwen3-8B:nscale` |
| Chinese text garbled in a file | written with PowerShell `>` (UTF-16) | write files from Python with `encoding="utf-8"` |
| `Connection was reset` on `git pull` | GitHub blocked | git proxy, see the README |
