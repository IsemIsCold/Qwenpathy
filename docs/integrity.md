# Integrity and compliance

What we used, what we didn't, and how each claim can be checked in this repo. Written for the expert review (30 points: integrity, code quality, platform records).

## 1. Data

| Data | How we used it | Check |
|---|---|---|
| Official **train** set (`train_public.jsonl`, 8,338 conversations) | The only source of training data. `train/build_sft.py` turns it into SFT examples using exactly the inference prompt. 5% of train conversations (seed 42) are held out as a dev set. | `data/sft/build_report.json` |
| Official **val** set (`val_public.jsonl`, 900 conversations) | **Evaluation only, never trained on.** Two fixed slices of 50 points (n50 = first 50; n50b = 50 points sampled once with seed 42 from other conversations) for every comparison. | `train/check_sft.py`: 0 val conversations in the training data (`data/sft/check_report.json`) |
| Test set | Never seen. Only the organisers run it, through our `start.sh`. | — |
| External datasets | **None.** We considered public Chinese emotional-dialogue datasets and asked the organisers, but used none. | `train/build_sft.py` reads only `train_public.jsonl` |
| LLM-generated or rewritten training data | **None.** We tested DeepSeek V4.1-Flash as a possible data tool (EXP-017) but never generated training data with it. | EXP-017 README; `build_sft.py` |

**Data we filtered out of training, and why** (`build_report.json`):
- 7 training examples whose gold reply contains the data generator's leaked `<think>…</think>` reasoning (not a valid reply).
- 2,940 examples with a gold reply longer than 350 characters, so the model learns reply lengths close to the val distribution (train reply median 129, val 98).
- One val gold (`val_conv_000331_t17`) has the same `<think>` leak. It's not in n50 or n50b, so it doesn't affect any reported number.

## 2. The submission runs fully offline

- `participant/start.sh` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, starts a **local** vLLM server on `127.0.0.1`, and `run_inference.py` only talks to that server.
- No external API, no internet access, no key. Before saving the image (T9, Oct 9) we checked that no API token is in `~/.bashrc`, `~/.profile` or the image.
- Weights: the base Qwen3-8B with our LoRA merged into it (bf16 safetensors in `participant/model`). No adapters or extra downloads at run time.

## 3. Where we used external APIs (development only)

API models were used **only during development, on our own laptops**, never in the submission and never to produce training data:

| What | Why | Where |
|---|---|---|
| Qwen2.5-7B, Qwen3-8B / 14B via SiliconFlow and the Hugging Face router | Choose the base model and thinking vs non-thinking mode, and compare prompts without spending GPU hours | EXP-002 to EXP-016 |
| DeepSeek V4-Pro / V4.1-Flash via the Hugging Face router | Test whether a stronger model could serve as a data tool (EXP-017). Results reported; nothing from it went into training. | EXP-017 |

All API runs and their cost are listed in `LOG.md` → *API experiments*.

## 4. Model and software licences

| Component | Licence | Role |
|---|---|---|
| Qwen3-8B (Alibaba Qwen) | Apache 2.0 | Base model |
| LLaMA-Factory 0.9.5 | Apache 2.0 | QLoRA training |
| vLLM 0.19.0 | Apache 2.0 | Inference server (the organisers' public image) |
| bitsandbytes, PyTorch, Transformers, PEFT | MIT / BSD / Apache 2.0 | Training stack |

## 5. Rule-based parts of the output (disclosed)

The model writes the whole JSON answer. `run_inference.py` then applies a few **fixed, deterministic rules** (no model, no data lookup):

- **`memory_refs` is always `[]`.** The official baseline does the same: the test set has no public memory bank, so no memory id can be valid.
- **Emotion alias map (`EMOTION_ALIASES`):** if the model invents a label outside the 16 allowed ones (e.g. "frustration"), it's mapped to the closest allowed label instead of being left blank. Since fine-tuning, the model almost never invents labels.
- **Profile tag filter (`OVERPREDICTED_TAGS`):** three tags the prompted base model over-predicted (`emotionally_stable`, `study_exam`, `colloquial`) are removed. Measured in EXP-013 and confirmed on n50b (EXP-016).
- Invalid fields are normalised to empty, as in the official parser.

## 6. How the numbers were produced

- Every score in the repo comes from `scripts/eval.py` (accuracy, macro-F1, profile F1 / exact match, char-level BLEU-4 and ROUGE-L), run on the fixed n50 / n50b slices. Latency comes from the official `performance_report.json`.
- Every model was evaluated through the submission's own `start.sh` on Migu, greedy decoding, so dev numbers and the submission use the same code path.
- 50-point slices are small (about ±7 accuracy points of noise). Every decision needed a result on **both** slices, and n50b was never used for tuning.
- Negative results are kept, not hidden: rejected prompts (EXP-010 to EXP-012), the failed FP8 quality gate (EXP-020), and the decision not to run a second training run (`docs/p2-imbalance-plan.md`).

## 7. Platform records (Migu 咪咕同学, Mīgū Tóngxué)

- Compute: 30 card-hours (voucher). Spending is logged per session in `LOG.md` → *Compute*, reconciled against the Migu billing page on Oct 12.
- Submissions: 2 successful (Oct 4, Oct 9), both logged in `LOG.md` with the model, prompt version and val scores.
- Incidents, reported as they happened:
  - **Storage, Oct 8:** while our instance was switched off, `merges.txt` files in 3 checkpoint folders on the shared `/mnt/storage` were replaced by OpenSSL-encrypted copies (`merges.txt.enc`). Nothing else was touched, and the merged models on `/root` were intact. We reported it to the organisers, and since then tokenizer files are always copied from the base model.
  - **GPUs lost after training (Oct 8, twice):** within a minute of a training process exiting, the GPUs failed (`Failed to initialize NVML: Unknown Error`). Rule since: train in one boot, evaluate in a fresh boot.

## 8. Team and tools

- Who built what: `docs/team-work.md` and the owner column in `TASKS.md`.
- AI assistants (Claude, Kimi) were used for code review, documentation and research during development. Every number in this repo comes from our own scripts' output on our own runs, and every claim taken from AI-written research was checked against our data first (see `docs/p2-imbalance-plan.md` §7 for an example of corrections).
