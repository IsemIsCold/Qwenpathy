# Technical plan

> Owner: Ismail F. (captain). Last updated: **2026-10-04** (base model → Qwen3-8B). Update this file whenever a decision changes.

## 1. How we're scored (official rules page)

| Part | Points | What it means for us |
|---|---|---|
| Emotion **accuracy** | 25 | Single label, 16 classes. Always-anxiety = 0.319 on val, so that's the floor. |
| Reply quality: **BLEU-4 / ROUGE-L / BERTScore vs the gold reply** | 20 | Reference-based, so gold-like length and phrasing earn points. Fine-tuning on gold replies matters. |
| User profile (average) | 15 | Exact metric (F1 vs exact-match) still unclear. Gold is empty 52–75% of the time per field. |
| Latency (first 100 samples) | 10 | No thinking mode. Keep replies close to gold length. |
| Expert review | 30 | Integrity, code quality, platform records. The repo, experiment logs and docs are evidence. |

## 2. Base model decision

**Qwen3-8B, non-thinking mode** (decided Oct 4 from EXP-003 and EXP-004; was 14B earlier that day).

| Run | Model | Result (50 val points unless noted) |
|---|---|---|
| EXP-001 | Qwen2.5-0.5B (3 samples) | Pipeline works; output unusable (speaks as the user) |
| EXP-002 | Qwen2.5-7B (3 samples) | Correct role, all fields filled; generic replies; `style` always empty |
| EXP-003 | Qwen3-14B thinking vs non-thinking | Thinking: emotion 0.42 vs 0.46, 2.2× slower → **no thinking** |
| EXP-004 | Qwen3-8B vs 14B, non-thinking | Tie on emotion (0.46 each), 8B faster → **8B** |

Why 8B: ties 14B before tuning; ~1.7× lower latency; half the QLoRA time (more iterations before Oct 18); ~16 GB in bf16 fits one 4090 with room for long contexts; smaller image.
Conditions: the organisers must allow Qwen3 (question #4); B7 must confirm it runs in the official image.
Fallback: **Qwen3-14B** with the same pipeline if time allows (only the model name changes); Qwen2.5-7B if Qwen3 is not allowed.

## 3. Inference (submission) — Elly

- Entry point unchanged: `bash /root/participant/start.sh TEST_FILE RESULT_DIR`, backend `openai_compatible` (vLLM, `--tensor-parallel-size 2`; also try TP=1 for latency), merged bf16 weights in `participant/model`.
- Thinking off: `"api_options": {"chat_template_kwargs": {"enable_thinking": false}}` in `configs/inference_config.json`.
- **Patch `run_inference.py`:** `", ".join(sorted(EMOTIONS))` instead of `", ".join(EMOTIONS)` (a Python set, so the order is random per run). Training and inference prompts must match byte for byte.
- **Truncation:** same rule as training. If the rendered user message exceeds 3,500 tokens, drop the oldest turns, never start on an assistant turn, always keep the last user turn (I5).
- Check inside the official image: vLLM ≥ 0.8.5 and transformers ≥ 4.51 (needed for Qwen3), and `conda_env` exists.
- Measure the latency of the first 100 samples on the 2×4090.

## 4. Prompt — Ismail F. (B5)

Changes to the system prompt, each measured on the same 50 val points with `eval.py`:
1. Emotion first in the JSON (decide the label before writing the reply).
2. One-line definitions for confusable labels (anxiety / fear / helplessness; relaxed / neutral; mixed).
3. Base rates: most messages are anxiety or relaxed; rare labels need clear evidence.
4. Profile: leave a field empty unless the conversation gives explicit evidence; describe how to judge `style` (rational / direct / detailed / brief).
5. Reply: pick up one specific detail the user said earlier; match the user's requested length.

If the JSON key order changes, rebuild the training data with the same `--field-order`.

## 5. Training data — `train/build_sft.py` (I1, done)

- Imports `SYSTEM_PROMPT`, `conversation_text()` and the label lists from the submitted `run_inference.py`, so there is no copy to drift.
- Example = [system, user, assistant]; target = the 4-field JSON with `memory_refs: []`, validated by the official parser.
- Split by `conversation_id` (5% dev). Result: 17,752 train / 905 dev examples, 0 overlap, 867 truncated points (estimate).
- Outputs: chat JSONL, LLaMA-Factory sharegpt + `dataset_info.json`, and Qwen3-rendered prompt/completion text (thinking off) with `--tokenizer`.

## 6. Fine-tuning — Ismail M. (I2)

- QLoRA on Qwen3-8B, on AutoDL (one 4090 is enough), sequence cap ≈4k tokens, gradient checkpointing.
- **Template:** Qwen3 *no-think* (LLaMA-Factory `qwen3_nothink`), not the Qwen2.5 `qwen` template, so training sees the same empty `<think></think>` block vLLM adds at inference.
- Loss on the assistant turn only.
- Smoke test first (≈500 examples, ≈1 h): the loss drops, JSON stays valid, 20 val points look right. Then the full run (1–2 epochs).
- Merge the adapter into the **bf16** base and ship plain safetensors (no `peft` at inference).
- **Don't start the full run until the organisers confirm Qwen3 is allowed.**

## 7. Evaluation — Yahya

- `validate.py` (format gate) + `eval.py` (emotion accuracy and macro-F1, profile micro-F1 and exact match, baselines, lengths). 58 tests pass.
- Every experiment: same 50 val points (`val_input_n50.jsonl`), result row in `docs/score-history.md`, log in `experiments/expNNN-*/README.md`.
- **Next:** (1) per-sample mistakes CSV; (2) character-level BLEU-4 and ROUGE-L, because they're 20 official points; (3) LLM reply judge (dev only).
- Decision metrics: emotion accuracy, macro-F1, profile micro-F1, plus BLEU/ROUGE once added.

## 8. Classifier option — Ismail F. (I3)

Fine-tune a small Chinese encoder (e.g. `hfl/chinese-roberta-wwm-ext`) on the 18.6k emotion labels (class weights for the imbalance) plus multi-label profile heads. If it beats the LLM's labels on val, inject its emotion into the prompt and overwrite the LLM's labels. It's fast and runs offline in the image.

## 9. Timeline (deadline Oct 20, 23:59; code freeze Oct 18)

| Dates | Milestone |
|---|---|
| Oct 5–6 | B5 prompt experiments (EXP-005, API, < $1); Migu instance up with Qwen3-8B (B7) |
| Oct 6–7 | **Submission #1**: Qwen3-8B non-thinking + best prompt |
| Oct 7–8 | QLoRA smoke test; BLEU/ROUGE in eval.py; classifier prototype |
| Oct 9–11 | Full fine-tune, merge, eval on val |
| Oct 12–14 | **Submissions #2–#3**: fine-tuned model (+ classifier if it wins) |
| Oct 15–17 | Pick the final config from val scores |
| Oct 18 | Final submission, code freeze |
| Oct 19–20 | Buffer: fix failed runs only |

## 10. Risks

| Risk | Mitigation |
|---|---|
| Qwen3 not allowed | Same pipeline with Qwen2.5-7B |
| Official image's vLLM too old for Qwen3 | Ask the organisers; install a newer vLLM inside our image |
| Train/inference format mismatch | `build_sft.py` imports the inference prompt; sorted emotions; same truncation rule |
| Migu card-hours run out | 无卡开机 (*wú kǎ kāijī*, start without GPU) for setup and downloads; every GPU job estimated in LOG.md first |
| Unknown profile metric | Ask the organisers; track both micro-F1 and exact match |
| `memory_refs` always empty | Ask the organisers; currently 0 points for everyone |

## 11. Open questions for the organisers

See [docs/organiser-answers.md](organiser-answers.md). Most urgent: Qwen3 allowed? Profile metric? `memory_refs` at test time? Submission mechanism and eval hardware?
