# Training plan V2 (QLoRA on Qwen3-8B, Migu only)

| | |
|---|---|
| **Owner** | Ismail M. (training lead) · review: Ismail F. |
| **Updated** | 2026-10-05 (replaces the B8 draft `training plan.docx`) |
| **Tasks** | [TASKS.md](../TASKS.md) Phase 3a, T1–T10 |
| **Compute** | Migu only, 2× RTX 4090. **No AutoDL.** Budget ≈ 10–11 card-hours of the 28.67 left; ≥ 5 in reserve. |

## 1. Goal

Teach Qwen3-8B the dataset's own conventions (which emotion label, when a profile field is empty, how gold replies sound) from the **18,657 gold-labelled training points**. Prompting is finished: 16 experiments, no prompt beats V0 on emotion beyond noise. Emotion (25 pts) and reply quality (20 pts) can only move with training.

**Success = beats V1.8 on both n50 and n50b, measured on the same Migu vLLM setup (T4).** Not a fixed number: V1.8 already has n50b emotion 0.62 / profile micro-F1 0.300.

## 2. What changed from the B8 draft, and why

| Draft | Now | Why |
|---|---|---|
| Train on outputs of EXP-005/007/008/009, filtered to correct labels | Train on **`train_public.jsonl` gold** via `train/build_sft.py` | Experiment outputs are the base model's own predictions (a few hundred); training on them teaches nothing about gold labels or gold reply style |
| Merge `val_gold_n50.jsonl` into training | **Val never goes into training** | n50 and n50b are our only honest tests; training on them makes every later score meaningless |
| Keep only samples with filled `user_profile` | Keep gold as it is, empty fields included | 62–75% of gold profiles are empty; filtering teaches over-filling (the bug V1.8 fixed) |
| Qwen3-8B-Instruct / download Qwen2.5-7B-Instruct | **Qwen3-8B** (already on the Migu instance) | "Qwen3-8B-Instruct" doesn't exist; Qwen2.5 is a different model; no download needed |
| max length 1024, batch 4 | **cutoff 4096**, batch 1 + grad accum, gradient checkpointing | System prompt ≈ 500–700 tokens + history: 1024 truncates most examples, often the target JSON itself; batch 4 × 4096 does not fit 24 GB |
| "full competition rules + JSON schema" as system message | **V1.8 `SYSTEM_PROMPT` imported** from `participant/run_inference.py`, same history formatter, template `qwen3_nothink`, `memory_refs: []` | Training input must be byte-identical to inference input, or most of the gain is lost |
| 2 epochs | **1 epoch** first (checkpoints every ~25%) | 2 epochs ≈ 10–12 card-hours, almost half the budget |
| Deliver the LoRA adapter | Deliver **merged bf16 safetensors** | The submission serves the model with vLLM; merged weights are simpler and safer offline |
| Success: emotion ≥ 0.50, profile micro-F1 ≥ 0.28 | Beat V1.8 on n50 **and** n50b | The draft's bar is below what we already have |
| 90/10 random split | 5% dev split **by conversation** (build_sft default), for the loss curve only | Points from one conversation share history; the real test is n50 + n50b |

Kept from the draft: QLoRA 4-bit NF4, r 16, α 32, dropout 0.05, all 7 projection modules, `paged_adamw_8bit`, cosine schedule, 5% warmup.

## 3. Data (T1)

```bash
python train/build_sft.py --train data/train_public.jsonl \
    --inference participant/run_inference.py --out-dir data/sft \
    --tokenizer /path/to/Qwen3-8B --max-user-tokens 3500
```

- Imports `SYSTEM_PROMPT`, `EMOTIONS`, `PROFILE_LABELS`, `conversation_text()` and `normalize_prediction()` from `participant/run_inference.py` (= V1.8). Nothing is retyped.
- Target = gold JSON in V0/V1.8 field order (`response_text`, `emotion_label`, `user_profile`, `memory_refs`), `memory_refs` forced to `[]`.
- **Fix (Oct 5):** V1.8's tag filter (`OVERPREDICTED_TAGS`) is switched off while building targets. Without that, every gold example containing `emotionally_stable` / `study_exam` / `colloquial` was silently dropped (synthetic test: 48 of 80 dropped before, 0 after).
- Histories longer than 3,500 tokens lose their oldest turns (≈ 3% of train points). Inference must apply the same rule (P4).
- Outputs in `data/sft/`: `train_sharegpt.json`, `dev_sharegpt.json`, `dataset_info.json` (datasets `migu_train`, `migu_dev`), `build_report.json`.
- **Report in the T1 PR:** number of examples, `skipped_invalid_gold` (should be ≈ 0 now), truncated points, emotion distribution, token length p50/p95/max.

## 4. Migu setup (T2, no GPU)

1. Start the instance with 无卡开机 (*wú kǎ kāijī*, no GPU).
2. New env, separate from the submission env `conda_env`:
   ```bash
   conda create -n train_env python=3.11 -y && conda activate train_env
   pip install "llamafactory[torch,metrics,bitsandbytes]"   # use -i https://pypi.tuna.tsinghua.edu.cn/simple if pip is slow
   ```
3. Reuse the Qwen3-8B weights already on the instance (from submission #1). Copy `data/sft/` and the repo.
4. Save as a private image `npu-ic-train`. The submission image stays untouched.

## 5. Pilot run (T5): 2,000 examples × 1 epoch ≈ 1 card-hour

`train/configs/qwen3_8b_qlora_pilot.yaml`:

```yaml
### model
model_name_or_path: /mnt/storage/models/Qwen3-8B     # adjust to the real path
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
quantization_bit: 4
quantization_method: bitsandbytes     # NF4, double quantization on by default

### dataset
dataset_dir: data/sft
dataset: migu_train
eval_dataset: migu_dev
template: qwen3_nothink
cutoff_len: 4096
max_samples: 2000
preprocessing_num_workers: 8
overwrite_cache: true

### output
output_dir: saves/qwen3-8b/pilot
logging_steps: 10
save_steps: 50
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8        # × 2 GPUs = effective batch 16
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
optim: paged_adamw_8bit
bf16: true
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
max_eval_samples: 200
```

Run on both GPUs, log the estimate in LOG.md first:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train train/configs/qwen3_8b_qlora_pilot.yaml
```

Loss is on the assistant turn only (LLaMA-Factory default, `train_on_prompt: false`). Watch: loss falls smoothly; no OOM; note tokens/s, which gives the full-run estimate.

## 6. Merge (T8, also for the pilot)

Merge into the **bf16** base, not the 4-bit one. `train/configs/qwen3_8b_merge.yaml`:

```yaml
model_name_or_path: /mnt/storage/models/Qwen3-8B
adapter_name_or_path: saves/qwen3-8b/pilot
template: qwen3_nothink
finetuning_type: lora
export_dir: models/qwen3-8b-sft-pilot
export_size: 5
export_device: cpu
export_legacy_format: false
```

```bash
llamafactory-cli export train/configs/qwen3_8b_merge.yaml
```

The merged folder is a normal HF model: point `participant/model` (or the T3 harness) at it and vLLM serves it like the base model.

## 7. Pilot gate (T6)

Run the T3 harness (vLLM, `participant/run_inference.py`, greedy, non-thinking) on n50 and n50b. Pass only if **all** hold:

1. 100% parseable JSON, 0 validator errors
2. no emotion collapse: no label > 60% of predictions
3. emotion accuracy clearly above T4 (V1.8 on vLLM) on **both** n50 and n50b
4. reply median length near gold (≈ 100 chars) and latency not much above V1.8's

Also run each slice twice, **with and without the V1.8 tag filter**. The filter was built for the base model's habits; a fine-tuned model may predict those tags correctly. Keep whichever scores higher on both slices. The emotion alias map stays either way.

Fail → fix data/config, rerun the pilot. No full run before a pass.

## 8. Full run (T7): full train set × 1 epoch ≈ 5–6 card-hours

Same YAML without `max_samples`, `output_dir: saves/qwen3-8b/full`, `save_steps` ≈ a quarter of the total steps. Exact hours = training tokens from `build_report.json` ÷ pilot tokens/s; write it in LOG.md before starting. Then merge (§6), evaluate every saved checkpoint on n50 + n50b, keep the best.

## 9. Card-hour budget

| Job | Card-hours |
|---|---|
| T4 V1.8 baseline on vLLM | ~0.3 |
| T5 pilot | ~1 |
| T6 pilot eval (×2: filter on/off) | ~0.4 |
| T7 full run | ~5–6 |
| T8 checkpoint evals | ~0.6 |
| T9 submission check + latency | ~0.5–1.5 |
| P3 AWQ test (if allowed) | ~1 |
| **Total** | **≈ 10–11** |
| Optional P2 second run | +~6, only if ≥ 11 left |

## 10. Timeline

| Date | Step |
|---|---|
| Oct 6 | T1 data build + report · T2 Migu `train_env` |
| Oct 7 | T3 harness · T4 V1.8 baseline · T5 pilot |
| Oct 8 | T6 gate |
| Oct 9 | T7 full run |
| Oct 10 | T8 merge + checkpoint choice |
| Oct 11 | T9 submission check → T10 submission #2 if it beats V1.8 |
| Oct 16 | Fallback: submit V1.8 if nothing beats it |
| Oct 18 | Code freeze |
