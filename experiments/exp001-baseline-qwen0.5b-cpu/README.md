# EXP-001 — Official baseline, Qwen2.5-0.5B-Instruct, local CPU

| | |
|---|---|
| **Date** | 2026-10-03 |
| **Author** | Ismail (captain) |
| **Goal** | Prove the official inference pipeline runs end to end and writes a valid submission. Reply quality was not a goal. |
| **Status** | ✅ Pipeline works · ❌ Output quality unusable (expected for 0.5B) |

## Setup

| Item | Value |
|---|---|
| Machine | Windows laptop, 8 GB RAM, GTX 1050 Ti (not used; CPU only) |
| Python | 3.12 venv (`$HOME\venvs\migu`), matching the platform image |
| Packages | `torch` (CPU wheel), `transformers`, `accelerate` from `participant/requirements.txt`; `modelscope` for download |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` from ModelScope → `participant/model` |
| Backend | `transformers` (config unchanged from the organisers': `torch_dtype: auto`, `device_map: auto`, `max_new_tokens: 1200`, greedy) |
| Prompt | Official baseline prompt in `run_inference.py`, unchanged |
| Test data | `test_inference_data.jsonl` (3 samples, from the organiser package) |
| Offline | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |

Command (from `participant/`, on Windows, so `start.sh` was bypassed; with the `transformers` backend it only runs this line):

```powershell
$env:EVALUATION_BACKEND="transformers"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python run_inference.py ..\test_inference_data.jsonl ..\out
```

## Results

### Format / pipeline
- `submission.jsonl` validated: **PASS**
- `performance_report.json` validated: **PASS**
- Parse failures: **0 / 3**

### Latency (CPU; not representative of the 4090 eval hardware)
| Sample | Latency |
|---|---|
| test1_000001_t5 | 152.2 s |
| test1_000001_t7 | 223.5 s |
| test1_000002_t5 | 83.1 s |
| **Mean / median** | **152.9 s / 152.2 s** |

### Output quality
| Sample | `response_text` (translated) | `emotion_label` | Problem |
|---|---|---|---|
| test1_000001_t5 | 嗯，我觉得自己可能需要调整一些心态……我会尽量保持冷静 (*ńg, wǒ juéde zìjǐ kěnéng xūyào tiáozhěng yìxiē xīntài… wǒ huì jǐnliàng bǎochí lěngjìng*): "Um, I think I might need to adjust my mindset… I'll try to stay calm." | relaxed | Speaks **as the user**, not the companion |
| test1_000001_t7 | 最近面试碰壁有点多，今天刚面完一家，感觉脑子有点乱，想随便聊聊。(*zuìjìn miànshì pèngbì yǒudiǎn duō, jīntiān gāng miàn wán yì jiā, gǎnjué nǎozi yǒudiǎn luàn, xiǎng suíbiàn liáoliao*): "I've hit a wall in a lot of interviews lately, I just finished one today, my head's a mess, I want to chat." | neutral | Almost a **copy of the user's own message** |
| test1_000002_t5 | 最近挺烦的，为了出片，爱好快变成任务了。(*zuìjìn tǐng fán de, wèile chū piàn, àihào kuài biànchéng rènwu le*): "I've been annoyed lately. Getting good photos has turned my hobby into a chore." | *(empty)* | User's voice; **empty emotion label → scores 0** |

- `user_profile`: empty in all 3 samples
- `memory_refs`: empty in all 3 samples (same as the official baseline; no memory bank is provided)

## Conclusions
1. The full pipeline works locally: model loading, prompt, generation, JSON parsing, validation and both output files.
2. 0.5B is too small to follow the official prompt (role + JSON + 4 fields): it confuses roles, copies user text and leaves fields empty. **Don't tune prompts on 0.5B.** Use it only to smoke-test the pipeline.
3. CPU latency (~150 s/sample) doesn't matter for scoring. The eval runs on GPUs.

## Next steps
- [ ] Start the platform instance without a GPU (无卡开机, *wú kǎ kāijī*): check that `conda_env` exists and record the torch/transformers/vLLM versions.
- [ ] Download Qwen2.5-7B-Instruct onto the instance while it has no GPU.
- [ ] Short GPU run (3 samples, ≈0.2 card-hours) → EXP-002: check role and field filling with 7B.
- [ ] If EXP-002 looks sensible → package → submission #1.

## Files
- `submission.jsonl`: raw predictions
- `performance_report.json`: official timing report
- `inference_config.json`: config used (unchanged from the organisers')
