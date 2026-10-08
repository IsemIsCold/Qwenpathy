# EXP-017: DeepSeek V4 as a "teacher" (data tool for P2)

| | |
|---|---|
| **Date** | 2026-10-07 |
| **Author** | Ismail F. |
| **Goal** | Find a strong API model that follows our V1.8 prompt and output format well enough to **generate or rewrite training data** (P2: shorten long gold replies, add examples for rare emotions). It can never be the submission: the contest runs offline. |
| **Status** | **Done. PASS** for DeepSeek V4.1-Flash. DeepSeek V4-Pro-0813 is unusable through the Hugging Face router. Flash is the P2 data tool, pending the organisers' answer on LLM-generated training data (O1). |

## Setup

- `scripts/exp3_compare.py`: the official prompt, history format and parser, through an OpenAI-compatible API, then `validate.py` + `eval.py`. Prompt = **V1.8** (`participant/`), greedy (temperature 0), seed 42, non-thinking.
- Router: `https://router.huggingface.co/v1` (key in `HF_TOKEN`).
- New option `--api-options <json>`: extra request fields, used to switch thinking off. PowerShell 5 strips inner quotes, so escape them: `'{\"thinking\": {\"type\": \"disabled\"}}'`.

Two stages:

1. **Switch sweep (n9).** Does the provider return clean JSON with thinking off? DeepSeek V4-Pro-0813 on 4 providers × 3 ways of turning thinking off, plus V4.1-Flash. 9 val points (`val_input_n9.jsonl`), so only format and the thinking switch matter here, not the scores.
2. **Teacher test (n50 + n50b).** The model that passes stage 1 on both val slices, compared with V1.8 run the same way (API).

```powershell
python scripts\exp3_compare.py --participant participant `
  --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl --scripts scripts `
  --out experiments\exp017-teacher\v4.1-flash_n50 --modes nothink --greedy `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 `
  --model "deepseek-ai/DeepSeek-V4.1-Flash:novita" `
  --api-options '{\"thinking\": {\"type\": \"disabled\"}}'
```

Same for n50b (`--input data\val_input_n50b.jsonl --gold data\val_gold_n50b.jsonl --out ...\v4.1-flash_n50b`). Sweep runs: same command with `--input data\val_input_n9.jsonl`, the model/provider and switch from the table below, output in `switch/<model>_<provider>_<switch>/`.

## Stage 1: switch sweep (n9)

Switches: `deepseek` = `{"thinking": {"type": "disabled"}}` (DeepSeek's own API field), `tmpl_thinking` = `{"chat_template_kwargs": {"thinking": false}}`, `tmpl_enable_thinking` = `{"chat_template_kwargs": {"enable_thinking": false}}`.

| Model : provider | Switch | Parse failures (of 9) | Reasoning chars | Latency (s) |
|---|---|---|---|---|
| V4-Pro-0813 : baseten | deepseek / tmpl_thinking / tmpl_enable_thinking | 8 / 8 / 8 | 0 | 1.9–2.8 |
| V4-Pro-0813 : deepinfra | deepseek / tmpl_thinking / tmpl_enable_thinking | 8 / 9 / 9 | 0 | 4.5–4.7 |
| V4-Pro-0813 : deepinfra, no streaming | deepseek | 9 | n/a | 5.6 |
| V4-Pro-0813 : novita | tmpl_thinking / tmpl_enable_thinking | 8 / 7 | **≈350** (switch ignored) | 6.5–6.7 |
| V4-Pro-0813 : together | all three | request errors | – | – |
| **V4.1-Flash : novita** | **deepseek** | **0** | **0** | **3.3** |

The first V4-Pro attempt (`../exp017-switchtest/`, novita + `deepseek` switch) also gave 7/9 parse failures.

**V4-Pro-0813 is unusable on the HF router:** on every provider almost every answer fails our parser (the output starts with `_text:` instead of a JSON object), together returns errors, and novita ignores the thinking switch. **V4.1-Flash on novita works**: 0 parse failures, no reasoning tokens, 3.3 s per sample.

## Stage 2: teacher test (n50 + n50b)

| Metric | n50: V1.8 (API) | **n50: V4.1-Flash** | n50b: V1.8 (API) | **n50b: V4.1-Flash** |
|---|---|---|---|---|
| Emotion accuracy | 0.46 | **0.62** | 0.62 | **0.72** |
| Emotion macro-F1 | 0.204 | **0.395** | 0.374 | **0.420** |
| Profile micro-F1 (avg) | 0.373 | **0.449** | 0.300 | **0.352** |
| Profile exact match (avg) | **0.320** | 0.287 | 0.247 | 0.247 |
| ROUGE-L (char) | 0.178 | **0.231** | 0.200 | **0.249** |
| BLEU-4 (char) | 0.025 | **0.077** | 0.068 | **0.119** |
| Reply median, chars (gold) | – | 126 (146) | – | 120 (100.5) |
| Parse failures | 0 | 0 | 0 | 0 |
| Latency (API, s) | – | 4.0 | – | 3.5 |

V1.8 (API) numbers: EXP-015 (n50) and EXP-016 (n50b).

**PASS.** Flash beats V1.8 on both slices: +16 / +10 points emotion accuracy, ≈ +0.05 ROUGE-L, about 3× / 2× BLEU-4. Its replies run slightly long (median 120 vs gold 100.5 on n50b), so any rewrite prompt must give it a length target.

## Decision

- **Flash = the P2 data tool**, not a submission (the contest is offline). Planned uses: rewrite the ~2.9k gold replies longer than 350 chars that `build_sft.py` now skips, and generate examples for rare emotions (surprise 60, care 77, disgust 117, joy 162 in train) with fixed labels and real train conversations as style examples. Capped at 10–15% of the training set. Estimated cost $1–2 off-peak.
- **Blocked on the organisers:** whether LLM-generated or rewritten training data is allowed (O1). The expert review checks integrity, so we don't use it without a yes.
- DeepSeek's own API (alternative to the HF router): thinking is on by default; switch it off with `extra_body={"thinking": {"type": "disabled"}}`. Peak hours 09:00–12:00 and 14:00–18:00 Beijing time cost 2×.
- Update (Oct 8): the fine-tuned Qwen3-8B (EXP-019) now beats Flash too (0.80 / 0.86 emotion accuracy), so P2 only makes sense for the specific errors that remain (fear → anxiety, relaxed → anxiety).

## Files

- `v4.1-flash_n50/`, `v4.1-flash_n50b/`: `comparison.md`, `config_nothink.json`, `out_nothink/` (`metrics.json`, `performance_report.json`, `submission.jsonl`, `raw_outputs.jsonl`)
- `switch/<model>_<provider>_<switch>/`: the n9 sweep runs, same layout
- `nostream-deepinfra/`: V4-Pro on deepinfra without streaming
- `../exp017-switchtest/`: the first V4-Pro switch test on novita (n9)
