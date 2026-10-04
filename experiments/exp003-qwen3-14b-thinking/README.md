# EXP-003: Qwen3-14B, thinking vs non-thinking (partial: 9 val points, non-thinking only)

| | |
|---|---|
| **Date** | 2026-10-04 |
| **Author** | Ismail F. |
| **Goal** | Compare Qwen3-14B thinking and non-thinking mode with the official prompt; decide whether a bigger model is worth it. |
| **Status** | ⚠️ Partial. Non-thinking ran on 9 of 50 val points before the free API credit ran out; thinking not run. **Working decision: Qwen3-14B non-thinking** (see [docs/technical-plan.md](../../docs/technical-plan.md)). Full 50-point run planned on Migu. |

## Setup
| Item | Value |
|---|---|
| Script | [`scripts/exp3_compare.py`](../../scripts/exp3_compare.py) (official prompt, history format and parser; validate + eval per mode) |
| Model / provider | `Qwen/Qwen3-14B:nscale` via Hugging Face router (`https://router.huggingface.co/v1`), streaming |
| Thinking switch | `--soft-switch`: Qwen3 `/no_think` tag appended to the user message (provider ignores `enable_thinking`) |
| Decoding | Qwen recommended non-thinking sampling: temperature 0.7, top_p 0.8, top_k 20, seed 42, max 1200 tokens |
| Data | `data/val_input_n50.jsonl` / `val_gold_n50.jsonl` (Zakaria's B2); scored on the first 9 points |
| Cost | Hugging Face free monthly credit ($0.10), used up after 9 samples. 0 Migu card-hours. |

```powershell
python scripts/exp3_compare.py --participant <package>/participant `
  --input data/val_input_n50.jsonl --gold data/val_gold_n50.jsonl `
  --scripts scripts --out experiments/exp003-qwen3-14b-thinking `
  --soft-switch --key-env HF_TOKEN --modes nothink `
  --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-14B:nscale"
```

## Attempts that failed (for reference)
| Route | Result |
|---|---|
| SiliconFlow `.cn` | 401: account is on the international site |
| SiliconFlow `.com` | 402: Qwen3-14B is paid, balance 0 (Qwen2.5-7B is free) |
| siliconflow.cn / Alibaba Bailian / ModelScope | need Chinese real-name verification: not possible for a foreign student |
| Hugging Face router | worked, free credit exhausted after 9 samples |

## Results (9 val points, non-thinking): [`out_nothink/metrics_n9.json`](out_nothink/metrics_n9.json)
| Metric | Qwen3-14B non-thinking | Always-anxiety baseline |
|---|---|---|
| Emotion accuracy | **0.444** (4/9) | 0.222 |
| Emotion macro-F1 | 0.123 | n/a |
| Profile micro-F1 (avg) | 0.254 | 0 |
| Profile exact-match (avg) | **0.074** | n/a |
| Reply length mean / median (chars) | 128 / 109 | gold: 307 / 146 |
| Chinese-char ratio | 0.885 | gold mean 0.855 |
| Parse failures | 0 | n/a |
| Latency (API, provider-dependent) | 2.2–4.0 s/sample (≈2.8 s) | n/a |

## Observations
1. **Emotion:** right on anxiety, loneliness and relaxed. It drifts to **sadness** when the user sounds down (helplessness→sadness, anger→sadness, relaxed→sadness), and confused gratitude with anxiety.
2. **Profile (main problem):** fills **every** field every time with nearly the same template (sensitive + introverted / social / high_emotional_expression + colloquial). Gold style is mostly `rational`, and 2 of 9 gold profiles are empty. This is the opposite of Qwen2.5-7B (EXP-002), which left `style` empty.
3. **Replies:** generic encouragement (你不是一个人在战斗, *nǐ bú shì yí gè rén zài zhàndòu*, "you're not fighting alone"; 加油，我相信你可以的, *jiāyóu, wǒ xiāngxìn nǐ kěyǐ de*, "come on, I believe in you"), even when the user explicitly asked only to be listened to. It's much shorter and shallower than the long, structured gold replies, which hurts the BLEU/ROUGE/BERTScore part of the score.
4. **Latency:** API latency reflects the provider's hardware and load, not the model: 14B via Nscale (≈2.8 s) looked faster than 7B via SiliconFlow (≈4.9 s, EXP-002). On the same GPU, 14B is roughly 2× slower than 7B. Only latency measured on the Migu 2×4090 counts.

## Conclusions
- 9 points is too few to decide on its own (one sample = 11 accuracy points), but 14B clearly beats the majority baseline on emotion, and the problems (profile over-filling, generic replies) are prompt/fine-tuning issues, not model limits.
- Thinking mode is unlikely to be worth it: latency is 10 official points.
- **Decision (team, Oct 4): Qwen3-14B in non-thinking mode as the working base model**, pending (a) the organisers' answer on whether Qwen3 is allowed and (b) a full 50-point run on Migu.

## Next steps
- [ ] Full EXP-003 (50 points, both modes) on the Migu instance with vLLM once B7 is up (≈0.3 card-hours); measure real latency.
- [ ] Prompt fix for profile over-filling: "leave a field empty unless the conversation gives clear evidence" (B5).
- [ ] Same 50 points with Qwen2.5-7B (free on SiliconFlow) as the reference row.

## Files
- `out_nothink/submission.jsonl`: the 9 predictions
- `out_nothink/metrics_n9.json`: eval.py on those 9 points (`data/val_gold_n9.jsonl` = first 9 gold rows)
- `config_nothink.json`: config used
- `comparison.md`: auto-generated table from the script (shows FAILED because the run stopped at the credit limit; use this README)
