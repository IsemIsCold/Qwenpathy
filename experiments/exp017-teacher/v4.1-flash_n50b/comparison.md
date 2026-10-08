# Results: deepseek-ai/DeepSeek-V4.1-Flash:novita via https://router.huggingface.co/v1, val_input_n50b.jsonl

Decoding: greedy (temperature 0), seed 42. Thinking switch: api_options {"thinking": {"type": "disabled"}}.

| Metric | nothink |
|---|---|
| Valid submission | True |
| Emotion accuracy | 0.72 |
| Emotion macro-F1 | 0.4203 |
| Profile micro-F1 (avg) | 0.3523 |
| Profile exact-match (avg) | 0.2467 |
| Parse failures | 0 |
| Blank predictions (scored as wrong) | 0 |
| Avg latency per sample (s, API) | 3.51 |
| Avg reasoning length (chars) | 0.0 |
| Wall time (min) | 2.9 |
| Reply length mean (chars) | 181.7 |

Reference: always-anxiety accuracy = 0.34, gold reply length mean = 144 chars.

Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. If not, the provider ignored the thinking switch.
