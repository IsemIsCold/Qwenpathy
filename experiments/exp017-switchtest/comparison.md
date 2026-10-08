# Results: deepseek-ai/DeepSeek-V4-Pro-0813:novita via https://router.huggingface.co/v1, val_input_n9.jsonl

Decoding: greedy (temperature 0), seed 42. Thinking switch: api_options {"thinking": {"type": "disabled"}}.

| Metric | nothink |
|---|---|
| Valid submission | True |
| Emotion accuracy | 0.1111 |
| Emotion macro-F1 | 0.0139 |
| Profile micro-F1 (avg) | 0.0741 |
| Profile exact-match (avg) | 0.4074 |
| Parse failures | 7 |
| Blank predictions (scored as wrong) | 0 |
| Avg latency per sample (s, API) | 6.36 |
| Avg reasoning length (chars) | 269.1 |
| Wall time (min) | 1.0 |
| Reply length mean (chars) | 132.6 |

Reference: always-anxiety accuracy = 0.2222, gold reply length mean = 306.9 chars.

Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. If not, the provider ignored the thinking switch.
