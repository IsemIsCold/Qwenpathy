# Results: deepseek-ai/DeepSeek-V4-Pro-0813:deepinfra via https://router.huggingface.co/v1, val_input_n9.jsonl

Decoding: greedy (temperature 0), seed 42. Thinking switch: api_options {"thinking": {"type": "disabled"}}.

| Metric | nothink |
|---|---|
| Valid submission | False |
| Emotion accuracy | 0.2222 |
| Emotion macro-F1 | 0.0227 |
| Profile micro-F1 (avg) | 0.0 |
| Profile exact-match (avg) | 0.4815 |
| Parse failures | 8 |
| Blank predictions (scored as wrong) | 0 |
| Avg latency per sample (s, API) | 4.7 |
| Avg reasoning length (chars) | 0.0 |
| Wall time (min) | 0.7 |
| Reply length mean (chars) | 153.7 |

Reference: always-anxiety accuracy = 0.2222, gold reply length mean = 306.9 chars.

Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. If not, the provider ignored the thinking switch.
