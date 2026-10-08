# Results: deepseek-ai/DeepSeek-V4-Pro-0813:together via https://router.huggingface.co/v1, val_input_n9.jsonl

Decoding: greedy (temperature 0), seed 42. Thinking switch: api_options {"chat_template_kwargs": {"enable_thinking": false}}.

| Metric | nothink |
|---|---|
| Valid submission | FAILED |
| Emotion accuracy | FAILED |
| Emotion macro-F1 | FAILED |
| Profile micro-F1 (avg) | FAILED |
| Profile exact-match (avg) | FAILED |
| Parse failures | FAILED |
| Blank predictions (scored as wrong) | FAILED |
| Avg latency per sample (s, API) | FAILED |
| Avg reasoning length (chars) | FAILED |
| Wall time (min) | FAILED |
| Reply length mean (chars) | FAILED |

Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. If not, the provider ignored the thinking switch.
