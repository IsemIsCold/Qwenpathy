# Results: Qwen/Qwen3-8B:nscale via https://router.huggingface.co/v1, val_input_n50.jsonl

Decoding: greedy (temperature 0), seed 42. Thinking switch: /think, /no_think tags.

| Metric | nothink |
|---|---|
| Valid submission | True |
| Emotion accuracy | 0.44 |
| Emotion macro-F1 | 0.2006 |
| Profile micro-F1 (avg) | 0.3352 |
| Profile exact-match (avg) | 0.3133 |
| Parse failures | 0 |
| Blank predictions (scored as wrong) | 0 |
| Avg latency per sample (s, API) | 2.95 |
| Avg reasoning length (chars) | 0.0 |
| Wall time (min) | 2.5 |
| Reply length mean (chars) | 97.9 |

Reference: always-anxiety accuracy = 0.32, gold reply length mean = 216.0 chars.

Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. If not, the provider ignored the thinking switch.
