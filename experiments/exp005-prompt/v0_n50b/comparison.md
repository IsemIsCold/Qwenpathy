# Results: Qwen/Qwen3-8B:nscale via https://router.huggingface.co/v1, val_input_n50b.jsonl

Decoding: greedy (temperature 0), seed 42. Thinking switch: /think, /no_think tags.

| Metric | nothink |
|---|---|
| Valid submission | False |
| Emotion accuracy | 0.6 |
| Emotion macro-F1 | 0.3658 |
| Profile micro-F1 (avg) | 0.175 |
| Profile exact-match (avg) | 0.1067 |
| Parse failures | 0 |
| Blank predictions (scored as wrong) | 3 |
| Avg latency per sample (s, API) | 2.71 |
| Avg reasoning length (chars) | 0.0 |
| Wall time (min) | 2.3 |
| Reply length mean (chars) | 111.6 |

Reference: always-anxiety accuracy = 0.34, gold reply length mean = 144 chars.

Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. If not, the provider ignored the thinking switch.
