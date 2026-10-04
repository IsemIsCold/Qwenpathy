# EXP-003 results: Qwen/Qwen3-14B:nscale via https://router.huggingface.co/v1, val_input_n50.jsonl

Decoding: Qwen recommended sampling per mode, seed 42. Thinking switch: /think, /no_think tags.

| Metric | think |
|---|---|
| Valid submission | True |
| Emotion accuracy | 0.42 |
| Emotion macro-F1 | 0.1481 |
| Profile micro-F1 (avg) | 0.3665 |
| Profile exact-match (avg) | 0.18 |
| Parse failures | 0 |
| Blank predictions (scored as wrong) | 0 |
| Avg latency per sample (s, API) | 5.9 |
| Avg reasoning length (chars) | 835.1 |
| Wall time (min) | 4.9 |
| Reply length mean (chars) | 107.1 |

Reference: always-anxiety accuracy = 0.32, gold reply length mean = 216.0 chars.

Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. If not, the provider ignored the thinking switch.
