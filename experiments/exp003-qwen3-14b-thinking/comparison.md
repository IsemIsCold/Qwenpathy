# EXP-003 results: Qwen/Qwen3-14B:nscale via https://router.huggingface.co/v1, val_input_n50.jsonl

Decoding: Qwen recommended per mode, seed 42

| Metric | nothink |
|---|---|
| Valid submission | FAILED |
| Emotion accuracy | FAILED |
| Emotion macro-F1 | FAILED |
| Profile micro-F1 (avg) | FAILED |
| Profile exact-match (avg) | FAILED |
| Parse failures | FAILED |
| Avg latency per sample (s, API) | FAILED |
| Avg reasoning length (chars) | FAILED |
| Wall time (min) | FAILED |
| Reply length mean (chars) | FAILED |

Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. If not, the provider ignored enable_thinking.
