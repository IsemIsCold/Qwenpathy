# Latency (10 contest points, ranked against other teams)

What the organisers told us: latency is **ranked relative to the other teams** (not a fixed threshold) and **vLLM start-up is excluded**. Our number is the one `participant/run_inference.py` writes to `performance_report.json`: mean over the first 100 samples, timing scope "HTTP request + server inference + response decode + parse + validate; server start-up excluded".

Setup for every row: Migu instance, 2× RTX 4090, vLLM 0.19.0, tensor parallel 2, `--max-model-len 32768`, greedy decoding, `max_new_tokens` 1200, the same `start.sh` the organisers run.

| Run | Date | What changed | Input (rounds) | Mean | Median | p95 | Max | Source |
|---|---|---|---|---|---|---|---|---|
| **Submission #1** | Oct 4 | Base Qwen3-8B, official prompt V0, bf16 | Migu val50 (50) | 1.41 s | 1.40 s | 1.79 s | 2.01 s | `outputs/migu_val50/val50/performance_report.json`; LOG.md #1 |
| **Submission #2** | Oct 9 | Full QLoRA model (EXP-019), V1.8 prompt, bf16 | t9 = n100 (100) | 1.55 s | 1.35 s | 3.59 s | 4.85 s | T9 check run; LOG.md #2 |
| EXP-020 (test) | Oct 12 | #2 + vLLM `--quantization fp8` | t9 (100) | 1.16 s | 1.00 s | 2.32 s | – | LOG.md EXP-020 |
| EXP-021 (rejected) | Oct 12 | #2 + n-gram speculative decoding | t9 (100) | 1.91 s | 1.62 s | 3.90 s | – | LOG.md EXP-021 |
| **Submission #3** | Oct 16 | FP8 + CPU profile classifier blend (EXP-025) | t9 (100) | **1.14 s** | 0.97 s | 2.40 s | – | EXP-025 check run, Oct 15 |
| Submission #4 (candidate) | – | #3 + stop generating after `emotion_label` (EXP-027) | t9 (100) | pending | | | | EXP-027 check run; its pass rule: mean ≤ 0.90 × #3 in the same boot |

## What moved the number

- **#1 → #2, +10%:** the fine-tuned model writes longer replies (median 150 / 106 characters on n50 / n50b for T7, vs 85 / 96 for the base model with the V1.8 prompt in T4), and generation time grows with output length.
- **FP8, −25%:** the weights are read in 8 bits instead of 16, and decoding one token at a time is limited by memory bandwidth, so it gets faster. It costs a little accuracy on 100 points (EXP-020); #3 accepted that because the profile blend gained more than FP8 lost (EXP-025).
- **n-gram speculation, +23%:** rejected: vLLM turns off async scheduling when speculative decoding is on (EXP-021).
- **#4's idea:** about 41% of each answer's characters come after the emotion label (the profile and `memory_refs`). The profile now comes from the CPU classifier, so the model can stop right after the label and skip that part.
- **The CPU profile classifier** adds about 1 ms per sample.

First-100 numbers vary from boot to boot, so small differences are only trusted when both runs come from the same boot (that's why the EXP-027 rule compares against #3 measured in the same boot).
