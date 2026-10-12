# EXP-021: n-gram speculative decoding (latency test)

| | |
|---|---|
| **Date** | 2026-10-12 |
| **Author** | Ismail F. |
| **Goal** | Latency is scored **by ranking** among teams (organisers, Oct 12). Find a speed-up that, unlike FP8 (EXP-020), doesn't change the output. |
| **Status** | **Done. Rejected:** 23% *slower*. |

## Setup

- Same merged full model as EXP-019, bf16, greedy, same prompt.
- One vLLM flag: `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'`, in `participant/start_ngram.sh` (= start.sh + the flag; start.sh untouched).
- Idea: vLLM proposes the next tokens by copying matching n-grams from the prompt (JSON keys, emotion and profile labels are all in the system prompt), and the model verifies 4 at once. With greedy decoding the output stays the model's own.
- One 2-GPU boot: `start_ngram.sh` on `t9_input.jsonl` (n50 + n50b), then eval in the same boot.

## Results (T9 = same model, plain bf16)

| | T9 | n-gram |
|---|---|---|
| n50 emotion accuracy | 0.80 | 0.82 |
| n50b emotion accuracy | 0.84 | 0.86 |
| Profile micro-F1 n50 / n50b | 0.490 / 0.495 | 0.500 / 0.450 |
| ROUGE-L n50 / n50b | 0.267 / 0.292 | 0.271 / 0.292 |
| **Latency mean** | **1.55 s** | **1.91 s** |
| Latency median / p95 | 1.35 / 3.59 s | 1.62 / 3.90 s |

Quality is the same within run-to-run noise (as expected for greedy speculative decoding); 0 parse failures.

## Why slower

The vLLM log says `Async scheduling not supported with ngram-based speculative decoding and will be disabled`. Our requests are sequential (batch of 1), where async scheduling hides CPU overhead; losing it, plus the verification overhead when few proposed tokens are accepted (the reply text is new text, not copied from the history), costs more than the copied JSON tokens save.

## Decision

Rejected. Latency options left: plain bf16 (submission #2) or FP8 (EXP-020).
