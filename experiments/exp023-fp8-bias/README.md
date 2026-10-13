# EXP-023: FP8 + anxiety bias as submission #3 (decided and checked in one boot)

| | |
|---|---|
| **Date** | 2026-10-13 |
| **Author** | Ismail F. |
| **Goal** | Decide whether FP8 weights (≈ 25% faster, EXP-020) plus the anxiety-prior correction (EXP-022) should replace submission #2, on 500 points instead of 100, with a rule fixed before the run. |
| **Status** | **Done. NO-GO** (profile micro-F1 below the bar). Submission #2 stays; `/root/participant` untouched. |

## Why

EXP-022 left three options:

| Option | Latency | Emotion n100 | Profile micro-F1 n100 |
|---|---|---|---|
| A: keep #2 (bf16) | 1.55 s (T9) | 0.83 | 0.481 |
| B: FP8, no bias | ≈ 1.16 s (EXP-020) | 0.79 | 0.467 |
| C: FP8 + b 0.5 | ≈ 1.25 s (1.16 + logprob overhead) | 0.82 | 0.467 |

C beats B (+3 labels for ≈ 0.09 s). C vs A costs ≈ 0.5 measurable points for a −19% latency that is scored by ranking among teams (unknown value). 100 points is too thin for that call, and FP8 varies from run to run (EXP-020 measured profile −0.05 on the same 100 points; EXP-022 −0.014).

Reply quality was checked first, offline, on the EXP-022 n100 outputs: ROUGE-L 0.2743 vs 0.2827, BLEU-4 0.1319 vs 0.1403 (both −0.008, inside the −0.01 line) → go ahead.

## Gate (fixed Oct 13, before any FP8 tune500 number was seen)

On **tune500**, FP8 with **its own** b (tuned on tune500, top-5 alternatives, tag filter on) must reach all three, otherwise submission #2 stays:

1. emotion accuracy ≥ **0.782** (bf16 #2, no bias)
2. profile micro-F1 ≥ **0.4461** (bf16 0.4561 minus 1 point)
3. ROUGE-L ≥ bf16 ROUGE-L on the same 500 points − 0.01 (bf16 0.2963 → bar **0.2863**)

GO would also need a clean check run: `validate.py` 0 errors, 0 parse failures, `complete: true`, mean latency < 1.40 s.

## Method

`scripts/exp023_session.sh`, one 2-GPU boot:

1. FP8 server (= `start_fp8.sh` args: TP2, 32k, `--quantization fp8`) → `emotion_logprobs.py` on t10 (600 points = n100 + tune500), 8 requests at once.
2. `tune_emotion_bias.py --top-k 5` picks FP8's b on tune500 and reports n50 / n50b / n100; `scripts/exp023_gate.py` applies the gate → GO / NO-GO (exit 0 / 3).
3. NO-GO: stop. GO: `scripts/exp023_install.py` backs up #2's `run_inference.py`, `start.sh`, `inference_config.json` to `/root/sub2_backup` (never overwritten; `--restore` puts them back, sha-checked), patches `start.sh`, installs the new `run_inference.py`, sets the config.
4. Check run = plain `bash /root/participant/start.sh t9_input.jsonl …` (exactly what would be submitted) + `validate.py` + `eval.py` on n100 + latency from `performance_report.json`.

Upload = one archive `exp023_upload.tgz` (repo layout: the 7 scripts, the new `run_inference.py`, `val_gold_tune500.jsonl`, the bf16 tune500 reference) unpacked to `/mnt/storage/exp023_upload/`.

### Code (in main, inert unless the config asks for it)

- `participant/run_inference.py`: config keys `emotion_bias` (b, default 0 = off) and `emotion_bias_top_logprobs` (default 5). With b > 0 the request adds `logprobs` + `top_logprobs`, and after parsing, the label is switched with exactly the rule and functions of `tune_emotion_bias.py`. Prints `emotion bias … switched n/N` at the end; `performance_report.json` keys unchanged. Without the keys the output is byte-identical to submission #2's `run_inference.py`.
- `participant/start.sh`: reads `vllm_quantization` from the config and adds `--quantization <q>` (both the 1- and 2-GPU branch). Absent or empty = bf16 as before.
- `scripts/tune_emotion_bias.py --top-k K`: score with only the K best alternatives (= what `top_logprobs: K` returns).

### Tests (`tests/test_emotion_bias.py` + `tests/stub_vllm.py`, Linux, no GPU): 42/42

- Scoring in `run_inference.py` = the tuner's on all 700 EXP-022 points, top-5 and top-20, 7 values of b.
- Stub server (answers from gold, says anxiety on ~40% of other points with the gold label 0.1–1.5 logprob behind): no key → no logprobs requested and the submission is byte-identical to the old `run_inference.py`; b 0.5 / 1.0 → every label equals the expected switch; bad config values rejected.
- `start.sh` patch with a fake `vllm`: `--quantization fp8` passed with the key, not without; patch = original + one block; install twice = one patch, backup kept; restore = #2's exact files.
- Whole session on the fake vLLM: GO path (tuner recovers the planted gaps, check run validates) and NO-GO path (participant untouched, no backup made).

## Run (Oct 13)

- First boot: the archive was never uploaded (`find / -maxdepth 3 -name 'exp023_upload*.tgz'` found nothing); uploaded, then launched. The instance was shut down before `/exp023.tgz` was downloaded → a short no-GPU boot to read the log and download (results live on `/mnt/storage/out/exp023/`).
- FP8 server ready after 79 s. 600 points: label logprobs on 587 (top 5), top label = parsed label 586/587.
- Submission #2's files on Migu: `run_inference.py` sha256 `7408b4ab…`, `start.sh` `a7698a02…`. **Note:** the repo's `start.sh` before this experiment was `e87f86ce…`, so Migu's `start.sh` is not byte-identical to the repo's (difference not checked yet).

## Results

### b on tune500 (FP8, top 5)

| b | 0 | 0.25 | 0.5 | **0.65** | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| tune500 accuracy | 0.778 | 0.770 | 0.776 | **0.788** | 0.786 | 0.782 | 0.770 | 0.754 |

Anxiety predicted on tune500: 38.8% → 33.8% at b = 0.65 (gold 32.4%); 25 labels switched.

### Gate

| tune500 | FP8, b 0.65 | Bar | bf16 #2 | |
|---|---|---|---|---|
| Emotion accuracy | 0.7880 | 0.7820 | 0.782 | ok (in-sample) |
| Profile micro-F1 | **0.4350** | 0.4461 | 0.4561 | **FAIL** |
| ROUGE-L | 0.2966 | 0.2863 | 0.2963 | ok |
| BLEU-4 (info) | 0.1468 | | 0.1471 | |

**NO-GO.** No install, no check run.

### Where FP8 loses profile quality (tune500, filter on, b 0)

| Field | bf16 micro-F1 (tags predicted) | FP8 micro-F1 (tags predicted) | Gold tags |
|---|---|---|---|
| personality_traits | 0.417 (427) | 0.416 (372) | 249 |
| interests | **0.576** (112) | **0.537** (103) | 124 |
| style | **0.375** (342) | **0.352** (331) | 277 |
| Average | 0.456 | 0.435 | |

Exact match is about the same (0.552 vs 0.544): FP8 predicts fewer tags, and loses mostly on interests and style. Emotion macro-F1 also drops (0.692 → 0.627 at b 0), though the official metric is accuracy.

### Other slices (info only; b not picked on them)

| Slice | FP8 emotion b 0 → 0.65 (switched) | FP8 profile F1 | bf16 emotion / profile (EXP-022) |
|---|---|---|---|
| n50 | 0.82 → 0.82 (2) | 0.475 | 0.80 / 0.487 |
| n50b | 0.84 → 0.84 (0) | 0.449 | 0.86 / 0.477 |
| n100 | 0.83 → 0.83 (2) | 0.466 | 0.83 / 0.481 |

FP8 n100 accuracy without bias was 0.79 in EXP-022 and 0.83 here, with identical settings (batched greedy decoding differs in a few tokens between runs). 100 points cannot separate FP8 from bf16 on emotion; tune500 is the decision slice.

## Reading

- FP8's cost is in the **profile tags**, every time it was measured: −0.05 (EXP-020, n100), −0.014 (EXP-022, n100), −0.021 (here, tune500). The bias only recovers emotion accuracy, not tags.
- Expected points vs #2: profile −0.021 × 15 ≈ −0.3, emotion +0.006 × 25 ≈ +0.15 (in-sample, optimistic), replies ≈ 0 → ≈ −0.15 before latency. The latency gain (1.55 → ≈ 1.25 s) is ranked against unknown teams. The gate exists so this close call isn't decided after seeing the numbers.

## Decision

**Submission #2 stays.** No further GPU work planned; 3 submissions unused. The inference-time levers for this model are now all tested: prompt (EXP-005–016), tag filter (EXP-013/022), FP8 (EXP-020/023), n-gram speculation (EXP-021), anxiety bias (EXP-022/023).

## Files

- Laptop: `outputs/exp023/` (`lp_fp8_t10.jsonl`, `fp8/` tuner outputs + `summary.json`, `gate.json`, `vllm_fp8.log`), `outputs/exp023_session.log`. Archives: `migu-backups/exp023_upload.tgz`, `migu-backups/exp023.tgz`.
- Migu: `/mnt/storage/out/exp023/`, `/mnt/storage/exp023_upload/`.
- Compute: one 2-GPU boot (≈ 0.5 card-hours, including the upload retry) + one no-GPU boot (≈ 0.1). Read the exact numbers from the Migu page.
