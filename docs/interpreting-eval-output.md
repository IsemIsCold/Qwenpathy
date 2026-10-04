# Interpreting `eval.py` Output — Field Guide for the Team

> Owner: Yahya (Evaluation). Companion to `docs/eval-design.md` (architecture & contracts).
> Scope: how to verify the evaluator is correct, and how to read every number it prints.
> All anchor values below are computed from the real `data/val_gold.jsonl` (2,354 samples).

---

## Part 1 — Is the evaluator itself correct?

Run these 5 checks before trusting ANY number. A broken ruler makes every
model look wrong (or right).

| # | Check | How | Expected |
| --- | --- | --- | --- |
| 1 | Test suite green | `python tests/run_tests.py` | 57/57 PASS |
| 2 | Perfect ceiling | score `val_gold.jsonl` as its own submission | exactly **1.0000** everywhere (accuracy, macro-F1, profile micro-F1, exact-match) |
| 3 | Useless floor | always-anxiety + empty-profile submission | accuracy **0.3190**, delta **+0.0000**, macro-F1 ≈ **0.03** |
| 4 | Order invariance | shuffle the lines of any submission, rescore | byte-identical `metrics.json` |
| 5 | Reproducibility | run the same file twice | byte-identical output |

Any deviation in checks 2–5 means the **evaluator** is broken — stop and fix
the evaluator before debugging any model.

---

## Part 2 — Reading the numbers

### EMOTION (official metric, 25 pts)

| Number | Meaning | Healthy | Red flag |
| --- | --- | --- | --- |
| `accuracy` | THE competition score (single-label, 16 classes) | **> 0.3190** (above the useless baseline) | ≈ 0.3190 → you shipped majority-class collapse |
| `macro_f1` | Unweighted mean of per-class F1 — the anti-collapse detector | gap vs accuracy **< 0.15** | accuracy 0.55 with macro-F1 0.20 → model only "sees" anxiety/relaxed |
| `delta vs always-anxiety` | One-line summary of real signal | **> 0** | ≤ 0 → the model contributes nothing |
| per-class F1 on rare classes (`care`, `surprise`, `joy`, `pride`, `disgust`; support < 30) | Where macro-F1 is won or lost | some non-zero values | all ≈ 0 → normal for naive models; class-balancing work (LDAM, class-balanced loss) must move these |
| top confusion pairs | Which classes bleed into each other | — | e.g. `anxiety -> fear x200` means prompt/model confuses that pair |

### USER PROFILE (15 pts)

| Number | Meaning | Healthy | Red flag |
| --- | --- | --- | --- |
| `micro_f1_avg` | Multi-label quality averaged over the 3 fields | clearly > 0 (empty baseline = 0.0000) | 0.0000 → model predicts nothing |
| `exact_match_avg` | Strict set-equality rate | above the empty rates below | — |
| `empty_gold_rate` (per field) | How often gold itself is empty | traits **0.619**, interests **0.749**, style **0.517** | a model that always predicts non-empty is wrong ~50–75% of the time before picking a single label |

### RESPONSE (diagnostic for the 20 quality pts)

| Number | Gold anchor | Red flag |
| --- | --- | --- |
| pred length mean/median | gold: **mean 158 / median 98 chars** | pred mean 2–3× gold → verbosity (also hurts the 10 latency pts); pred ≈ 20 chars → lazy replies |
| `chinese_char_ratio_pred` | gold: **mean 0.855, 5th percentile 0.745** | **< 0.70** → model is leaking English / non-Chinese output |

### INTEGRITY (must-always-be)

| Number | Must be |
| --- | --- |
| `n_scored` | 2354 |
| `n_invalid_predictions` | **0** — anything else means an invalid label reached scoring; `validate.py` should have caught it |
| `empty_profile_exact_match_avg` (baseline row) | ≈ 0.628 (1 − all-fields-empty rate 0.372) |

---

## The 30-second scan (run this on EVERY eval)

1. **delta vs always-anxiety > 0?** — no → stop; something is fundamentally broken.
2. **accuracy − macro-F1 gap < 0.15?** — no → majority-class collapse.
3. **n_invalid_predictions = 0 and n_scored = 2354?** — no → pipeline bug; distrust everything.
4. **chinese ratio ≥ 0.70 and pred length within ~2× of gold?** — no → output drift.
5. **Rare-class F1s moved?** — training/prompt wins show up here first, before overall accuracy.

---

## Worked example — the useless baseline (what "sick" looks like)

Submission: always predicts `anxiety`, empty profile, short generic Chinese reply.

```javascript
accuracy (OFFICIAL)   0.3190    <- looks "not terrible"
macro-F1              0.0302    <- reveals uselessness on 15/16 classes
delta vs always-anxiety   +0.0000   <- a useless model, proven useless
profile micro-F1      0.0000
```

Lesson: **accuracy alone lies on imbalanced data.** Always read it together
with macro-F1 and the delta line.

---

## Metric tiers (decision discipline)

Per `docs/eval-design.md`, only three numbers may decide KEEP/REJECT:

| Tier | Metrics |
| --- | --- |
| **DECISION** | emotion accuracy (official), macro-F1, profile micro-F1 |
| **DIAGNOSTIC** | per-class P/R/F1, confusion pairs, lengths, Chinese ratio, BLEU/ROUGE/BERTScore (when added) |
| **QUALITATIVE** | judge scores (dev-only, never official) |

If a debate cites a diagnostic number as the deciding argument, the debate is
invalid by definition.