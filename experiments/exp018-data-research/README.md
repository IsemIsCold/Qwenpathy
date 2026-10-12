# EXP-018: data research for a second training run (Z4)

| | |
|---|---|
| **Date** | 2026-10-12 |
| **Author** | Ismail F. (task Z4, originally Zakaria's) |
| **Goal** | Decide whether a second training run (P2) can gain ≥ 1 point, and with what data change. |
| **Status** | **Done.** No training change reaches 1 point. But it found an **inference-side mismatch**: the profile tag filter deletes tags the model was trained to output (§4), which can be fixed without training. |

All numbers: offline on the laptop/CPU, no GPU, no API. Scores use `scripts/eval.py`'s own `bleu4_corpus()` and `rouge_l_macro()`.

## 1. Reply length (R3): is there a length lever?

**R3a, oracle:** score the gold reply cut to N characters against the full gold (all 2,354 val points).

| N | 80 | 100 | 150 | 200 | 300 | full |
|---|---|---|---|---|---|---|
| BLEU-4 | 0.276 | 0.378 | 0.559 | 0.674 | 0.807 | 1.000 |
| ROUGE-L | 0.785 | 0.836 | 0.905 | 0.940 | 0.972 | 1.000 |

Even a perfect reply loses heavily on BLEU-4 when cut short (corpus BLEU has a brevity penalty): length below gold always costs.

**R3b, our model:** the full fine-tuned model's replies (T7, n50 + n50b) cut at a sentence end to ≤ N characters.

| N | 100 | 150 | 200 | 250 | 300 | 400 | none |
|---|---|---|---|---|---|---|---|
| BLEU-4 | 0.043 | 0.080 | 0.099 | 0.115 | 0.123 | 0.134 | **0.139** |
| ROUGE-L | 0.251 | 0.269 | 0.273 | 0.277 | 0.279 | 0.281 | **0.281** |
| replies cut | 65 | 40 | 25 | 16 | 10 | 5 | 0 |

Every cut lowers BLEU-4 and none raises ROUGE-L. The model's reply median (124 chars) already equals the gold median (125) on these points. **No length lever, neither at inference nor in training.**

## 2. The 350-character training filter (R1)

What each cutoff on the gold reply length removes from the 18,657 training points (val reply median: 98.5):

| Cutoff | Removed | Kept median | Classes hit hardest (share of the class removed) |
|---|---|---|---|
| 150 | 43.1% | 76 | helplessness 55%, fear 55%, shame 54% |
| 200 | 31.4% | 88 | fear 42%, helplessness 40%, anxiety 40% |
| 250 | 24.0% | 97 | fear 33%, anxiety 31%, helplessness 30% |
| 300 | 19.0% | 102 | fear 26%, anxiety 25%, helplessness 24% |
| **350 (used)** | **15.8%** | **106** | fear 22%, anxiety 21%, neutral 20% |
| 400 | 13.5% | 109 | fear 19%, anxiety 18%, helplessness 16% |
| 500 | 10.4% | 113 | fear 14%, anxiety 14%, neutral 13% |

350 keeps 84% of the data with a median close to val. Fear loses the most (22%), but it still has 1,044 training examples, and the fear → anxiety errors sit where the gold labels themselves are split (`docs/p2-imbalance-plan.md` §4). A different cutoff trades length match against a few hundred examples; no evidence it's worth a run.

## 3. Emotion and augmentation

- The remaining emotion errors are mostly arguable or gold-inconsistent (P1 review: 8 model wrong, 6 arguable, 3 gold wrong; `docs/p1-error-review.md`).
- The truly rare classes (pride, joy, disgust, care, surprise) are 4.3% of val points: a perfect fix is worth ≤ 1 point, realistically much less.
- Generated training data: permission asked again on Oct 12, no answer yet.

## 4. Finding: the tag filter fights the training data

`run_inference.py` removes three tags from every prediction (`OVERPREDICTED_TAGS`: `emotionally_stable`, `study_exam`, `colloquial`). The filter was added for the **prompted base model** (EXP-013), which predicted them far too often. But `train/build_sft.py` **disables the filter for training targets** (line 76–78), so the fine-tuned model was trained to output them:

| Tag | In training targets (14,942) | In val gold (2,354) |
|---|---|---|
| `study_exam` (interests) | 586 (3.9%) | 95 (4.0%) = **15% of all val interest labels** |
| `colloquial` (style) | 603 (4.0%) | 54 (2.3%) |
| `emotionally_stable` (personality) | 279 (1.9%) | 9 (0.4%) |

Every correct prediction of these tags is deleted after the model makes it. Training and inference are inconsistent here, and `study_exam` matters for the interests field.

**Fix:** set `OVERPREDICTED_TAGS` to empty for the fine-tuned model (parser-only, no training). **To measure:** one inference run with the filter off on the 100-point slice (~0.25 card-hours); adopt it if profile micro-F1 rises on both slices without hurting exact match.

## Decision

- **No second training run (P2):** no data change found that is likely to gain ≥ 1 point.
- **Test the filter-off parser** on Migu (Oct 13), together with the FP8 decision.
