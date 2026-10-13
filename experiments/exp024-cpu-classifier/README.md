# EXP-024: CPU classifiers next to the LLM (offline, no GPU)

| | |
|---|---|
| **Date** | 2026-10-13 |
| **Author** | Ismail F. |
| **Goal** | Find points without training or GPU time: can a small CPU classifier, trained on `train_public` only, beat or improve the fine-tuned LLM (submission #2) on profile tags or emotion? |
| **Status** | Offline results in. Profile: **blend LLM + classifier +0.035 micro-F1** (95% CI +0.014 to +0.055) on 600 val points. Emotion: no gain. Not in any submission yet. |

## Setup

- **Data:** 18,657 points from `train_public.jsonl` (8,333 conversations). Val = the 600 scored points (`tune500` + `n100`); val conversations are separate from train, so nothing here was fitted on val except where marked "CV".
- **Profile classifier:** char 1–3-gram TF-IDF (300k features, sublinear tf) over all user messages up to the point (last 3,000 chars) + number of user turns; one logistic regression per tag (33 tags, C = 4). Per-field thresholds tuned on a 10% held-out slice of **train** (by conversation): personality 0.225, interests 0.10, style 0.175.
- **Emotion classifier:** char 1–3-gram TF-IDF of the last user message + 1–2-gram of the one before; multinomial LR (16 labels).
- **LLM reference:** submission #2's bf16 answers from EXP-022 (`outputs/exp022/bf16/*_filter_on_b0.jsonl`, tag filter on) and its label logprobs (`lp_bf16.jsonl`).
- **Blend (profile):** keep a tag if `a·[LLM output has it] + (1−a)·p_classifier ≥ t`, with (a, t) per field. Picked by **5-fold CV over val conversations** (each fold's score uses (a, t) fitted on the other 4 folds).
- **Uncertainty:** 1,000 bootstrap resamples of val conversations, paired difference vs #2.
- All F1 numbers cross-checked with `scripts/eval.py`'s `score_profile`.

## Results: profile (micro-F1 averaged over the 3 fields)

| Variant | tune500 | n100 | All 600 | Δ vs #2 [95% CI] | Exact match (600) |
|---|---|---|---|---|---|
| LLM #2 | 0.4561 | 0.4806 | 0.4609 | — | 0.534 |
| Classifier alone | 0.4692 | 0.4642 | 0.4682 | +0.007 [−0.033, +0.045] | **0.568** |
| Style from classifier, rest LLM | 0.4768 | 0.5034 | 0.4824 | +0.022 [+0.009, +0.035] | 0.551 |
| Style + personality from classifier | 0.4923 | 0.4715 | 0.4876 | +0.027 [+0.004, +0.047] | **0.597** |
| **Blend LLM + classifier (CV)** | **0.4883** | **0.5234** | **0.4960** | **+0.035 [+0.014, +0.055]** | 0.550 |

Per field (600): LLM 0.422 / 0.574 / 0.386 (personality / interests / style); blend 0.442 / 0.602 / 0.444. The classifier is much better than the LLM on **style**, worse on **interests**.

Blend parameters are stable across folds: personality (a 0.2, t 0.25) in 3 of 5 folds (a = 0.2 in 4 of 5), interests (0.5, 0.15) in 4 of 5, style (0.0, 0.15) in 4 of 5. In words: keep the LLM's personality and interest tags unless the classifier strongly disagrees, add classifier tags it is confident about, and take style from the classifier alone.

Contest points (15 for profile): blend ≈ **+0.5** if profile is scored as micro-F1, ≈ +0.25 if exact match. Which one the organisers use is still unanswered (O1).

## Results: emotion

| | tune500 | n100 | All 600 |
|---|---|---|---|
| LLM #2 | 0.782 | 0.830 | 0.790 |
| Classifier alone | 0.628 | 0.720 | 0.643 |
| LLM logprobs + w·log p_classifier (CV, top 5 or 20) | 0.770 | 0.810 | 0.777 (−0.013, CI −0.023 to −0.005) |

The classifier is far weaker than the LLM and the blend loses accuracy. **Emotion stays with the LLM.**

## Early stop (estimate)

If profile came from the classifier alone, generation could stop after `emotion_label`. On the 600 raw answers the profile + memory part is ≈ 32 of ≈ 110 output tokens (≈ 24%, estimated from characters, not tokenised), so latency could drop ≈ 20% (1.55 → ≈ 1.25 s) with emotion and replies unchanged. The blend needs the LLM's tags, so it can't use early stop.

## Also tried (rejected)

Tuning a list of profile tags to drop on tune500: +1.2 F1 there, −1.5 on n100. Overfits; not used.

## Files

`data.py`, `profile_clf.py` (fit + thresholds, ≈ 3 min on 2 CPUs), `profile_hybrid.py` (variants, CV, bootstrap), `emotion_clf.py`. Run from this folder with the data files in `data/` and EXP-022 outputs in `outputs/exp022/`.
