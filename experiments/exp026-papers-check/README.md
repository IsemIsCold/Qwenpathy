# EXP-026: ideas from the "Research proposal" papers + Zakaria's P2 package, checked offline

| | |
|---|---|
| **Date** | 2026-10-16 |
| **Author** | Ismail F. |
| **Goal** | Find points for a possible submission #4 before the Oct 18 freeze, without GPU time: test every idea from the 12 papers (from a master's student) and Zakaria's `p2_anxiety_fix_package` that can be tested on data we already hold. |
| **Status** | Done (CPU only, ≈ 30 s). **Nothing worth a submission #4.** One idea gains on profile, but only by reading the user's *future* messages (an integrity problem), and the gain disappears if the test set lacks full conversations. |

Run: `python experiments/exp026-papers-check/exp026.py` (needs `data/train_public.jsonl`, `data/val_public.jsonl`, `outputs/exp022/lp_bf16.jsonl`, `outputs/exp023/lp_fp8_t10.jsonl`). Writes `summary.json`. Every val-tuned parameter is picked by 5-fold CV over val conversations (EXP-024 folds); 95% CIs from 1,000 bootstrap resamples of conversations. "#3 FP8" = the raw answers of #3's model (FP8) from EXP-023.

## The papers

They are a reading list for a research proposal on **LLM-agent simulation of how emotions spread through social networks**, not on dialogue models. What transfers:

| Paper | Idea | Use for us |
|---|---|---|
| Schöller et al. 2018 (state-trait dynamics in psychotherapy); Deboeck & Bergeman 2013 (reservoir model); Sprott 2005 (dynamical models of happiness) | Fast **states** move around slow **traits**; a state has inertia. | Tested: profile = trait (§1, §3), emotion = state with inertia (§2). |
| McCrae & Costa 1991; Anglim & O'Connor 2019 (Big Five, HEXACO) | Traits are bipolar dimensions. | Tested: opposite tags (§6). |
| Sacchi & Dan-Glauser 2025 (Big Five → cognitive appraisals) | Emotion comes from appraisals (relevance, implications, coping potential, norms). In appraisal theory, fear = a certain, near threat; anxiety = an uncertain, future threat; relaxed = the threat is resolved. | Exactly our fear/relaxed → anxiety confusion. Needs prompting or training data with appraisals → **final (December)**, not testable offline. |
| Park et al. 2023 (Generative Agents) | Memory stream; retrieval = recency (decay 0.995) + importance + relevance, min-max scaled; reflections when summed importance > 150. | "Memory fusion" for long conversations. Our histories fit the 32k context easily (the longest val histories checked were ≈ 7.5k tokens), so retrieval adds nothing now → **final** (expert review, long sessions). |
| Li et al. (MatrAIx, 8.3B persona agents) | Simulated users with persona records test chatbots at scale. | Evaluation harness for empathy (expert-review proxy) → **final**, dev-time only (API). |
| Li et al. 2025 (HySID: LLM + diffusion) | Run the LLM where semantics matter, propagate with a cheap model. | Already what EXP-024/025 do (LLM tags + CPU classifier). Good citation for the defence. |
| Fowler & Christakis 2008, Bail et al. 2018, Barabási & Albert 1999 | Happiness spread, polarisation, scale-free networks. | Nothing for this task. |

## Results

### 1. Profile is a trait

| | same as previous point | grows | shrinks | other |
|---|---|---|---|---|
| train (10,324 pairs) | 72.5% | 11.5% | 5.1% | 10.9% |
| val (1,454 pairs) | 71.9% | 14.0% | 6.9% | 7.2% |

The gold profile of a conversation's **last** point, used as the prediction for every point of that conversation, scores micro-F1 **0.907** / exact **0.922** on the 600 points. The profile is (almost) one label per conversation.

### 2. Emotion is a state, and inertia adds nothing

Label = previous point's label in 26.7% of val pairs (16.6% by chance). Adding `w · PMI(label | previous label)` (train transition counts) to the LLM's label logprobs, **even with the gold previous label** (an upper bound): bf16 0.7900 → 0.7867, FP8 0.7867 → 0.7817 (CV). The LLM already reads the whole history. **Dead.**

### 3 + 4. Profile: pooling across a conversation's points, and the tag filter (Zakaria A1)

Keep a tag if `a·[LLM has it] + (1−a)·(λ·p_own + (1−λ)·p_pool) ≥ t`; (a, t, λ) per field by CV.

| Micro-F1, #3 FP8 answers | tune500 | n100 | all 600 | exact | Δ vs #3 blend [95% CI] |
|---|---|---|---|---|---|
| LLM alone | 0.4350 | 0.4656 | 0.4411 | 0.5406 | −0.040 [−0.067, −0.014] |
| **Blend, own point (= #3)** | 0.4740 | 0.5079 | 0.4815 | 0.5494 | — |
| Blend, filter off (Zakaria A1) | 0.4764 | 0.5079 | 0.4833 | 0.5511 | +0.002 [−0.002, +0.006] |
| + pool, past points only | 0.4721 | 0.5026 | 0.4788 | 0.5483 | −0.003 [−0.011, +0.005] |
| + pool, all points | 0.4765 | 0.4994 | 0.4813 | 0.5695 | −0.000 [−0.014, +0.012] |
| + pool, last point **(future messages)** | 0.4880 | 0.5247 | 0.4960 | 0.5555 | **+0.015 [+0.005, +0.024]** |
| same, only the last point present in the 600 | 0.4785 | 0.5224 | 0.4882 | 0.5444 | +0.007 [−0.001, +0.015] |

bf16 (#2) answers: same pattern (filter off +0.001, past −0.002, last +0.011 [−0.001, +0.023]). The filter off changes only 19–24 tags of 600 points.

- **Filter off (A1): noise** (+0.002 F1 ≈ +0.03 contest points), as in EXP-022. Not "+1–3 points".
- **Pooling only helps with the last point**, whose history contains the user's messages *after* the point being predicted. A deployed companion never has those; the task gives "history up to the latest user message". Using later test lines to answer earlier ones is a leak an expert reviewer could mark down (integrity is part of the 30-point review). The gain is ≤ +0.2 contest points and halves if the test file lacks complete conversations. **Not used.**

### 5. Anxiety bias on FP8, honest estimate

| | no bias | bias, 5-fold CV | Δ [95% CI] | b picks |
|---|---|---|---|---|
| #2 bf16 | 0.7900 | 0.7983 | +0.008 [−0.007, +0.023] | 0.5 ×4, 0.75 |
| #3 FP8 | 0.7867 | 0.7933 | +0.007 [−0.008, +0.023] | 0.65 ×4, 0.75 |

≈ +0.17 contest points, CI includes 0, and requesting logprobs costs +8% latency (EXP-022), which is ranked. **No.**

### 6. Big Five structure

The gold tags are not Big Five-consistent: `emotionally_stable` + `sensitive` co-occur 150 times in train (93 expected by chance). Exclusive pairs exist (`brief`/`detailed` 0 vs 70 expected; `formal`/`colloquial`), but our blend never outputs them together on the 600 points. **Nothing to fix.**

### 7. Rare-class oversampling (Zakaria B1): the ceiling

Rare classes (surprise, care, disgust, joy, pride) are 4.3% of the 2,354 val points. Of #3's 128 errors on the 600 points, **10** are rare-class golds: fixing *all* of them = +1.7 accuracy points = **+0.42 contest points**; 54 errors are predicted anxiety. The contest scores accuracy, not macro-F1. A retrain costs ≈ 3.7 card-hours (T7) plus a fresh boot for evaluation (GPUs break after training), plus merge and check run; 3.93 card-hours are left. **Not this round.**

## Zakaria's P2 package, item by item

| Item | Verdict |
|---|---|
| A1 filter off (`rescore_profile_filter.py`) | Tested twice (EXP-022, here with the blend): +0.001…+0.002 F1, noise. The script keys outputs and gold by `conversation_id`, so only one point per conversation is scored, and it averages per-sample F1 instead of eval.py's micro-F1. |
| A2 audit + relabel ambiguous labels | Needs a Chinese reviewer and a retrain. Risky in principle: the test gold almost certainly comes from the same annotation process. Where they split 38% anxiety / 24% relaxed on the same kind of message, predicting their majority is what maximises accuracy; "correcting" train labels toward what we think is right moves the model away from the test annotators. |
| B1 oversample rare emotions | Ceiling +0.42 points (§7); costs a full retrain we can't afford. |
| B2 dev loss + `load_best_model_at_end` | Good practice for the final. Adds eval time every 100 steps; `save_steps` must be a multiple of `eval_steps` (it is: 100/100). |
| B3 teacher paraphrases | Organisers allow generated training data (answer #11). For the final. |
| Points table in the doc | Uses 40/30/30 + 30; the official split is 25 emotion accuracy / 20 reply / 15 profile / 10 latency / 30 expert review. Macro-F1 isn't scored. |

## Decision

Submission #3 (FP8 + profile blend) stays our final round-1 entry; the 2 remaining submissions stay as a buffer. Ideas for the final: appraisal-based handling of fear / anxiety / relaxed, Generative-Agents-style memory with reflection (a running trait profile, causal only), persona-agent evaluation (MatrAIx), dev-loss checkpointing and generated rare-class data (Zakaria B2/B3).
