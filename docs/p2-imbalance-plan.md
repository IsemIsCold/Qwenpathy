# P2 plan: the anxiety bias, checked against our pipeline

| | |
|---|---|
| **Date** | 2026-10-10 |
| **Input** | Yahya's research report (Kimi, Oct 9), re-checked against our own data, the T7/T9 outputs and the official rules |
| **Scope** | What a second training run (P2) can realistically fix, and what it costs |
| **Status** | **Decided Oct 11: no emotion-targeted P2.** Submission #2 stays. See *Decision* below. |

## Decision (Oct 11)

P1 step 1 is done (`docs/p1-error-review.md`, reviewed by Ismail F. from English summaries): of the 17 errors, **8 model wrong, 6 arguable, 3 gold wrong**.

- About half the errors are not real model mistakes.
- The real ones cluster where a user is worried and then says they feel better (#5, #8, #9 in the review sheet). For exactly that message type the training gold says `anxiety` 38% vs `relaxed` 24% (§4), and the test set almost certainly comes from the same labeller. Predicting `anxiety` there is the bet that is right most often; teaching the model otherwise would fix these and break more anxiety-labelled test cases.
- **So: no second training run aimed at emotion.** Submission #2 stays the model to beat. Card-hours go to latency (E4/P3) and the remaining reserve.
- P2 reopens only for option B (rare-class generated data) **if** the organisers allow generated data, and only through the gate in §6.

## TL;DR

1. **The official metric is emotion accuracy, not macro-F1.** We optimise accuracy and keep anxiety recall at 1.0.
2. **Every emotion error the model makes is a wrong `anxiety`** or a near-neighbour mix-up. Anxiety recall is 1.000 and precision 0.76–0.77 on both slices.
3. **It is not data scarcity.** The two main confusions involve `fear` (1,044 train examples) and `relaxed` (2,509), both well represented.
4. **The training gold is itself split on these messages.** Messages that contain both a worry word and a relief word are labelled `anxiety` 38% and `relaxed` 24% of the time in train. The model learned that majority. Reweighting or oversampling can't make labels consistent, so the expected gain on these errors is small.
5. **Most techniques in the original report need a classification head; we don't have one.** What fits: oversampling rare classes in `build_sft.py`, generated rare-emotion data (only if allowed), and a dev-loss eval. P2 runs only if P1 finds a fix with a clear expected gain.

## 1. What we are optimising

| Component | Points | Metric |
|---|---|---|
| Emotion | 25 | **accuracy** (our `eval.py` marks macro-F1 as diagnostic only) |
| Response quality | 20 | BLEU-4 / ROUGE-L / BERTScore vs gold, so greedy decoding stays |
| Profile | 15 | average; F1 vs exact match still unclear (O1) |
| Latency | 10 | first 100 samples |
| Expert review | 30 | integrity, code quality, records |

**Anxiety is 31.9% of val points.** Any change that trades anxiety recall for rare-class recall has to win back more than it loses. 1 accuracy point on the test set ≈ 0.25 competition points.

**Macro-F1 has a ceiling on our slices.** `eval.py` averages all 16 labels, including labels with zero support in the slice, which score F1 = 0 even with no mistakes.

| Slice | Labels present | Max macro-F1 | T9 macro-F1 | Share of max |
|---|---|---|---|---|
| n50 | 12 / 16 | 0.75 | 0.509 | 68% |
| n50b | 13 / 16 | 0.81 | 0.574 | 71% |

On 50 points, macro-F1 also swings by a lot from a single sample of a rare class. Use it as a diagnostic, never as a gate.

## 2. Where the errors are (T7 outputs; T9 is identical within noise)

n50: 10 errors. n50b: 7 errors. Every error is listed, with the last user message paraphrased.

| Slice | Gold → pred | What the user says (paraphrase) |
|---|---|---|
| n50 | relaxed → anxiety | "Your question made me feel much lighter… I overthink easily… what should I send so he doesn't feel pressured?" |
| n50 | relaxed → anxiety | "I'm afraid people will think I'm petty… but after talking to you I feel much more at ease." |
| n50 | relaxed → anxiety | "I'm just afraid people find me hard to get along with… hearing you, I feel a bit more confident." |
| n50 | gratitude → anxiety | "Thanks for breaking it down, I feel clearer… but there's someone I'm afraid to contact." |
| n50 | gratitude → relaxed | "That you understand is already great… I'm at ease now, no advice needed." |
| n50 | anger → anxiety | "Let me vent first… I'm both scared and annoyed, I just want someone to listen." |
| n50 | helplessness → loneliness | "Nobody to talk to… should I meet him or wait two days?" |
| n50 | loneliness → sadness | "Days at work are fine, the nights alone at home are the hard part." |
| n50 | neutral → sadness | "No advice please. The offer isn't ideal. I'll digest it myself." |
| n50 | care → mixed | "The light bothers me, but I mostly feel for my colleague with migraines." |
| n50b | fear → anxiety | "What I'm most afraid of is that his silence means he doesn't care… how can I tell?" |
| n50b | fear → anxiety | "What stops me is the possibility that I don't want to stay here at all… I don't want to face that." |
| n50b | fear → anxiety | "I only dare focus on two chapters… thinking of last time, I'll probably mess it up again." |
| n50b | mixed → anxiety | "I have an admin offer my parents like, but it feels stifling; whatever I pick feels wrong." |
| n50b | helplessness → mixed | "What bothers me is that it's out of my control… let's keep it light." |
| n50b | care → neutral | "OK, I'll stop dithering. Give me a plan for the first week of evening walks." |
| n50b | neutral → relaxed | "OK, that's about it for this part. Let's stop here." |

**Pattern:** 9 of 17 errors predict `anxiety`, and almost all of them are messages that contain 怕 (*pà*, "afraid / worried") **together with** another feeling: relief, thanks, annoyance, a concrete feared outcome. Several gold labels are arguable even for a human reader (anger vs anxiety in "both scared and annoyed", gratitude vs relaxed in "thanks… I'm at ease now").

## 3. Is it data scarcity? No

Training set (`data/sft/build_report.json`, 14,942 examples):

| anxiety | relaxed | helplessness | fear | sadness | mixed | loneliness | anger |
|---|---|---|---|---|---|---|---|
| 4,797 | 2,509 | 1,202 | 1,044 | 908 | 799 | 721 | 679 |

| gratitude | shame | neutral | pride | joy | disgust | care | surprise |
|---|---|---|---|---|---|---|---|
| 623 | 593 | 499 | 196 | 145 | 102 | 72 | 53 |

The two biggest confusions (fear → anxiety, relaxed → anxiety) involve the 2nd and 4th most common classes. The truly rare classes (pride, joy, disgust, care, surprise) together are **4.3% of val points**. Even a perfect fix there is worth at most ~4 accuracy points, realistically ~1.

## 4. Is the gold consistent on these messages? Not very

Gold emotion of training examples whose last user message matches a keyword pattern:

| Last user message contains | Share of train | Gold label split |
|---|---|---|
| (all) | 100% | anxiety 32%, relaxed 17%, helplessness 8%, fear 7% |
| 怕 (*pà*, afraid/worried) | 24.5% | anxiety 45%, fear 24%, relaxed 7%, helplessness 5% |
| 怕 + a relief word (踏实 *tāshi* "at ease", 轻松 *qīngsōng* "relaxed", 安心 *ānxīn* "reassured", …) | 2.2% | **anxiety 38%, relaxed 24%**, fear 11%, gratitude 6% |
| 怕 + thanks (谢谢 *xièxie*) | 1.5% | relaxed 30%, gratitude 23%, anxiety 20%, fear 8% |
| strong fear words (最怕 *zuì pà* "most afraid", 害怕 *hàipà* "scared", 恐惧 *kǒngjù* "dread", …) | 5.0% | **fear 47%, anxiety 31%**, shame 5% |

Word order doesn't decide it either. With the relief word **after** 怕 the split is anxiety 36% / relaxed 25%; with 怕 after the relief word it is anxiety 40% / relaxed 24%.

**Reading:** for the exact message type the model gets wrong, the training labels disagree with each other, and `anxiety` is the most common answer. The model is doing what the data taught it. Reweighting or oversampling `fear`/`relaxed` would push these messages the other way and create new anxiety → fear/relaxed errors on the anxiety-labelled ones. Expected net gain: close to zero.

Caveat: keywords are a crude proxy. The data generator may follow cues we can't see with a regex, so the P1 reading in §6 is still needed before deciding.

## 5. The original report's techniques vs our pipeline

Our model writes the emotion as text inside the JSON answer. LLaMA-Factory trains with next-token cross-entropy over the vocabulary, not over 16 classes.

| Technique (report) | Fits our pipeline? | Verdict |
|---|---|---|
| Class-balanced loss on an "emotion head" | No head. The closest equivalent is **oversampling** examples by gold emotion in `build_sft.py` | Rare classes only (see §6), mild. Not for fear/relaxed (§4) |
| Larger LoRA rank (r=64) | Yes, a config change | No evidence capacity is the bottleneck. Don't combine it with another change in a single run |
| LLM-generated rare-emotion data | Yes: DeepSeek Flash pipeline (EXP-017, Z4) | Only if the organisers allow generated data (O1). Cap 10–15% |
| Label smoothing | Only as `label_smoothing_factor`, which applies to every token including the reply | Skip: it would also smooth the reply text that BLEU/ROUGE score |
| Focal loss, LDAM, mixup, contrastive, hierarchical heads | No: each needs a classification head or a custom trainer | Skip |
| Profile BCE `pos_weight` | No: profile is generated text too | Skip. The real profile lever is the tag-filter re-score (P1) |
| Temperature schedule / "affective" beam search | Hurts reference metrics; not available in the vLLM OpenAI server | Skip: greedy stays |
| Dev-set eval loss during training | Yes: `val_size` / `eval_steps` | Add to any P2 run (a few minutes per eval) |
| 2–3 epochs | Yes | Not by itself: the last quarter of T7 gained only −0.02 loss (EXP-019 README) |

## 6. Recommended plan

### P1: no GPU, do first

1. **Read the 17 errors above with a Chinese reader** (Ismail M. or Elly) and mark each one: *model wrong*, *gold arguable*, or *gold wrong*. Sheet with the full Chinese text: `docs/p1-error-review.md`. If most are arguable, emotion-targeted P2 is off.
2. ~~Profile tag filter re-score~~ **Dropped (Oct 10).** `submission.jsonl` already has the tags removed and `start.sh` saves no raw outputs, so it would need a GPU re-run. It isn't worth one: the three tags are rare in gold (val: `study_exam` 4.0%, `colloquial` 2.3%, `emotionally_stable` 0.4% of points), so turning the filter off can win back very little and risks false positives.
3. **`analyze.py --compare`** (Yahya's v3) on T9 vs V1.8, to look for anything else that got worse.

### P2: only if P1 justifies it (Oct 11: P1 did not; option A dropped, option B waits for O1)

| Option | Change | When | Est. card-hours |
|---|---|---|---|
| A | Oversample the 5 rarest classes (surprise, care, disgust, joy, pride) toward ~√-frequency, e.g. ×3 for surprise/care, ×2 for disgust/joy, ×1.5 for pride; everything else ×1. One flag in `build_sft.py` | P1 shows the rare-class misses are *model wrong*, not gold noise | ≈ 4.5 (train 3.6 + eval 0.4 + check 0.5) |
| B | A + generated rare-emotion examples (Flash) | Organisers say generated data is allowed | ≈ 4.5 + ~$2 API |
| — | Fear/relaxed oversampling, loss reweighting, rank change | — | not planned (§4, §5) |

**Gate for any P2 model:** it must beat T9 on **accuracy** on both n50 and n50b, keep anxiety recall ≥ 0.95, 0 parse failures, and latency not worse than 1.6 s. Otherwise submission #2 stays.

**Budget:** 28.67 card-hours after Oct 4, minus ≈ 6.6 estimated for Oct 7–9 (actuals pending, E2) ≈ **22 left**. Reserve 5. One P2 run fits; two would leave no room for P3 (AWQ latency test, ~1) or a failed submission.

### Where else the hours could go

- **Latency (10 points):** T9 mean 1.55 s vs 1.41 s for submission #1. P3/E4 is likely worth more than a P2 run whose emotion gain is uncertain.
- **Profile (15 points):** no cheap lever left. The tag filter is not worth a re-run (P1 step 2).

## 7. Corrections to the original report (for the record)

- The competition is ranked on emotion **accuracy**. The report's claim that organisers chose macro-F1, and its advice to give up anxiety accuracy for it, is wrong for us.
- Class counts in the report's code (`[750, …, 10]`, "surprise ≈ 10 examples") are invented. The real train counts are in §3 (surprise 53).
- "anxiety F1 > 0.95, relaxed/pride F1 < 0.20" is false. T9 per-class F1: anxiety 0.865 / 0.872, relaxed 0.842 / 0.952, pride 1.0 (n50).
- Trainable parameters are 0.53% (43.6M), not 0.24%. Hardware is 2× RTX 4090, not A100/H100.
- "emotionally_stable 47%, colloquial 76%" are the V0 prompted model's over-predictions (EXP-013), not properties of our data or of the fine-tuned model.
- Citation [^4^] "EmotiAug, ACL 2025" links to *Nemotron-CC* (a pretraining-data paper). Citation [^7^] "CUE-Bench" links to *The Linguistic Connectivities Within Large Language Models*. No EmotiAug paper was found.
- MoodBench 1.0 exists (arXiv 2511.18926), but it is an evaluation benchmark (1,200 data sets, 41 tasks, top overall score 70.09 by gpt-5-mini). The "2,898 dialogues, 28 emotions, 0.72–0.78 macro-F1 leaderboard, +0.08 from class-balanced loss" numbers attributed to it are not in the paper.
- Nothing from §11 of the report ("Chinese Emotional AI Landscape") should be quoted in our docs: the expert review checks integrity.

## 8. Side check closed: long5 / context length

The 5 longest val points outside n50/n50b (`data/val_input_long5.jsonl`) have 8,979–11,729 characters of history. At ~1.6 chars/token that is ≤ ~7.5k tokens + 355 (system prompt) + 1,200 (output), far below the 32k `max_model_len`. No request in val can hit an HTTP 400. Test conversations longer than anything in val or train remain Z1's job (truncation).

## Sources

- Yahya, *EXP-019 Deep Research Report: Breaking the Anxiety Ceiling* (Kimi, Oct 9), not in the repo
- `data/sft/build_report.json`, `data/sft/train.jsonl`, `data/val_*_n50*.jsonl`, `experiments/exp019-qlora-full/n50*/submission.jsonl`, T9 eval output (`/mnt/storage/out/t9/eval_n50*.json`)
- [MoodBench 1.0, arXiv 2511.18926](https://arxiv.org/html/2511.18926v1)
- [ACL 2025 long #123 (Nemotron-CC)](https://aclanthology.org/2025.acl-long.123/) · [ACL 2025 Findings #456](https://aclanthology.org/2025.findings-acl.456/)
