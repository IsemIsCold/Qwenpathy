# Brief: Ismail F. (classification + captain)

> **Original brief from Oct 1, kept for reference.** Some details are outdated (e.g. the base model is now Qwen3-8B, prompts live in `run_inference.py`). Current status: [docs/team-work.md](../team-work.md) and [TASKS.md](../../TASKS.md).

**You own:** `emotion_label` and `user_profile`, the system prompt, merging PRs, and **all submissions**.

**Tasks:** B5 emotion-first prompt (Oct 3), B9 submission #1 (Oct 4), I3 dedicated classifier (Oct 10), I6/F2 submissions.

## B5: emotion-first prompt (`prompts/system_v1.txt`)

The idea comes from MoEL (Lin et al., 2019): first work out the emotion, then reply in a way suited to it.

- Output `emotion_label` **before** `response_text` in the JSON.
- Add 2–3 lines of guidance each for `anxiety`, `relaxed`, `helplessness`, `fear` and `sadness`.
- Target reply length is about 100–130 characters.
- Measure it: 50 val samples → `validate.py` → `eval.py`, and compare against the official prompt.

## I3: dedicated classifier

Labels are very imbalanced, and small LLMs are poor classifiers. Test whether a small Chinese encoder (for example `chinese-roberta-wwm-ext`), fine-tuned on the train set, beats the LLM:

- **Emotion:** single-label, 16 classes. Try class weights for the imbalance.
- **Profile:** multi-label. That's one sigmoid per label with a threshold, not a softmax.
- Input: the last user message plus some context. Try a few window sizes.
- If it wins on val, the final pipeline becomes: classifier → labels, LLM → reply (conditioned on the predicted emotion).

## Captain duties

- Merge PRs only when `main` still runs end to end.
- Go through the CONTRIBUTING.md checklist before every submission, and log it in LOG.md.
- Chase the organisers' answers and record them in `docs/organiser-answers.md`.
- Post the evening sync summary.
