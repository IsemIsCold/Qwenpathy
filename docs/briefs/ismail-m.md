# Brief: Ismail M. (training lead)

> **Original brief from Oct 1, kept for reference.** Some details are outdated (e.g. the base model is now Qwen3-8B, prompts live in `run_inference.py`). Current status: [docs/team-work.md](../team-work.md) and [TASKS.md](../../TASKS.md).

**You own:** the model we ship. That means choosing the base model, fine-tuning it with LoRA/QLoRA on AutoDL, and handing over weights that load offline.

**Tasks:** B8 training plan (Oct 4), I2 LoRA runs (Oct 9). You're also the team's ML reference, so others will come to you with questions.

## B8: `docs/training-plan.md`

Decide and write down:

- **Base model + size:** for example Qwen2.5-7B-Instruct. Check that it fits the eval hardware (probably 2×24 GB) **and** the latency budget. We're still waiting on the organisers' answer about model size limits.
- **Method:** LoRA or QLoRA, rank, target modules, learning rate, epochs, max sequence length (use Zakaria's token stats).
- **Target format:** do we train the model to output the full JSON (emotion + profile + reply) or only the reply? Agree it with Zakaria (data) and Ismail F. (classification).
- **Compute estimate:** GPU type and hours on AutoDL, plus the cost in ¥.

## I2: LoRA runs

1. **Smoke test first:** 50 examples, 1 epoch. Check that the loss goes down, then save → load → generate on 3 val samples.
2. Full run. Log the hours and cost in LOG.md **before** you start it.
3. Hand Yahya the predictions on val for scoring. A run only counts once `eval.py` shows it beats the baseline.
4. Deliverable: merged weights, or an adapter plus a loading snippet, that `run_inference.py` loads **offline**. Agree the format with Elly.

**Hint:** train and infer with the exact same system prompt and chat template. If they differ, fine-tuning gains quietly disappear.
