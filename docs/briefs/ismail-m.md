# Brief: Ismail M. (fine-tuning lead)

**Tasks:** B6 `build_sft.py` (due Oct 4), then I1 LoRA fine-tuning on AutoDL (Oct 9).

## B6: `train/build_sft.py`

Turn the train set (8,338 conversations, 18,657 prediction points) into chat-format SFT examples:

```json
{"messages": [
  {"role": "system", "content": "<same system prompt as inference>"},
  {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."},
  ...
  {"role": "assistant", "content": "{\"emotion_label\": ..., \"response_text\": ..., \"user_profile\": ...}"}
]}
```

**Decisions to make, and write down in the PR:**

- **Target format:** is the final assistant turn the full JSON (so we train label prediction and the reply together, like MoEL) or only the reply text?
- **Long conversations:** the longest is about 40k characters. Pick a max token length (say 4k), and decide how to truncate. Keep the most recent turns.
- **Train/dev split:** split by `conversation_id`, not by prediction point. Points from the same conversation must not end up in both sets.
- Write the system prompt **exactly** as inference uses it (`prompts/`). If training and inference prompts differ, the fine-tune is wasted.

## I1: LoRA on AutoDL

- **Base model:** probably Qwen2.5-7B-Instruct with LoRA or QLoRA, to fit in 24 GB. We'll confirm after the organisers say what model size is allowed.
- **Always do a smoke test first:** 50 examples, 1 epoch, check the loss goes down, then save → load → generate. Only after that, the full run.
- Estimate the GPU-hours and log them in LOG.md before every run.
- Deliverable: merged weights, or an adapter plus a loading snippet, that `run_inference.py` can load **offline**.
