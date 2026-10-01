# Brief: Elly (prompts + reply quality)

**Tasks:** B5 the emotion-first prompt (due Oct 3), then I2 prompt iterations (ongoing).

## Background: why "emotion first"

The paper *MoEL: Mixture of Empathetic Listeners* (Lin et al., 2019) found that replies are more empathetic when the model **first works out the user's emotion, then answers in a way suited to that emotion**. We don't need their architecture. We copy the idea in the prompt.

## B5: `prompts/system_v1.txt`

Start from the official prompt in `run_inference.py` and change it so that:

1. The model outputs `emotion_label` **before** `response_text` in its JSON. It "decides" the emotion first, and then the reply is written with that decision already made.
2. There's short guidance for the top 5 emotions in the data:
   - `anxiety` (34%)
   - `relaxed` (15%)
   - `helplessness` (8%)
   - `fear` (7%)
   - `sadness`

   Two or three lines each, on how a good companion responds. For example, anxiety: acknowledge the worry → make it concrete → one small next step. Loneliness: presence and warmth, not advice.
3. Reply length is about 100–130 Chinese characters, which is the median in the data.

**Hints:**

- Read 10–20 real gold replies from the train set before you write. Copy their tone, not your own idea of empathy.
- Keep the prompt in Chinese if the replies must be in Chinese. Small models follow same-language instructions better.
- Small models (0.5B–7B) break JSON easily. Keep the output format instructions short and strict, and give one example.

## How to test

You only ever change one thing at a time. For each change:

1. Run on about 50 val samples.
2. Check that `validate.py` passes.
3. Run `eval.py`.

Then write a line in the PR: `v1 → v2: changed X, emotion macro-F1 0.21 → 0.24`.
