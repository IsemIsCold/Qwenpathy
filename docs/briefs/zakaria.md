# Brief: Zakaria (evaluation)

**Tasks:** B4 `eval.py` (due Oct 3), then I3 profile prediction (Oct 10).

Your script is how we decide everything: which prompt, which model, and whether a submission is worth one of our 5 attempts.

## B4: `scripts/eval.py`

`python scripts/eval.py outputs/submission.jsonl data/val_gold.jsonl`

The file formats come from Yahya's `convert_val.py`. Agree on them with him on day 1.

**Report:**

1. **Emotion:** accuracy, **macro-F1** and a per-label table (precision, recall, count).
2. **Profile:** F1 for each sub-field (`personality_traits`, `interests`, `style`). These are multi-label, so think about how to score a predicted set against a gold set.
3. **Baselines, for context:** the score of "always predict `anxiety`" and of "always predict an empty profile".
4. **Response length:** mean and median character length, predicted vs. gold.

**Hints:**

- Why macro-F1 and not just accuracy? About 34% of the labels are `anxiety`. A model that always says "anxiety" gets 34% accuracy and learns nothing. Macro-F1 averages over the classes, so the rare ones count.
- For the profile, look up micro- vs. macro-F1 and Jaccard similarity. Pick one, and write in the PR why you chose it.
- About 37% of gold profiles are empty. Decide what score an empty prediction against an empty gold should get.
- `sklearn.metrics` has most of this already. You don't need to write F1 by hand.
- Also save the results to a JSON file, so we can compare runs over time.

We don't have an official scoring formula yet (we've asked the organisers). Keep the metrics separate rather than inventing a single combined number.
