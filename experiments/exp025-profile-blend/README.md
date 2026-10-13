# EXP-025: submission #3 candidate = #2 + profile blend (LLM tags + CPU classifier)

| | |
|---|---|
| **Date** | 2026-10-13 (built and tested offline), check run on Migu pending |
| **Author** | Ismail F. |
| **Goal** | Ship the EXP-024 profile blend in the submission: same model, same prompt, same emotion and reply; only `user_profile` changes. |
| **Status** | Code + tests done (19/19 + 42/42). Migu check run pending. |

## What changes vs submission #2

| | #2 | Candidate |
|---|---|---|
| Model, prompt, vLLM args, `start.sh` | — | unchanged |
| `run_inference.py` | sha256 `7408b4ab…` (Oct 9) | `ec3cfdf5…` = repo main: Z1 truncation (off), emotion bias (off), **profile classifier (on)** |
| `profile_clf/` | — | `profile_clf.json` (`6b3c37c2…`: 300k char n-grams, tags, blend), `profile_clf_1.npz` (`7717df76…`) + `profile_clf_2.npz` (`ba198df5…`): idf, biases, 33 × 300,001 LR weights split in two files < 30 MB |
| `configs/inference_config.json` | — | + `"profile_classifier": "profile_clf"` |

Per field, a tag is kept if `a · [LLM output has it] + (1 − a) · p_classifier ≥ t`:

| Field | a | t | In words |
|---|---|---|---|
| personality_traits | 0.2 | 0.25 | LLM tag kept if p ≥ 0.06; classifier adds a tag if p ≥ 0.31 |
| interests | 0.5 | 0.15 | LLM tags always kept; classifier adds a tag if p ≥ 0.30 |
| style | 0.0 | 0.15 | classifier only: tag if p ≥ 0.15 |

(a, t) = the most frequent 5-fold-CV pick in EXP-024, and also the best fit on all 600 val points.

## Expected effect (EXP-024, 600 val points, submission #2's own raw answers)

| Profile micro-F1 | tune500 | n100 | All 600 |
|---|---|---|---|
| #2 | 0.456 | 0.481 | 0.461 |
| Candidate, CV estimate (honest) | 0.488 | 0.523 | **0.496** (+0.035, 95% CI +0.014 to +0.055) |
| Candidate, final settings (in-sample) | 0.499 | 0.530 | 0.506 |

Exact match 0.534 → 0.558. Emotion, reply text, ids and order are identical by construction (tested). ≈ +0.5 points if profile is scored as micro-F1, ≈ +0.25 if exact match.

## Implementation

`ProfileClassifier` in `participant/run_inference.py`: a numpy re-implementation of sklearn's `TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)` + `LogisticRegression.predict_proba` (no sklearn in the image needed). Loaded once before the timed loop; the per-sample work (≈ 1 ms) is inside the timed region. Prints `profile classifier exp025-v1: tags added …, removed …` at the end. `performance_report.json` keys unchanged.

Export: `experiments/exp024-cpu-classifier/finalize.py` (from the EXP-024 model trained on 90% of `train_public` conversations).

## Tests (`tests/test_profile_blend.py`, 19/19; `tests/test_emotion_bias.py` still 42/42)

1. numpy probabilities = sklearn's on the 600 val points (max |diff| 1.8e-8), 0 blend decisions flipped; ≈ 1 ms per sample; blend settings = the CV picks.
2. **Replay** of #2's real raw answers (`outputs/exp022/lp_bf16.jsonl`) through `run_inference.py` on a stub server: no key → profile F1 0.4609 and byte-identical output to the old `run_inference.py`; key on → **0.5055** on 600 (= the offline number), only `user_profile` differs, `validate.py` 0 errors.
3. Bad `profile_classifier` values rejected at start-up.
4. `scripts/exp025_session.sh` with a fake vLLM: PASS path keeps the candidate installed; FAIL path restores #2's files exactly and removes `profile_clf/`.

## Check run (one 2-GPU boot): pass rules, fixed before the run

`scripts/exp025_session.sh`: install (backup of #2 in `/root/sub2_backup_exp025`) → plain `bash /root/participant/start.sh t9_input.jsonl` → `validate.py` → `eval.py` on n100. PASS needs all of:

- `validate.py` 0 errors, 0 parse failures, `complete: true`
- emotion accuracy on n100 ≥ 0.80 (T9: 0.82)
- profile micro-F1 on n100 ≥ 0.4806 (#2 on the same 100 points)
- latency mean ≤ 1.65 s (T9: 1.55 s)

FAIL → #2's files are restored automatically. PASS → shut down, save image, submit #3.

## Results

(pending)
