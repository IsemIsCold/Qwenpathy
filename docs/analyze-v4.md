# analyze.py v4 — what changed and how to read it

| | |
|---|---|
| **Version** | 4.0 (Oct 6), built on Yahya's v3 "Error Genome" |
| **Tests** | `python tests/test_analyze.py` → 74/74 (Yahya's 57 + 17 new v4 tests). Needs `data/val_gold.jsonl` + `data/val_input.jsonl` |
| **Scores** | always from `scripts/eval.py` (imported, never copied) |

## Why v4

v3 had three problems that made its output look precise but mean little:

1. **Clusters were unions.** Each error was hashed together with all its context (turn, keywords, reply length), so almost every sample became its own signature (count 1–2) and was filtered out. Whatever survived was merged with anything sharing *one* condition, and the cluster listed *all* conditions: e.g. "pred_long + pred_medium + turn_4plus" (contradictory, and turn_4plus is in almost every sample).
2. **Counterfactual points were formulas**, e.g. replies `count/n × 20 × 0.5`, profile assuming exact match.
3. **The genome was not deterministic.** "Last run" came from a Python `set`, so on our own runs V0 → V1.3 → V1.4 → V1.7 the persistent/resolved counts changed with `PYTHONHASHSEED` (2 different results over 4 seeds).

## What v4 does

| Part | v4 |
|---|---|
| Signatures | one per error type: `emotion.confusion fear→anxiety`, `profile.false_positive personality_traits:emotionally_stable`, `response.too_short`… |
| Regimes | a context condition is attached to an error group only if it covers ≥ 60% of the group, is ≥ 1.5× more frequent there than overall, and appears in ≤ 80% of all samples. Otherwise: "none (spread out)" |
| Counterfactuals | **measured**: the fix is applied to the predictions and re-scored with eval.py. Emotion = oracle upper bound. Profile false positive = exactly what a post-filter does (true positives lost included, so a harmful filter shows negative points). Profile points shown for **both** F1 and exact match. Replies: no points (official composite unknown), only ROUGE-L of the group vs the rest |
| New tables | prediction distribution + **collapse check** (no label > 60%), emotion accuracy **by turn and by history length**, **bidirectional confusions** (a→b and b→a ≥ 2: label-ambiguity signal) |
| Genome | stable signatures, run order kept in registration order (deterministic) |
| Advice | only what the plan allows: no synthetic data, no temperature/top_p, no class-balanced loss |
| Speed | ≈ 0.5 s on the full val set (2,354 points); per-sample ROUGE-L computed once |

Sanity check on V0 (n50): the measured `study_exam` filter gives interests micro-F1 0.361 → 0.512 and exact match 0.10 → 0.34, exactly what we computed by hand in EXP-013.

## How to use it

```powershell
# one run
python scripts\analyze.py <submission.jsonl> data\val_gold_n50b.jsonl --input data\val_input_n50b.jsonl --outdir <dir> --run-label pilot

# pilot gate (T6): new model vs V1.8 on the same slice
python scripts\analyze.py <pilot_submission.jsonl> data\val_gold_n50b.jsonl --input data\val_input_n50b.jsonl `
    --compare <v18_submission.jsonl> --outdir <dir> --run-label pilot
```

Add `--force` for a submission with blank emotion labels. The genome registry defaults to `<outdir>/.analyze_registry`; pass the same `--registry` to several runs to track signatures across models.

## How to read it (rules for decisions)

- **Decide on n50 + n50b together**, never on one slice. A filter found and measured on one slice overfits it.
- **Collapse check** must say `ok` for any fine-tuned model (T6 gate).
- **Regimes are descriptions, not causes.** "gratitude→relaxed: hist_500_1500 (100% vs 42%)" on 3 samples is a lead to read, not a finding.
- **Profile points**: until the organisers confirm the metric, a change is only "safe" if it is ≥ 0 under both F1 and exact match.
