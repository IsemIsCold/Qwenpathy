# analyze.py v4.1 — what changed and how to read it

| | |
|---|---|
| **Version** | 4.1 (Oct 6): v4 + the 10 review fixes. Built on Yahya's v3 "Error Genome" |
| **Tests** | `python tests/test_analyze.py` → 102/102 (Yahya's 57, 17 v4, 28 v4.1). Needs the **full** val set `data/val_gold.jsonl` + `data/val_input.jsonl` (gitignored; build with `convert_val.py`), not n50/n50b. ≈ 25 s |
| **Scores** | always from `scripts/eval.py` (imported, never copied) |

## Why v4

v3 had three problems that made its output look precise but mean little:

1. **Clusters were unions.** Each error was hashed together with all its context (turn, keywords, reply length), so almost every sample became its own signature (count 1–2) and was filtered out. Whatever survived was merged with anything sharing *one* condition, and the cluster listed *all* conditions: e.g. "pred_long + pred_medium + turn_4plus" (contradictory, and turn_4plus is in almost every sample).
2. **Counterfactual points were formulas**, e.g. replies `count/n × 20 × 0.5`, profile assuming exact match.
3. **The genome was not deterministic.** "Last run" came from a Python `set`, so on our own runs V0 → V1.3 → V1.4 → V1.7 the persistent/resolved counts changed with `PYTHONHASHSEED` (2 different results over 4 seeds).

## v4.1: review fixes (Oct 6)

| # | Problem in v4.0 | Fix |
|---|---|---|
| 1 | Regimes on n50 were noise: shuffling the conditions 200× still gave 6.7 "regimes" (real data: 8) | min group 8 + hypergeometric test. Null test now **0.05** on n50, 0.5 on full val; a planted regime is still found (test C02) |
| 2 | `--compare` only said "see delta.csv"; `emotion_delta` meant "changed", not "improved" | v2.2 table restored (old/new correct, improved, regressed, unchanged, net) |
| 3 | Hypotheses still showed formula points; correlations used emotion's 25 pts for profile combos | ranked by affected samples, no points |
| 4 | Oracle upper bounds and real filters were ranked in one list | three tables (A filters, B oracle, C replies) |
| 5 | Genome said "novel" for everything by default; no check that runs used the same slice | hidden without history; gold fingerprint; other-slice runs skipped with a warning |
| 6 | "pts if F1" hid its assumption | stated in the report |
| 7 | New reply error types had no description | added |
| 8 | Reply row said "before → after" but showed group vs rest | own table, own columns |
| 9 | `--mistakes-only` help said "skip full report" | help fixed (report still written) |
| 10 | Tests silently needed the full val files | documented at the top of the test file; clear exit if missing |

Also: the taxonomy table now shows occurrences, not just distinct types; paths show `v1_7_n50b/out_nothink/submission.jsonl`, not just `submission.jsonl`.

## What v4 does

| Part | v4 |
|---|---|
| Signatures | one per error type: `emotion.confusion fear→anxiety`, `profile.false_positive personality_traits:emotionally_stable`, `response.too_short`… |
| Regimes | a context condition is attached to an error group only if the group has ≥ 8 samples, the condition covers ≥ 60% of it, is ≥ 1.5× more frequent there than overall, appears in ≤ 80% of all samples, **and** passes a hypergeometric test (Bonferroni p < 0.05). Reply-length conditions are not tested on reply groups (circular) |
| Counterfactuals | **measured**: the fix is applied to the predictions and re-scored with eval.py, in **three separate tables**: A. post-filters (achievable now, true positives lost included), B. oracle upper bounds (emotion, missed profile labels: not achievable by a rule), C. reply groups (no points; ROUGE-L in group vs rest). Profile points for **both** F1 and exact match, assuming profile score = mean of the 3 per-field metrics × 15 |
| New tables | prediction distribution + **collapse check** (no label > 60%), emotion accuracy **by turn and by history length**, **bidirectional confusions** (a→b and b→a ≥ 2: label-ambiguity signal) |
| Genome | stable signatures, deterministic run order; compares only runs on the **same gold file** (sha256 fingerprint), warns about the others; hidden when there is no earlier run |
| Hypotheses | ranked by affected samples; no point formulas (measured points are in the counterfactual tables) |
| `--compare` | improved / regressed / unchanged **vs gold** per field, with net delta; `delta.csv` has old/new correctness per sample |
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

Add `--force` for a submission with blank emotion labels. The genome registry defaults to `<outdir>/.analyze_registry` (so the genome section is empty). To track error types across models, pass the same registry to every run on the same slice:

```powershell
python scripts\analyze.py <sub> data\val_gold_n50b.jsonl --input data\val_input_n50b.jsonl --outdir <dir> `
    --run-label v1.8 --registry experiments\.analyze_registry_n50b
```

## How to read it (rules for decisions)

- **Decide on n50 + n50b together**, never on one slice. A filter found and measured on one slice overfits it.
- **Collapse check** must say `ok` for any fine-tuned model (T6 gate).
- **Regimes are descriptions, not causes.** On n50 expect almost none (groups are too small); that is the honest answer. They become useful on bigger slices.
- **Read `--compare` by net delta.** On 50 samples ±3–4 samples is noise. Example (V1.7 vs V0, n50b): emotion +1 (noise), interests +14, style +5 (real).
- **Profile points**: until the organisers confirm the metric, a change is only "safe" if it is ≥ 0 under both F1 and exact match.
