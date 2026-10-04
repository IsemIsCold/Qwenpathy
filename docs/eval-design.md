# Evaluation System Design — Final Review (v1.0, 2026-10-04)

> Owner: Yahya (Evaluation). Status: APPROVED FOR IMPLEMENTATION.
> Team reading guide: `docs/interpreting-eval-output.md` (how to verify the evaluator and read every number).
> Scope: `validate.py` (B3), `eval.py` (B4), `analyze.py`, `judge.py` (I4), run registry.

---

## 1. Schema review (verified against real data, 2026-10-04)

Three artifacts, one contract each:

**Input** (`val_input.jsonl`, produced by `convert_val.py` — DONE, verified)
`{id, conversation_id, target_user_turn_id, history: [{turn_id, role, content}]}`

**Gold** (`val_gold.jsonl` — DONE, verified)
`{id, response_text, emotion_label, user_profile: {personality_traits[], interests[], style[]}, memory_refs[]}`

**Submission** (`submission.jsonl`, produced by `run_inference.py` — Elly)
Same 5 keys as gold, but predictions. This is the ONLY artifact our system consumes that we don't control.

Verified invariants (2,354 val points, all PASS):

| Invariant | Result |
|---|---|
| input/gold IDs identical, same order, no duplicates | PASS |
| history always ends at `target_user_turn_id` with a `user` turn | PASS |
| gold `response_text` never empty | PASS |
| all `memory_refs` match `mem_\d{6}` (empty in only 1.7% of gold) | PASS |
| all emotion labels in the official 16 | PASS |
| profile values: 0 out-of-vocab vs corrected vocab (train ∪ val verified) | PASS |
| files are clean UTF-8, no BOM | PASS |

Edge cases the validator MUST handle: Windows UTF-16 files (PowerShell `>` gotcha),
BOM, blank lines, CRLF, duplicate JSON keys, NaN/Infinity, non-object lines,
duplicate/missing/extra prediction IDs, empty `response_text`, wrong types.

**Locked decisions:**
- 16 emotions → hard reject. Structure/IDs → hard reject. Profile values →
  HARD REJECT (Oct 4 team review: vocabulary confirmed via the official
  participant package `templates/submission_schema.json`; toggle
  `PROFILE_OOV_HARD_FAIL = False` in validate.py reverts to warn-only).
- ID order mismatch between input and submission → WARN, not reject
  (official evaluator most likely matches by ID; we must not fail what the official accepts).
- No invented length caps on `response_text` or profile list sizes (gold reply max 1,809 chars;
  history max 14k chars val / 40k train).
- **Tolerated contract** (validator accepts, documented): blank lines, CRLF,
  UTF-8 BOM (stripped + warned), duplicate labels inside one profile/memory list.
- Input integrity: empty input, duplicate input IDs, or non-string input IDs
  → exit 2 (pipeline bug, not a submission bug) — added after review.
  Non-string submission IDs are flagged as structural violations; alignment
  set-ops only touch string IDs (no unhashable crashes).
- Control characters inside strings: tolerated (official contract does not
  prohibit them; val gold contains zero — verified 2026-10-04). Documented
  decision, not an accident.
- **eval.py gate**: refuses to score submissions failing record-level checks
  (reuses validate.py logic via import; no duplication). Override: `--force`.
  eval.py scope = core scorer (emotion/profile/baselines/length); BLEU/ROUGE/
  BERTScore, latency ingestion, bootstrap CI and paired comparison are later
  layers of the build order, intentionally not in this file.

## 2. Architecture (accepted, with the Layer-4 fix)

```
submission → validate.py → eval.py ──────► runs/<id>/metrics.json    [OFFICIAL PATH]
                              ├─► analyze.py ──► runs/<id>/report.html
                              ├─► judge.py  ───► runs/<id>/judge.json  [DEV-ONLY, side branch]
                              └──────────────► registry (append-only)
```

**Separation principle (enforced structurally, not by convention):**
validate never fixes · eval never diagnoses · analyze never changes the score ·
judge never defines the score · registry never loses an experiment.

- `metrics.json` contains deterministic metrics only.
- `judge.json` is a separate file with `source: "llm_judge"` provenance; never merged.
- `score-history.md` shows official metrics; judge column explicitly labeled non-official.
- Registry is append-only; rejected runs are recorded with reasons (negative results
  are evidence for the expert-review 30 points).

## 3. How things link (file contracts)

| Producer → Consumer | Artifact | Contract |
|---|---|---|
| Zakaria `convert_val.py` → everyone | `val_input.jsonl`, `val_gold.jsonl` | §1, frozen |
| Elly `run_inference.py` → Yahya | `outputs/submission.jsonl` | gold schema, 5 keys |
| Elly → Yahya (PROPOSAL) | `outputs/latency.jsonl` sidecar | `{id, latency_s, n_tokens}`; eval reads it, no re-runs to measure latency |
| Yahya `eval.py` → registry | `runs/<id>/metrics.json` + `score-history.md` row | §4 schema |
| Yahya `analyze.py` → team | `runs/<id>/report.html` | English; failure buckets with Chinese + translation side-by-side. Ends with a **HYPOTHESES** section (max 5, ranked): each = one-sentence finding + suspected cause + owner + suggested next probe ("compare 30 regression examples"). analyze.py turns metrics into actionable hypotheses; it must NOT become a dashboard of 15 graphs. |
| Yahya `judge.py` → analyze | `runs/<id>/judge.json` | dev-only; stratified sample |

Run manifest (`runs/<id>/manifest.json`) is the single source of truth:
model path + weights hash, prompt version, git sha, seed, temperature, hardware,
data file hashes, input hash, timestamp. Every layer reads it; nothing recomputes it.

Registry index: `runs/index.jsonl` (append-only), one line per run.
Run ID: `YYYYMMDD-<model>-<promptver>-<gitsha7>`.

## 4. metrics.json schema (deterministic only)

CANONICAL = what `scripts/eval.py` actually writes (verified against code,
2026-10-04). If design and code disagree, the code wins and this doc is updated.

```
{
  "submission": <path>, "gold": <path>, "n_scored": 2354,
  "emotion": {
    "n", "accuracy" (OFFICIAL), "macro_f1", "weighted_f1",
    "n_invalid_predictions",
    "per_class": { <each of 16>: {precision, recall, f1, support} },
    "confusion_top": { <gold class>: { <predicted label>: count } }
  },
  "profile": {
    "personality_traits": <field block>, "interests": <field block>,
    "style": <field block>,
    "micro_f1_avg", "macro_f1_avg", "exact_match_avg"
  },
  <field block> = {
    "n", "n_empty_gold", "empty_gold_rate",
    "micro_f1", "micro_precision", "micro_recall",
    "macro_f1", "exact_match",
    "per_label": { <vocab label>: {precision, recall, f1, support} }
  },
  "response": {
    "length_pred": {mean, median, min, max},
    "length_gold": {mean, median, min, max},
    "chinese_char_ratio_pred"   (mean of per-sample CJK ratios; length-unweighted)
  },
  "baselines": { "always_anxiety_accuracy", "empty_profile_exact_match_avg" }
}
```

Empty-vs-empty: counts as an exact match per field; contributes no TP/FP/FN to
micro-F1 (standard multi-label behavior) — verified by tests E01/E02.

DEFERRED to later layers (build order): `latency` (sidecar ingestion),
response-quality metrics (`bleu4`, `rouge_l`, `bertscore`, repetition,
english-drift counter), run identity fields (`run_id`, `created_utc`,
manifest ref), stored per-sample scores (paired diffs), bootstrap CIs.

**Metric tiers (anti "metric soup" rule):** every metric is labeled exactly one of
- **DECISION** — the only numbers a KEEP/REJECT discussion may cite: emotion accuracy
  (official), macro-F1 (our diagnostic-decision), profile micro-F1.
- **DIAGNOSTIC** — confusion pairs, per-label P/R, length/repetition/drift, latency,
  BLEU/ROUGE/BERTScore. They explain decisions; they never make them.
- **QUALITATIVE** — judge scores (dev-only). If a debate cites a diagnostic or
  qualitative number as a deciding argument, the debate is invalid by definition.

**Metric choices (locked):** profile scored per sub-field, micro-F1 primary +
macro-F1 reported; **macro-F1 averages only labels with gold support > 0**
(Oct 4 team review fix — zero-support labels made a perfect submission score
0.818 on style); empty-vs-empty counts as correct match;
BLEU/ROUGE character-level tokenization (Chinese standard); BERTScore with cached
`bert-base-chinese`, weights pre-downloaded (eval box is offline).
**Points source (Oct 4 team review item):** official competition rules page
(screenshot, WeChat-translated): 70 auto = 25 emotion Acc + 20 BLEU-4/ROUGE-L/
BERTScore composite + 15 profile avg + 10 latency; 30 expert review
(integrity / code prescriptiveness / platform records).

**Statistical layer (locked):** bootstrap CIs on all headline metrics; paired
per-sample comparison between runs. Run comparison outcomes are ternary —
**WIN / LOSS / INCONCLUSIVE** — never forced into binary KEEP/REJECT: an
inconclusive CI (e.g. delta +0.6, CI [-0.1, +1.3]) means "no evidence of change,"
not "reject." A CI over the val set measures *evaluation-sampling* uncertainty
only; it does NOT capture training-seed/model stochasticity. Practical rule:
prompt/decoding comparisons on a fixed model weight can trust the CI; comparisons
between separately trained models treat a non-significant CI as necessary-but-
not-sufficient evidence and prefer the safer/older artifact unless the delta
is clearly decisive.

## 5. Efficiency breakthroughs (ranked by real impact)

1. **Frozen-gold cache.** Gold never changes → precompute ONCE, reuse forever:
   BERTScore reference embeddings (the expensive half of every BERTScore call),
   gold tokenizations, gold English translations (never re-pay API for the same line),
   gold profile indicator matrices. Per-run cost drops to O(predictions) only.
2. **Content-addressed runs.** metrics keyed by sha256(submission) + gold version +
   metrics version. An already-scored submission is never rescored.
3. **Tiered evaluation.** Fast path = validate + emotion + profile + length stats
   (sklearn, milliseconds) for every iteration. Full path (+ BLEU/ROUGE/BERTScore,
   judge, report) only for candidate runs. Iteration friction ≈ 0.
4. **Stratified judge sampling.** Judge runs on a stratified ~200-sample subset
   (all rare classes oversampled) with CIs for routine iteration; full 2,354 only
   for final candidates. Cuts judge cost ~12×.
5. **Early-stopping paired bootstrap.** Sequential resampling stops when the sign
   is decisive (>99% agreement); spends resamples only on close calls.
6. **GPU-batched BERTScore.** Batch-encode on GPU; 2,354 short pairs in seconds.
7. **Latency sidecar** (proposal to Elly): inference writes per-sample latency;
   eval never re-runs the model to measure it. Cold-start vs steady-state separated
   (official scores the first 100 samples — warmup is a scored artifact).

## 6. Final recommendations

**Roadmap re-prioritized (Oct 4 team review, 16-day constraint):** DONE —
(1) validate.py + attack tests, (2) eval.py core. **NEXT: (3) analyze.py,
minimal version = per-sample mistakes CSV (id, gold, pred, field) so the team
can actually read errors; then (4) reply-quality judge (I4).** DEFERRED to
post-contest: run registry/manifests, content-addressed caching, bootstrap CIs,
BERTScore, HTML report. (BLEU/ROUGE were also deferred by team decision.)
Each shipped layer lands value the day it exists.
2. **Run the official baseline through the whole pipeline immediately** (B1 is done):
   the first end-to-end run sets the score floor and smoke-tests every layer.
3. **Never over-optimize BLEU** — one gold reference makes it high-variance;
   BERTScore is the primary response proxy; the judge is the empathy guardrail.
4. **Measure latency on the Migu instance once, early** (B7 dependency): local CPU
   numbers are meaningless for the 10 latency points.
5. **Judge discipline:** pinned model version, temperature 0, cache keyed by
   (reply_hash, judge_prompt_version); validate against human judgement on 20
   samples (per brief) before trusting any judge number.
6. **This document goes in `docs/eval-design.md`** — it is simultaneously the
   implementation spec and evidence for the expert-review 30 points.

**Known unknowns (tracked, not blocking):** official response-quality composite
weights; **profile scoring mode — if the official metric is exact-match rather
than F1, an all-empty profile prediction already scores 62.8% and cautious
empty predictions become a strong strategy (Oct 4 team review); this must go
into question #1 for the organisers**; final eval hardware spec. Mitigation:
mirror the announced components, document every assumption, confirm with
organizers when possible.

**Housekeeping (Oct 4 team review):** `eval-design1.md` near-duplicate —
keep `eval-design.md`, delete the other; `tests/run_tests.py` must be committed
and shared with reviewers (58 tests as of Oct 4).
