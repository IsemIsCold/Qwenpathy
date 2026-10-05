# analyze.py Architecture v2.0 — Specification (DRAFT for self-critique)

> Owner: Yahya (Evaluation). Target: replace monolithic analyze.py with a modular diagnosis engine.
> Principle: analyze READS scores, never computes them; eval.py is the single source of truth.

---

## 1. Why rewrite?

Current analyze.py (v1) has structural limits that block the team from using it as the "intellectual center" of iteration:

| Limitation | Impact |
|---|---|
| Hypotheses ranked by raw count, not score impact | A rare-class error (1 sample) can drag macro-F1 more than 50 common-class errors — but v1 buries it |
| No cross-field correlation | Can't detect systematic prompt failures ("when emotion is wrong, profile is ALSO wrong 80% of the time") |
| Hard-coded owner strings in source | Team structure changes → code changes → git churn |
| Markdown-only output | No programmatic access for CI/automation; no delta between runs |
| Generic fix suggestions | "prompt may not separate these two" — not actionable enough |
| No stratified analysis | Can't answer "how do we do on long-context conversations?" or "are we worse at night?" |
| Static severity thresholds | `floor = max(5, 0.01*n)` — ignores class support; 5 errors on surprise (support 10) is catastrophic, 5 on anxiety (support 750) is noise |

## 2. Core Architecture

```
submission + gold + input ──► Gate (precheck_or_die) ──► Analyzer Pipeline ──► Hypothesis Engine ──► Outputs
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────────────────┐
                    ▼                                     ▼                                     ▼
           [EmotionAnalyzer]                    [ProfileAnalyzer]                     [ResponseAnalyzer]
                    │                                     │                                     │
                    └─────────────────────────────────────┼─────────────────────────────────────┘
                                                          ▼
                                              [CorrelationAnalyzer]
                                                          │
                                                          ▼
                                              [ComparativeAnalyzer]  (optional: second submission)
                                                          │
                                                          ▼
                                              HypothesisEngine.rank()
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    ▼                                           ▼
                         report.md (human-readable)                 analysis.json (machine-readable)
```

### 2.1 Analyzer Contract

Every analyzer is a pure function:

```python
def analyze(context: AnalysisContext) -> Findings:
    """Returns findings. Never modifies context. Never prints."""
```

`AnalysisContext` carries: sub dict, gold dict, gold_order list, input dict (optional), eval metrics dict (optional), and a second submission dict (optional, for comparative mode).

`Findings` is a list of `Finding` objects:

```python
@dataclass
class Finding:
    category: str           # "emotion", "profile", "response", "correlation"
    subcategory: str        # e.g. "confusion", "rare_class", "length_drift"
    severity: float         # 0.0–1.0, computed by impact on OFFICIAL score
    count: int              # raw sample count
    description: str        # one-sentence human-readable
    impact: str             # e.g. "macro_f1_drag_estimated_0.08"
    example_ids: List[str]  # up to 5 specific ids, verified to exist
    fix_direction: str      # concrete next step, not generic advice
    confidence: str         # "high" (pattern in >100 samples), "medium" (30-100), "speculative" (<30)
```

### 2.2 Severity Formula (the key innovation)

Severity is NOT count. It is **estimated impact on the official score**:

**Emotion finding severity:**
```
severity = sum_over_errors( 1 / (n * support[gold_class]) )
```
This weights rare-class errors heavily (macro-F1 is unweighted; missing a class with support 10 is 100× worse than missing one with support 1000).

**Profile finding severity:**
```
severity = error_rate * field_weight
```
Field weight = 1/3 (equal, unless official scoring reveals otherwise).

**Response finding severity:**
```
severity = diagnostic_only  # no direct official-score formula; uses z-score of deviation
```

### 2.3 Hypothesis Engine

1. Collect all findings from all analyzers
2. **Deduplicate**: two analyzers flagging the same pattern → merge, boost confidence
3. **Cross-validate**: a finding with < 3 example_ids is downgraded to "speculative"
4. **Rank by severity** (descending), then confidence (high > medium > speculative)
5. **Cap at 7**: max 7 hypotheses; team can't act on more
6. **Group by owner category**: model_training, prompt_engineering, data_pipeline, decoding
   (NOT hard-coded names; configurable mapping from finding subcategory → owner category)

### 2.4 Output: Dual Format

**analysis.json** — machine-readable, versioned schema:
```json
{
  "schema_version": "2.0",
  "meta": { "submission", "gold", "n", "timestamp", "comparative": null|path },
  "summary": { "emotion_acc", "macro_f1", "profile_micro_f1", "n_hypotheses", "n_findings" },
  "findings": [ ... ],
  "hypotheses": [ ... ],
  "per_sample": { "id": { "flags": [...], "scores": {...} } }  // for programmatic drill-down
}
```

**report.md** — human-readable, expanded from v1:
- Executive summary (3 bullets: biggest win, biggest risk, one surprising finding)
- Emotion section with severity-ranked confusion pairs + rare-class callout
- Profile section with systematic bias detection ("over-predicts 'introverted' by 3×")
- Response section with length/CJK/BLEU/ROUGE distribution + outlier samples
- Cross-field section (NEW)
- Hypotheses (ranked, with example IDs and fix directions)
- Appendix: methodology notes

---

## 3. Analyzer Details

### 3.1 EmotionAnalyzer

Inputs: emotion_correct per sample, gold/pred labels, eval metrics.
Outputs:
- Confusion pairs (severity-weighted by 1/support[gold])
- Rare-class failures (classes with support < 50 and recall < 0.3)
- Majority-class collapse detector (accuracy >> macro_f1)
- Per-class accuracy breakdown

### 3.2 ProfileAnalyzer

Inputs: per-field gold/pred/exact/missing/extra from build_row.
Outputs:
- Per-field exact-match rate
- Systematic bias: labels predicted > 2× their gold rate (over-confidence)
- Systematic omission: labels in gold > 2× their prediction rate (under-confidence)
- Empty-pattern analysis: when gold is empty, does pred predict empty? (empty baseline)
- Cross-field correlation: e.g. "introverted" always paired with "low_emotional_expression"

### 3.3 ResponseAnalyzer

Inputs: response_text, gold_text, pred_len, gold_len, CJK ratios.
Outputs:
- Length distribution mismatch (KS-style: are pred lengths from a different distribution?)
- CJK drift severity (percentage of samples below threshold, not just count)
- BLEU/ROUGE/BERTScore per emotion class (are we worse at some emotions?)
- Response quality by gold length (are we worse on long-context replies?)
- Repetition detector (identical responses across > 5 samples → degenerate mode)

### 3.4 CorrelationAnalyzer (NEW)

Inputs: all per-sample flags.
Outputs:
- P(emotion wrong | profile wrong) vs baseline P(emotion wrong)
- P(LANG_DRIFT | emotion = rare_class) vs baseline
- P(LONG_PRED | gold_len > 200) vs baseline
- Clustering: groups of samples with identical multi-field failure patterns

### 3.5 ComparativeAnalyzer (NEW, optional)

Inputs: two submissions against same gold.
Outputs:
- Per-sample delta (which samples improved/regressed)
- Field-level delta (emotion ↑ but profile ↓?)
- Statistical significance of delta (binomial test on paired outcomes)

---

## 4. Efficiency Targets

| Metric | Target | Current v1 | How |
|---|---|---|---|
| Runtime | < 3s for 2,354 samples | ~1s | Pure stdlib, no model inference |
| Memory | < 50MB | ~20MB | Streaming per-sample rows, no full matrix |
| Output size | report < 50KB, json < 500KB | report ~10KB | Capped examples (5 per finding) |
| Extensibility | new analyzer = 1 function + 1 test | monolithic | Plugin pattern |

---

## 5. Configuration (not hard-coded)

```python
OWNER_MAP = {
    "emotion_confusion": "prompt_engineering",
    "emotion_rare_class": "model_training",
    "profile_bias": "prompt_engineering",
    "response_drift": "model_training",
    "length_outlier": "decoding",
    "repetition": "decoding",
}
```

Loaded from `config/owner_map.json` if exists; falls back to default. Team changes owner assignments without touching source.

---

## 6. Schema Lock

`analysis.json` schema versioned. Backward-compatible additions only (new fields okay, removing/renaming not okay without major version bump).
