# team-work.md Gap Analysis (O6) -- Yahya -> Ismail F.

**Date:** 2026-11-08  
**File:** `docs/team-work.md` (last updated 2026-10-12)  
**Deadline:** Nov 8 (today)

---

## 1. Critical Updates Needed

### 1.1 Y5 is DONE

`docs/eval-600.md` has been generated from real data (600 points: n100 + tune500). Key findings:

| | #2 bf16 | #3 FP8+blend | #4 cand. |
|---|---|---|---|
| Emotion accuracy | 0.790 | 0.787 | 0.788 |
| Profile exact-match (scored) | 0.534 | **0.562** | **0.652** |
| Profile micro-F1 | 0.461 | **0.499** | 0.387 |
| Reply ROUGE-L | 0.294 | 0.294 | 0.294 |

**#3** is a clear improvement over #2: same emotion/replies, +0.028 exact-match, +0.038 micro-F1.  
**#4** has even higher exact-match (0.652) but lower micro-F1 (0.387). Its thresholds were fitted in-sample on the 600 points (5-fold CV: 0.647).

### 1.2 Yahya's section is out of date

| Current text | What actually exists (Nov 8) |
|---|---|
| "Pre-flight gate for pilot runs, with 10 tests -- `pilot_gate.py` (Oct 9, not merged) -- reviewed Oct 10: needs fixes" | `scripts/pilot_gate.py` v2.0 -- rebuilt Nov 8 with all 8 fixes. 16 checks, PASS on T9. **Ready to merge.** |
| "Test suite for validate + eval -- done: 73/73" | `tests/run_tests.py` now has **102 tests** (validate 32 + eval 41 + analyze v4.1 29). Plus 15 pilot_gate tests, 19 profile_blend tests, 42 emotion_bias tests. |
| Missing: `audit_labels.py` | `scripts/audit_labels.py` -- Data Label Auditor for the 17 error patterns. |
| Missing: `eval_600.py` | `scripts/eval_600.py` -- Y5 deliverable. **Ran successfully on real data.** Report in `docs/eval-600.md`. |
| Missing: `latency-table.md` | `docs/latency.md` -- Y5a deliverable. Latency data from all `performance_report.json` files. |

### 1.3 Ismail F.'s section is missing recent experiments

| Experiment | Status | Why it matters |
|---|---|---|
| EXP-022: anxiety bias + tag filter test | Done Oct 13. Bias rejected, filter stays on. | Decided against anxiety bias and tag filter off. |
| EXP-023: FP8 + anxiety bias as #3 | Done Oct 13. **NO-GO.** | Profile micro-F1 0.435 < 0.4461 bar. FP8 rejected as standalone. |
| EXP-025: profile blend (CPU classifier) | Code + tests done. Check run done. | #3 candidate: FP8 + profile blend. Evaluated on 600 pts. |
| EXP-027: lenient parse + CPU classifier alone | Built offline, tested on 600 pts. | #4 candidate: higher exact-match (0.652) but in-sample thresholds. |

Current text says: "FP8 variant tested (EXP-020), submitted only if the latency formula makes it worth the quality loss."  
**Should say:** "FP8 tested and rejected standalone (EXP-023 NO-GO). Profile blend (EXP-025) = #3. Lenient parse + CPU alone (EXP-027) = #4 candidate."

### 1.4 "Where everything stands" table is wrong

| Row | Current (Oct 12) | Actual (Nov 8) |
|---|---|---|
| Submissions | "2 of 5 used. #2 = full QLoRA. FP8 variant tested, submitted only if latency formula makes it worth the quality loss" | **2 of 5 used.** #2 = bf16 QLoRA. #3 = FP8 + profile blend (evaluated). #4 = lenient + CPU (candidate). 1-2 remaining. |
| Compute | "4.95 of 30 card-hours left (Oct 12)" | Update with EXP-022 (approx 0.8) + EXP-023 (approx 0.5) + EXP-025 + EXP-027 consumed. |

---

## 2. Decision Needed: #3 vs #4

| Factor | #3 FP8+blend | #4 lenient+CPU |
|---|---|---|
| Emotion accuracy | 0.787 | 0.788 |
| Profile exact-match (scored) | 0.562 | **0.652** |
| Profile micro-F1 | **0.499** | 0.387 |
| Reply quality | 0.294 | 0.294 |
| Latency | approx 1.16 s | approx 1.16 s |
| Thresholds | CV-honest | **In-sample** (fitted on 600 pts) |

**If the official metric is exact-match per-field mean:** #4 is +0.090 better = **+1.35 competition points** (0.090 * 15).  
**If the official metric is micro-F1:** #3 is +0.112 better = **+1.68 competition points** (0.112 * 15).  
**Risk:** #4's thresholds are in-sample. On the hidden test set, they may overfit.

**Recommendation:** Submit **#3** (safer, CV-honest). Use #4 only if the team is confident the official metric is exact-match and willing to accept the overfitting risk.

---

## 3. Missing Deliverables (not in team-work.md at all)

| Deliverable | File | Owner | Task |
|---|---|---|---|
| Eval 600 script + report | `scripts/eval_600.py`, `docs/eval-600.md` | Yahya | Y5 DONE |
| Latency table | `docs/latency.md` | Yahya | Y5a DONE |
| Pre-flight gate v2.0 | `scripts/pilot_gate.py` | Yahya | Ready |
| Data Label Auditor | `scripts/audit_labels.py` | Yahya | Ready |
| Profile blend tests | `tests/test_profile_blend.py` (19/19) | Ismail F. | EXP-025 |
| Emotion bias tests | `tests/test_emotion_bias.py` (42/42) | Ismail F. | EXP-023 |
| FP8 gate script | `scripts/exp023_gate.py` | Ismail F. | EXP-023 |
| Profile blend session | `scripts/exp025_session.sh` | Ismail F. | EXP-025 |
| Imbalance plan | `docs/p2-imbalance-plan.md` | Ismail F. | P2 |

---

## 4. Action Items for Ismail F.

1. **Update the date** to 2026-11-08.
2. **Update Yahya's section:**
   - Change pilot_gate.py status from "not merged, needs fixes" to "merged Nov 8, v2.0, 16 checks, PASS on T9"
   - Add `audit_labels.py`, `eval_600.py`, `latency.md`
   - Update test counts: 102+ tests (not 73)
3. **Update Ismail F.'s section:**
   - Add EXP-022, EXP-023, EXP-025, EXP-027
   - Mark FP8 standalone as rejected
   - Mark #3 and #4 as evaluated candidates
4. **Rewrite "Where everything stands":**
   - Submissions: 2 used, #3/#4 evaluated, 1-2 remaining
   - Compute: update card-hour ledger
   - Model: add "+ CPU profile classifier (EXP-025) + lenient parse (EXP-027)"
5. **DECIDE:** #3 or #4 for submission? (See section 2 above)

---

## 5. What Yahya still needs from the team

- Confirmation from Ismail F. whether he will update `team-work.md` or if Yahya should send a PR
- Decision on #3 vs #4 for the next submission

---

*Generated by Yahya, 2026-11-08. Y5 report from real 600-point data.*
