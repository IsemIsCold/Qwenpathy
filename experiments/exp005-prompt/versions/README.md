# Prompt versions (EXP-005 series)

Every `run_inference.py` we tested, frozen as it was run. Only `SYSTEM_PROMPT` (and, from V1.7, `normalize_prediction`) differs from the official file. The `participant_v1_*` working folders are gitignored, so these copies are the record.

| File | Version | Tested in |
|---|---|---|
| `run_inference_v0.py` | official prompt (submission #1; also commit `e5209d2`) | EXP-005, EXP-016 |
| `run_inference_v1.py` | emotion-first + base-rate line + 10 definitions | EXP-005 |
| `run_inference_v1_2.py` | V1 without the base-rate line | not run |
| `run_inference_v1_3.py` | V0 + EXP-009 profile + length rules | EXP-010 |
| `run_inference_v1_4.py` | V0 + length rule | EXP-011 |
| `run_inference_v1_5.py` | V1.4 + emotion-first order | EXP-012 |
| `run_inference_v1_6.py` | Ismail M.'s 2nd prompt + 3 fixes | EXP-014 |
| `run_inference_v1_7.py` | V0 + "Leave [] if insufficient" + tag filter | EXP-015, EXP-016 |
| `run_inference_v1_8.py` | V1.7 + emotion alias map — **frozen, = `participant/run_inference.py` on main** | — |

Compare two versions: `git diff --no-index experiments/exp005-prompt/versions/run_inference_v0.py experiments/exp005-prompt/versions/run_inference_v1_8.py`
