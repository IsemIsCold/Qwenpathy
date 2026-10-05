# EXP-016: V0 vs V1.7 on n50b (independent slice)

| | |
|---|---|
| **Date** | 2026-10-05 |
| **Author** | Ismail F. |
| **Goal** | Confirm V1.7 on 50 val points nobody tuned on. |
| **Part of** | [EXP-005 prompt series](../exp005-prompt/README.md) (version table + all prompts there) |
| **Status** | Done. V1.7 confirmed. V1.8 = V1.7 + emotion alias map is the frozen prompt (now participant/ on main). |

## Setup

`Qwen/Qwen3-8B:nscale` via the Hugging Face router, non-thinking, greedy (temperature 0), val n50b. Prompt = `participant_v1_x/run_inference.py` (only `SYSTEM_PROMPT` / `normalize_prediction` differ from `participant/`; the exact text is in the EXP-005 README).

```powershell
python scripts\exp3_compare.py --participant participant `
  --input data\val_input_n50b.jsonl --gold data\val_gold_n50b.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v0_n50b --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

```powershell
python scripts\exp3_compare.py --participant participant_v1_7 `
  --input data\val_input_n50b.jsonl --gold data\val_gold_n50b.jsonl --scripts scripts `
  --out experiments\exp005-prompt\v1_7_n50b --modes nothink --greedy --soft-switch `
  --key-env HF_TOKEN --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-8B:nscale"
```

Note: when this ran (Oct 5), `participant/` was still **V0**. It is V1.8 now; to re-run V0 use the V0 file from git: `git show e5209d2:participant/run_inference.py`.

Run outputs (submission, raw outputs, performance report): `experiments/exp005-prompt/v0_n50b/`, `experiments/exp005-prompt/v1_7_n50b/`

## Results

n50b = 50 random val points (seed 42) from conversations not used in n50 (`convert_val.py --exclude data/val_gold_n50.jsonl --sample 50 --seed 42`). Nothing was tuned on it.

| Metric | V0 | V1.7 |
|---|---|---|
| Emotion accuracy | 0.60 | **0.62** |
| Emotion macro-F1 | 0.366 | **0.374** |
| Profile micro-F1 (avg) | 0.175 | **0.300** |
| Profile exact match (avg) | 0.107 | **0.247** (all-empty: 0.593) |
| — personality / interests / style micro-F1 | 0.14 / 0.24 / 0.15 | 0.20 / 0.43 / 0.27 |
| BLEU-4 | 0.068 | 0.068 |
| ROUGE-L | **0.205** | 0.200 |
| Reply median (chars) | 108 | 99 (gold 101) |
| Invalid emotion labels (blank) | 3 | 3 |

**V1.7 confirmed.** Emotion equal (31 vs 30 correct), the profile gain holds on unseen data (+12.5 micro-F1, +14 exact match), replies equal. n50b is easier than n50 for both (V0 0.60 vs 0.50): always compare versions on the same slice.

Both versions lose 3/50 samples to invented emotion labels (`frustration`, `annoyance`, `hopefulness`) that the parser turns into a blank = always wrong. Fix = **V1.8** (`participant_v1_8/`) = V1.7 + `EMOTION_ALIASES` in `normalize_prediction`, which maps common inventions to the closest official label, otherwise falls back to `anxiety`. Re-parsing V0's n50b raw outputs with it: 30 → 31 correct. It cannot make a valid label worse.

Open: an all-empty profile still has the highest exact match (0.593 on n50b). If the official profile metric is exact match, empty profiles win; if F1, V1.7 wins. The organisers' answer decides.

## Files

- `metrics_v0.json`: copy of `experiments/exp005-prompt/v0_n50b/out_nothink/metrics.json`
- `metrics_v1_7.json`: copy of `experiments/exp005-prompt/v1_7_n50b/out_nothink/metrics.json`
