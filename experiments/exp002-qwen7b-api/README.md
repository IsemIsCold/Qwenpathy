# EXP-002 — Official baseline, Qwen2.5-7B-Instruct via API (SiliconFlow)

| | |
|---|---|
| **Date** | 2026-10-03 |
| **Author** | Ismail (captain) |
| **Goal** | Check whether 7B, with the official prompt unchanged, replies as the companion and fills every field. Decide the model for submission #1. |
| **Status** | ✅ 7B is good enough for submission #1 |

## Setup
| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, full precision, served by SiliconFlow (硅基流动, *guījī liúdòng*) |
| Backend | `openai_compatible` via environment variables; `run_inference.py` and config **unchanged** |
| Prompt | Official baseline prompt |
| Test data | `test_inference_data.jsonl` (3 samples) |
| Cost | 0 Migu card-hours |

```powershell
$env:EVALUATION_BACKEND="openai_compatible"
$env:EVALUATION_BASE_URL="https://api.siliconflow.cn/v1"
$env:EVALUATION_API_MODEL="Qwen/Qwen2.5-7B-Instruct"
$env:EVALUATION_API_KEY="sk-..."   # never commit
python run_inference.py ..\test_inference_data.jsonl ..\out_7b
```

> Development only. The submission must run offline inside the Migu image, so no API backend can go in the image.

Things that did **not** work from China (for reference): Hugging Face router → Together AI was blocked by Cloudflare (403); Featherless AI wasn't tried.

## Results
- Format: `submission.jsonl` **PASS**, `performance_report.json` **PASS**, parse failures **0 / 3**
- Latency ≈ 5 s/sample (API; not representative of the 4090s)

| Sample | Context (last user turn) | Reply (gist) | emotion | profile | Verdict |
|---|---|---|---|---|---|
| 000001_t5 | Anxious after an interview, afraid of repeating a past rejection; asks: review now or relax? | Validates the anxiety, suggests relaxing first (music, walk) and reviewing later | anxiety | sensitive / career_development | ✅ Good, but doesn't mention the past rejection |
| 000001_t7 | Agrees to a hot shower, reveals an old boss always nitpicked, asks for concrete steps right now | Shower → music or a few pages of a book → review later | relaxed | sensitive / **reading_writing** | ⚠️ Generic; ignores the boss disclosure; "reading" interest was taken from its own suggestion; emotion debatable (fear of rejection?) |
| 000002_t5 | User asked for **short replies** earlier; says the photography hobby has become a chore | One short sentence about perfectionism pressure and rest | anxiety | art_design, travel_outdoor | ✅ Respected the "keep it short" instruction from earlier turns (memory works) |

## vs EXP-001 (0.5B)
| | 0.5B | 7B |
|---|---|---|
| Speaks as the companion | ❌ 0/3 | ✅ 3/3 |
| emotion_label filled | 2/3 | 3/3 |
| user_profile filled | 0/3 | 3/3 (style always empty) |
| Uses earlier context | ❌ | Partly |

## Conclusions
1. **Qwen2.5-7B-Instruct + the official prompt = submission #1.** No prompt work is needed first.
2. Main weaknesses for later: (a) empathy is generic, and it doesn't weave in earlier personal disclosures (the "memory fusion" score); (b) `style` is always empty; (c) unclear emotion choice in ambiguous cases.
3. We can't measure emotion/profile accuracy on the test file because it has no labels. **We need the val-set scorer** (Yahya) before comparing prompts or fine-tunes.

## Next steps
- [ ] Start the Migu instance without a GPU: check `conda_env` and versions; download 7B to `/root/participant/model`
- [ ] One GPU run inside the image (≈0.3 card-hours) → save the image → **submission #1**
- [ ] Val scorer: emotion accuracy + macro-F1, profile F1, empty-field rate
- [ ] Then: prompt that explicitly references earlier user disclosures and fills `style`; compare on val via API (free)
