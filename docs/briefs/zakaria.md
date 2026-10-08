# Brief: Zakaria (dataset): Oct 8 update

> Replaces the Oct 1 brief (in git history). Your open tasks are **Z1–Z4**; the table rows are in [TASKS.md](../../TASKS.md) under *Dataset (Zakaria)*.
> **Order: Z1 (Oct 10) → Z4 (Oct 12) → Z2 + Z3 (Oct 13).** Z1 goes into submission #2. Z4 decides what data the second training run (P2) uses.

## How to use this file with Kimi

For each task, paste Kimi **the "Rules for every task" section and that task's whole section**, plus the files listed under *Give Kimi*. Don't paste only the task title: the details are what make the result correct.

## Rules for every task

1. **Only change the files listed under *Touch*.** If another file seems to need a change, ask in the group first.
2. **No new dependencies and no network.** Python standard library, plus `transformers` (already in `conda_env` and in your venv). The submission runs offline.
3. **Every number in a doc or PR comes from a command you ran.** Paste the command and its output in the PR description. No unmeasured claims ("100% format match", "<1 s runtime", "ready").
4. **No HTML reports, no PNGs, no zip files.** A chart is allowed only if the script that draws it is in the repo and runs.
5. **One task = one branch = one PR:** `zakaria/z1-truncation`, `zakaria/z2-stats`, `zakaria/z3-format`. Ismail F. reviews and merges. Don't send zips on WeChat.
6. **Run the tests listed in the task before opening the PR**, and paste their output.
7. Credit: `train/build_sft.py` was written by Ismail F. (I1) and Ismail M. (T1). Don't list it as your deliverable.
8. If Kimi's code disagrees with this brief, the brief wins. If the brief looks wrong, ask Ismail F.

**You need locally:** `data/train_public.jsonl`, `data/val_public.jsonl`, the val files (`data/val_input.jsonl`, `val_gold.jsonl`, `val_*_n50.jsonl`, `val_*_n50b.jsonl`), and the Qwen3-8B tokenizer folder `models/qwen3-8b-tok/` (tokenizer files only, about 10 MB). The tokenizer is gitignored, so ask Ismail F. for the zip.

---

## Z1 (= P4): truncate long histories at inference, same rule as training. Due **Oct 10**

### Why
The fine-tuned model only saw user messages of **≤ 3,500 tokens** in training. `train/build_sft.py` → `fit_history()` drops the oldest turns of longer histories. At inference, `participant/run_inference.py` still sends the **whole** history. On val, **24 of 2,354 points** are longer: the longest is ≈ 8,950 tokens, and none of them are in n50. (These counts come from a reconstructed Qwen tokenizer; your test with the real one decides.)

Goals:
1. The model gets the **same input shape** at inference as in training.
2. A safety net: no request can be too long, even if Elly lowers vLLM `--max-model-len` for latency (E4). Today it is 32,768, so nothing fails yet.

**Not a goal:** speeding up normal samples. Don't change anything for samples under the budget.

### Touch
- `participant/run_inference.py`: add code only, as described below.
- `tests/test_truncation.py`: new file.

### Do NOT touch
`SYSTEM_PROMPT`, `conversation_text()`, `extract_object()`, `normalize_prediction()`, `parse_prediction()`, `EMOTION_ALIASES`, `OVERPREDICTED_TAGS`, `participant/configs/inference_config.json`, `participant/start.sh`, `train/build_sft.py`, `scripts/*`.
**Don't add `max_user_tokens` to the config in this PR.** Ismail F. switches it on for the fine-tuned submission (T9). With the key absent, V1.8 must behave exactly as it does today.

### Give Kimi
`participant/run_inference.py`, `train/build_sft.py` (`fit_history()` and the `TokenCounter` class), `participant/configs/inference_config.json`, `participant/start.sh` (around line 70: how the model folder is chosen).

### Spec
1. **New function in `run_inference.py`:**
   ```python
   def fit_history(sample: dict, count_tokens, max_tokens: int) -> tuple[dict, int]:
       """Return (sample with the oldest turns dropped so conversation_text() fits, number of turns dropped)."""
   ```
   Use **exactly the algorithm** of `build_sft.py` `fit_history()`:
   - Render with `conversation_text()`.
   - If the history has 1 turn, or the token count is ≤ `max_tokens`, return.
   - Otherwise drop the oldest turn, and keep dropping while the new first turn is an assistant turn (always keep at least the last turn). Repeat.

   Copy the logic; **don't import build_sft.py**, because the submission must be self-contained. Return a new dict and never modify the input sample. Keep `id` unchanged.
2. **Token count = `len(tokenizer.encode(text, add_special_tokens=False))`**, the same as `TokenCounter.count()` in build_sft.py. No chars-per-token estimate.
3. **Config switch:** an optional key `max_user_tokens` (positive int) in `inference_config.json`, checked in `load_config()`.
   - **Missing or `null`:** no tokenizer is loaded, `fit_history()` is never called, and the output is byte-identical to today.
4. **Tokenizer (only when the key is set):**
   - `transformers` backend: reuse the tokenizer it already loads.
   - `openai_compatible` backend: `AutoTokenizer.from_pretrained(<dir>, local_files_only=True)`. Choose `<dir>` in the same order as start.sh: env `VLLM_MODEL_DIR`, else env `EVALUATION_MODEL_PATH`, else the config's `model_path` (relative paths are relative to `participant/`).
   - Load it **once at start-up**, before the first sample. If it fails, raise a clear error right away, never in the middle of the run.
5. **Apply it** in the main loop, where `messages` is built: build the user message from the shortened sample. Write the prediction under the original `id`.
6. **Report:** when the switch is on, add `"truncation": {"max_user_tokens": N, "truncated_samples": k, "dropped_turns": m}` to `performance_report.json`.
7. **Match the training settings.** Ask Ismail M. or Ismail F. for `data/sft/build_report.json` from the training data, and put its `max_user_tokens` and `token_counting` values in the PR. If `token_counting` says "estimate", **stop and tell Ismail F.** before going on: inference must count tokens the same way training did.

### Tests: `tests/test_truncation.py` (CPU only: no GPU, no vLLM, no API)
Run with: `python tests/test_truncation.py --tokenizer models/qwen3-8b-tok --val data/val_input.jsonl --train data/train_public.jsonl`. Load both `run_inference.py` and `build_sft.py` with `importlib`, the same way build_sft.py loads run_inference.

| # | Check | Pass |
|---|---|---|
| a | Every val point with a huge budget (10**9) | returned sample == input, 0 dropped |
| b | Every val point, budget 3500 | user message ≤ 3500 tokens **or** 1 turn left; history is a suffix of the original; first turn is `user`; last turn is unchanged (role `user`, `turn_id == target_user_turn_id`); `dropped == original length − new length` |
| c | **Train/inference match:** every train point (build test-format samples the way build_sft.py `make_sample()` does), budget 3500 | `conversation_text()` of our result == build_sft.py `fit_history()` text, for **100%** of points |
| d | Counts on val, n50, n50b | print the number truncated; expect ≈ 24 / 0 / small. If very different, explain why |
| e | Input not modified | each sample is deep-equal to a copy taken before the call |
| f | Overhead | print mean and max ms per sample for count + truncate over val (expect a few ms) |

The script exits non-zero if any check fails. Paste its full output in the PR.

### Done when
- The tests above pass, and their output is in the PR.
- The PR shows a diff of `run_inference.py` with **only additions** outside `load_config()` and the main loop.
- `build_report.json` values are quoted in the PR.
- No GPU work for you. Elly runs the GPU check in T9: n50 + n50b with `max_user_tokens: 3500`, where n50 must be identical to a run without it.

---

## Z2: make `compute_stats.py` produce `docs/data-stats.md` with real tokens. Due **Oct 13**

### Why (review of your B6 zips, Oct 8)
Your val numbers are correct: they match an independent recount exactly. But:
- The doc was edited by hand. Running the script brings back the old "vocabularies differ" warning, and it claims a Qwen2.5 tokenizer that never loaded.
- When a tokenizer does load, the real token counts are computed and **never printed**. The tables always use 1.4 chars/token; the real figure is ≈ 1.6 on val.
- The PNGs come from code that isn't in the repo. `viz_length_distributions.png` is blank, and the threshold chart's numbers differ from the table.
- The 4K-context / 500-token-system-prompt framing is stale: V1.8's prompt is 355 tokens, vLLM runs with 32k, and training uses a 6,144 cutoff.

### Touch
`scripts/compute_stats.py`, `docs/data-stats.md` (generated only, never hand-edited).

### Spec
1. **Tokenizer required.** Make `--tokenizer` default to `models/qwen3-8b-tok` and load it with `local_files_only=True`. If it can't load, exit with a clear error. Delete the chars/token estimate.
2. **Count what the model sees.** Import `conversation_text`, `SYSTEM_PROMPT`, `EMOTIONS` and `PROFILE_LABELS` from `participant/run_inference.py` (via importlib, like build_sft.py). History tokens = tokens of `conversation_text(sample)` with `add_special_tokens=False`. Also print the system prompt tokens, formatted the way `run_inference.py` `main()` formats it (expect ≈ 355).
3. **Thresholds** (train and val): count and % of points whose user message is over **2048 / 3500 / 4096 / 8192** tokens. 3500 is the training and inference budget.
4. **Add:**
   - empty rate **per profile field**
   - share of points with non-empty `memory_refs`
   - prediction points per conversation (p50/p90/max)
   - gold replies containing `<think>` or `</think>`, with their ids
   - an "out-of-vocab" row in the emotion table, plus a check that the counts add up to the number of prediction points
5. **Remove:** the vocabulary warning, the hardcoded date (use `datetime.date.today()`), and the 4K / 500-token note.
6. **Reproducible:** run it twice; the second run must leave `git diff --exit-code docs/data-stats.md` clean, apart from the date line.
7. Charts are optional: a `--plots` flag that draws from the same numbers into `docs/img/`. Otherwise no charts. Remove the old PNGs from the PR.

### Numbers your output must match (checked independently on val)
2,354 points / 900 conversations · all-empty profile 37.2% · empty personality 61.9%, interests 74.9%, style 51.7% · memory_refs non-empty 98.3% · 1 val `<think>` leak (`val_conv_000331_t17`) · 7 train leaks (same as build_sft.py) · val reply chars p50 98 / p90 336 / p99 847 / max 1,754 · val user message > 3,500 tokens ≈ 24 points. If yours differ, find out why before opening the PR.

---

## Z3: bring `docs/data-format.md` up to date. Due **Oct 13**

### Touch
`docs/data-format.md` only.

### Spec
1. **§4 SFT data:** replace the old `{prompt, completion}` pairs with what `train/build_sft.py` writes now, read from its `write_outputs()`:
   - `data/sft/{train,dev}.jsonl` (messages)
   - `{train,dev}_sharegpt.json` (`system` + `conversations` with `from: human/gpt`)
   - `{train,dev}_text.jsonl` (only with `--tokenizer`)
   - `dataset_info.json` (`migu_train`, `migu_dev`, `migu_pilot2k`)
   - `build_report.json`

   Show one short example of each. Mention `--max-user-tokens 3500` and `--max-reply-chars 350`.
2. **§6 File naming:** **nothing in `data/` goes into git** (it's gitignored). For each file, give the command that regenerates it. That includes the n50b command: `--exclude data/val_gold_n50.jsonl --sample 50 --seed 42`.
3. Use val ids in the val examples (`val_conv_000001_t1`, not `train_conv_…`).
4. Add a "Known data issues" note: 7 train golds and 1 val gold (`val_conv_000331_t17`) contain leaked `<think>` reasoning. build_sft.py skips them, and the val one is not in n50 or n50b.

**Done when:** every path and field in the doc exists in the current code. Say in the PR which function or line you checked each one against.

---

## Z4: research. Which long-reply filter and which augmentation for the 2nd training run (P2)? Due **Oct 12**

### Why
P2 = **one** more training run at most (needs ≥ 11 card-hours left). It is our only chance to change the training data, so we pick the recipe from measurements, not guesses. Your job is a short memo that ends in **one recommended recipe**.

What we have now:
- **Long-reply filter:** `build_sft.py --max-reply-chars 350` drops 2,940 train examples, leaving 14,942. Train reply median goes from 129 to ≈ 106 chars (val 98); anxiety goes from 34% to 32% (val 31.9%).
- **Rare emotions after the filter:** surprise 60, care 77, disgust 117, joy 162 train examples (out of 14,942).
- **Teacher:** DeepSeek V4.1-Flash (EXP-017) beats our model on every metric. It is the planned P2 data tool: rewrite long replies to < 150 chars, and generate rare-emotion examples, capped at 10–15% of train.
- **Open rule question:** are LLM-generated or external data allowed? Ismail F. is asking the organisers (O1). The memo must say what changes if the answer is no.

### Touch
- New `scripts/reply_length_study.py`: R1–R4, offline, CPU.
- New `scripts/check_synthetic.py`: the R6 filters, offline.
- New `experiments/exp018-data-research/`: `README.md` (the memo), `prompts/` (R6), and the output tables as `.md` or `.json`.
- One row in `LOG.md` → Experiments (EXP-018).

**No API calls by you, no GPU.** Don't edit `train/build_sft.py`: recommend changes, and Ismail M. implements them.

### Give Kimi
`train/build_sft.py`, `scripts/eval.py` (`bleu4_corpus()`, `rouge_l_macro()`, `length_stats()`), `docs/data-stats.md`, `experiments/exp017-teacher/README.md`, and this section.

### Part A: long replies (all offline)

**R1: what does each cutoff cost?**
- For cutoffs **150 / 200 / 250 / 300 / 350 / 400 / 500 / none** on train, give one row per cutoff with:
  - examples kept
  - count per emotion (all 16)
  - reply median and p90
  - **emotion-mix distance to val** = ½·Σ|p_train − p_val| (total variation)
  - **length distance to val:** the same formula on length buckets 0–50 / 50–100 / 100–150 / 150–250 / 250–350 / 350+
- Separately, list which emotions and profile labels lose the largest share at 350.

**R2: what are long replies like?**
- For each length bucket (≤ 150, 150–350, > 350), in train **and** val, give the share of replies with:
  - markdown `**` or `#`
  - numbered or bulleted lists (`1.` / `-` / `•` at line start)
  - two or more paragraphs (`\n\n`)
  - a question at the end (`？` or `?`)
- Paste 5 random replies over 350 chars (seed 42), shortened to 300 chars each.
- Hypothesis to confirm or reject: long golds are list-style advice, a different style from most val golds.

**R3: what does a short reply cost? (oracle test on val)**
- For each val gold, keep its first *k* chars, cut back to the last `。！？!?` before *k*, for *k* = 80 / 100 / 150 / 200 / 300 / full.
- Score that cut version against the **full** gold with eval.py's own `rouge_l_macro()` and `bleu4_corpus()` (import them; don't rewrite them).
- Report the mean per *k*, for all val and for n50 + n50b.
- This tells us which reply length the reference metrics reward if the content were perfect.

**R4: trim instead of drop?**
- Of the 2,940 dropped examples, how many can be cut at a paragraph or sentence boundary to ≤ 350 chars (and to ≤ 200) and still be a complete reply? A complete reply is ≥ 60 chars, ends with `。！？`, and doesn't end on a list header or the middle of a list.
- Show 10 before/after pairs.
- Compare three options in one table: **drop** (now), **trim** (R4), **LLM rewrite** (P2 plan). Columns: examples kept, rare-emotion counts, reply median vs val, risk.

### Part B: augmentation

**R5: methods review (one page, every claim with a link)**
- Compare ways to get more rare-emotion data for Chinese dialogue SFT:
  1. duplicate oversampling
  2. LLM paraphrase of existing rare-label examples (rewrite the last user turn, keep the label)
  3. LLM generation of new multi-turn conversations from an emotion + profile, with real train conversations as style examples
  4. public Chinese emotional or empathetic dialogue datasets
- For (4), check at least **CPED, SMILE (SmileChat), CPsyCoun** and **ESConv** (English). For each give: link, license, size, its emotion labels and whether they map onto our 16, and format.
- If Kimi can't give a working link for a claim, write **"unverified"**. Never present a guess as a fact.
- For each method, note its rules risk: generated or external data may be disallowed, and the expert review checks integrity.

**R6: concrete recipe and offline checks**
- Write the generation prompts in `prompts/`:
  - **rewrite:** shorten one long gold reply to ≤ 150 chars, same content, same tone, no markdown lists
  - **rare-emotion:** a new example with a fixed label, 2–3 real train conversations as style examples, output in exactly our target JSON
- `scripts/check_synthetic.py <synthetic.jsonl>` runs these checks offline and prints the pass rate per check:
  - **format:** same keys and label vocab as build_sft.py targets
  - **length:** reply ≤ 150 chars
  - **dedup:** no exact duplicates; char 5-gram Jaccard < 0.8 against each other and against train
  - **leak:** no 20-char substring shared with any **val** conversation
  - **label-consistency hook:** it reads an optional second label column (from a later teacher re-label) and reports agreement
- Test it on a fake file you make by hand: 5 good rows and 5 rows that each break one check.
- Rules for the recipe: synthetic rows go in **train only, never dev**. Synthetic data is at most 15% of train. Every synthetic row gets `"source": "synthetic"` in build_report so it stays traceable.
- The 40-call pilot (20 rewrites + 20 rare-emotion) is run **by Ismail F.** with his key. You then run `check_synthetic.py` on its output.

### The memo (`experiments/exp018-data-research/README.md`)
1. Goal and exact commands.
2. R1–R4 tables.
3. R5 table.
4. **Recommendation in 5 lines:**
   - cutoff, or trim, or rewrite
   - target count per rare emotion and how to get it
   - synthetic share
   - final train size and **train tokens**. Card-hours ≈ tokens / 18.7M × 3.7 (pilot speed on 2 GPUs).
   - the fallback recipe if the organisers say no generated or external data
5. What the recipe does **not** fix.

### Done when
- Both scripts run from the repo root with no network, and their commands and outputs are in the PR.
- Every number in the memo comes from those outputs.
- R5 links are checked, or marked unverified.
- The recommendation fits in 5 lines and includes a token or card-hour estimate.
