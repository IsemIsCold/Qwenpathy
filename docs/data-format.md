# Data Format Specification

> **Scope:** This document defines every JSONL format used in the project — raw organiser dumps, test inputs, gold labels, and training pairs.  
> **Maintainer:** Zakaria (dataset owner); §4, §6 and §7 updated by Ismail F.  
> **Last updated:** 2026-10-12. Every path and field below was checked against the current code (`scripts/convert_val.py`, `train/build_sft.py` `write_outputs()` / `to_sharegpt()`, `train/check_sft.py`) and the files it writes.

---

## 1. Raw organiser format (`train_public.jsonl`, `val_public.jsonl`)

One conversation per line.

| Field | Type | Description |
|-------|------|-------------|
| `conversation_id` | string | Unique ID, e.g. `train_conv_000001` |
| `turns` | array | Chronological chat messages |
| `prediction_points` | array | Every point where we must predict the assistant reply |

### 1.1 Turn object

```json
{
  "turn_id": 1,
  "role": "user",           // or "assistant"
  "content": "..."
}
```

### 1.2 Prediction-point object

```json
{
  "target_user_turn_id": 9,        // last user message we see
  "assistant_response_turn_id": 10, // the gold assistant reply
  "target": {
    "response_text": "...",
    "emotion_label": "anxiety",
    "user_profile": {
      "personality_traits": ["sensitive"],
      "interests": [],
      "style": ["rational"]
    },
    "memory_refs": ["mem_000001", "mem_000002"]
  }
}
```

**Key rule:** `history` for a prediction point **always ends at `target_user_turn_id`** (inclusive). The assistant reply we must generate is the turn at `assistant_response_turn_id`.

---

## 2. Test-format input (`val_input.jsonl`)

Produced by `scripts/convert_val.py`. One JSON object per line. This is **exactly** what the official evaluator feeds to `start.sh`.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique point ID: `{conversation_id}_t{target_user_turn_id}` |
| `conversation_id` | string | Source conversation |
| `target_user_turn_id` | int | Last observed user turn |
| `history` | array | Turns up to and including `target_user_turn_id`, each stripped to `{turn_id, role, content}` |

```json
{
  "id": "val_conv_000001_t1",
  "conversation_id": "val_conv_000001",
  "target_user_turn_id": 1,
  "history": [
    {"turn_id": 1, "role": "user", "content": "..."}
  ]
}
```

---

## 3. Gold labels (`val_gold.jsonl`)

Produced by `scripts/convert_val.py`. Used by `scripts/eval.py` to score submissions.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Same as test-format `id` |
| `response_text` | string | Gold assistant reply |
| `emotion_label` | string | One of 16 emotions |
| `user_profile` | object | `{personality_traits[], interests[], style[]}` |
| `memory_refs` | array | List of memory IDs (may be empty) |

```json
{
  "id": "val_conv_000001_t1",
  "response_text": "...",
  "emotion_label": "anxiety",
  "user_profile": {
    "personality_traits": ["sensitive"],
    "interests": [],
    "style": ["rational"]
  },
  "memory_refs": ["mem_000001", "mem_000002"]
}
```

---

## 4. SFT training data (`data/sft/`)

Produced by `train/build_sft.py` (`write_outputs()`). The system prompt and the user message are built with `SYSTEM_PROMPT` and `conversation_text()` imported from `participant/run_inference.py`, so training sees exactly what inference sends. Defaults: `--max-user-tokens 3500` (oldest turns dropped until the rendered history fits), `--max-reply-chars 350` (longer gold replies skipped), `--dev-ratio 0.05` split by conversation, `--seed 42`.

| File | Format |
|---|---|
| `train.jsonl`, `dev.jsonl` | `{"id", "messages": [system, user, assistant]}` |
| `train_sharegpt.json`, `dev_sharegpt.json` | LLaMA-Factory sharegpt: a JSON list of `{"system", "conversations": [{"from": "human"}, {"from": "gpt"}]}` |
| `train_text.jsonl`, `dev_text.jsonl` | `{"id", "prompt", "completion"}`, rendered with the real Qwen3 chat template. Only written with `--tokenizer` |
| `dataset_info.json` | LLaMA-Factory registration: `migu_train`, `migu_dev` (by build_sft.py) and `migu_pilot2k` (added by check_sft.py) |
| `pilot2k_sharegpt.json` | 2,000 training examples (seed 42) for the pilot run, written by `train/check_sft.py` |
| `build_report.json` | counts, truncation, skipped examples, emotion distribution |
| `check_report.json` | sign-off by `train/check_sft.py`: val leakage, target validity, prompt identity, exact token lengths |

`train.jsonl` / `dev.jsonl`:

```json
{"id": "train_conv_000001_t9",
 "messages": [
   {"role": "system", "content": "你正在参加数字情感陪伴比赛。…"},
   {"role": "user", "content": "样本 ID: train_conv_000001_t9\n对话历史:\nuser: …\nassistant: …\nuser: …"},
   {"role": "assistant", "content": "{\"response_text\": \"…\", \"emotion_label\": \"anxiety\", \"user_profile\": {…}, \"memory_refs\": []}"}
 ]}
```

`*_sharegpt.json` (one list element):

```json
{"system": "你正在参加数字情感陪伴比赛。…",
 "conversations": [
   {"from": "human", "value": "样本 ID: train_conv_000003_t9\n对话历史:\nuser: …"},
   {"from": "gpt", "value": "{\"response_text\": \"…\", …}"}
 ]}
```

`*_text.jsonl`: `prompt` ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n` (the empty think block vLLM adds with `enable_thinking=False`); `completion` is the JSON answer.

```json
{"id": "…", "prompt": "<|im_start|>system\n你正在参加…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "completion": "{\"response_text\": …}"}
```

`dataset_info.json` (one entry):

```json
"migu_train": {"file_name": "train_sharegpt.json", "formatting": "sharegpt",
               "columns": {"messages": "conversations", "system": "system"},
               "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}
```

---

## 5. Quick reference: emotion & profile vocabularies

### 5.1 Emotions (16)

```
joy, gratitude, relaxed, care, pride, neutral, surprise,
mixed, sadness, loneliness, anxiety, anger, fear, disgust, shame, helplessness
```

### 5.2 Profile labels (from `submission_schema.json`)

| Category | Count | Labels |
|----------|-------|--------|
| `personality_traits` | 10 | `sensitive`, `conservative`, `high_conscientiousness`, `assertive`, `agreeable`, `introverted`, `emotionally_stable`, `casual`, `open`, `extroverted` |
| `interests` | 12 | `career_development`, `pets`, `study_exam`, `sports_fitness`, `games`, `reading_writing`, `art_design`, `social`, `travel_outdoor`, `music`, `film_animation`, `programming_technology` |
| `style` | 11 | `rational`, `direct`, `detailed`, `brief`, `indirect`, `high_emotional_expression`, `colloquial`, `formal`, `humorous`, `emoji_user`, `low_emotional_expression` |

**Note:** `emoji_user` and `low_emotional_expression` appear in train but were not observed in val (very rare). Include them in the system prompt so the model knows they are valid.

---

## 6. Files and how to regenerate them

`data/` is gitignored **except the fixed eval slices** (n50, n50b, n100, tune500, t10), which are tracked so everyone scores on identical files. Never regenerate the slices; the commands below are for the record. **Checked again Nov 8:** every command reproduces the tracked file record for record. Files written on Windows end lines with CRLF and files written on Linux/Migu with LF, so compare records, not bytes (command under the table).

| File | In git? | How to make it |
|---|---|---|
| `data/train_public.jsonl`, `data/val_public.jsonl` | no | organiser dataset download |
| `data/val_input.jsonl`, `data/val_gold.jsonl` | no | `python scripts/convert_val.py --in data/val_public.jsonl --out data/val_input.jsonl --gold data/val_gold.jsonl` |
| `data/val_input_n50.jsonl`, `data/val_gold_n50.jsonl` | **yes** | `python scripts/convert_val.py --in data/val_public.jsonl --out data/val_input_n50.jsonl --gold data/val_gold_n50.jsonl --n 50` |
| `data/val_input_n50b.jsonl`, `data/val_gold_n50b.jsonl` | **yes** | `python scripts/convert_val.py --in data/val_public.jsonl --out data/val_input_n50b.jsonl --gold data/val_gold_n50b.jsonl --exclude data/val_gold_n50.jsonl --sample 50 --seed 42` |
| `data/val_input_n100.jsonl`, `data/val_gold_n100.jsonl` | **yes** | n50 followed by n50b (concatenate the two files in that order) |
| `data/val_input_tune500.jsonl`, `data/val_gold_tune500.jsonl` | **yes** | `python scripts/convert_val.py --in data/val_public.jsonl --out data/val_input_tune500.jsonl --gold data/val_gold_tune500.jsonl --exclude data/val_gold_n100.jsonl --sample 500 --seed 7` (EXP-022: picks the emotion bias; no point shared with n100) |
| `data/t9_input.jsonl` (Migu: `/mnt/storage/val/`) | no | = `data/val_input_n100.jsonl` (n50 then n50b; 100 points). Used by the EXP-020/021/023 check runs |
| `data/t10_input.jsonl` | **yes** | `cat data/val_input_n100.jsonl data/val_input_tune500.jsonl > data/t10_input.jsonl` (made on Migu, LF). **The 600 val points** (488 conversations) used by EXP-022 to EXP-029 and `scripts/eval_600.py`; gold = `val_gold_n100.jsonl` + `val_gold_tune500.jsonl` |
| EXP-029 probe splits (no file in `data/`) | no | `scripts/exp029_extract.py plan`: **train** = 6,000 points sampled from `data/sft/train` (`random.Random(29).sample`), **dev** = all 768 points of `data/sft/dev` (5% of train conversations, held out from fine-tuning by `build_sft.py --dev-ratio 0.05 --seed 42`), **val** = the 600 points of `t10_input.jsonl`. The ids go to `meta.jsonl` in the feature cache |
| `data/val_input_long5.jsonl` | no | `python scripts/make_long5.py` (the 5 longest val points outside n50/n50b) |
| `data/sft/*` | no | `python train/build_sft.py --train data/train_public.jsonl --inference participant/run_inference.py --out-dir data/sft --tokenizer models/qwen3-8b-tok`, then `python train/check_sft.py --tokenizer models/qwen3-8b-tok --cutoff-len 6144` |
| `docs/data-stats.md` | yes | `python scripts/compute_stats.py` (never edit by hand) |
| `models/qwen3-8b-tok/` | no | tokenizer files of Qwen3-8B (ModelScope `Qwen/Qwen3-8B`) |
| `participant/model/` | no | model weights; for the submission, the merged full QLoRA model (see `experiments/exp019-qlora-full`) |

Check that a regenerated slice matches the tracked one (ignores line endings and key order):

```bash
python -c "import json,sys; r=lambda p:[json.loads(l) for l in open(p,encoding='utf-8-sig') if l.strip()]; a,b=r(sys.argv[1]),r(sys.argv[2]); print('same' if a==b else 'DIFFERENT', len(a), len(b))" new.jsonl data/val_input_tune500.jsonl
```

---

## 7. Known data issues

- **Leaked reasoning in gold replies:** 7 train golds and 1 val gold (`val_conv_000331_t17`) contain the data generator's `<think>…</think>` reasoning before the reply. `build_sft.py` skips them (`skipped_think_leak` in `build_report.json`). The val one is in neither n50 nor n50b. Full list of ids: `docs/data-stats.md` §6.
- **memory_refs:** about 98% of gold points list memory ids, but the test set has no memory bank, so every submission outputs `[]` (as the official baseline does).
- **Label vocabulary:** gold emotions and profile tags all come from the official lists (0 out-of-vocabulary labels; see `docs/data-stats.md` §2).
