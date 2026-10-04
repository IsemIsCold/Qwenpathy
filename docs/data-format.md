# Data Format Specification

> **Scope:** This document defines every JSONL format used in the project — raw organiser dumps, test inputs, gold labels, and training pairs.  
> **Maintainer:** Zakaria (dataset owner)  
> **Last updated:** 2026-10-04

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
  "id": "train_conv_000001_t9",
  "conversation_id": "train_conv_000001",
  "target_user_turn_id": 9,
  "history": [
    {"turn_id": 1, "role": "user", "content": "..."},
    {"turn_id": 2, "role": "assistant", "content": "..."},
    ...
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
  "id": "train_conv_000001_t9",
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

## 4. SFT training pairs (`train/sft_pairs.jsonl`)

Produced by `train/build_sft.py`. Each line is a `{prompt, completion}` pair ready for LLaMA-Factory or direct SFT.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Same point ID |
| `prompt` | string | Full prompt text (system prompt + rendered history) |
| `completion` | string | JSON string of the target output |

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

## 6. File naming conventions

| File | Purpose | In git? |
|------|---------|---------|
| `data/train_public.jsonl` | Official training data | **No** (too large) |
| `data/val_public.jsonl` | Official validation data | **No** |
| `data/val_input.jsonl` | Test-format validation inputs | Yes |
| `data/val_gold.jsonl` | Gold labels for validation | Yes |
| `data/sft_pairs.jsonl` | SFT training pairs | Yes |
| `participant/model/` | Model weights | **No** (git-lfs or download) |
