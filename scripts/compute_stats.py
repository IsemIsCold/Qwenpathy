#!/usr/bin/env python3
"""
scripts/compute_stats.py -- dataset statistics -> docs/data-stats.md

Every number in docs/data-stats.md is written by this script; never edit the
doc by hand. Token counts are exact: the real Qwen3-8B tokenizer on exactly the
text the model sees (SYSTEM_PROMPT and conversation_text() imported from
participant/run_inference.py, the submitted code). The output is deterministic:
running it twice leaves `git diff` clean.

Usage (repo root):
    python scripts/compute_stats.py
    python scripts/compute_stats.py --train data/train_public.jsonl --val data/val_public.jsonl \
        --tokenizer models/qwen3-8b-tok --out docs/data-stats.md

Needs: the `tokenizers` package (installed with transformers) and the tokenizer
folder models/qwen3-8b-tok (tokenizer.json). Exits with an error if either is missing.
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
]
PROFILE_KEYS = ["personality_traits", "interests", "style"]
THINK_RE = re.compile(r"</?think>")
THRESHOLDS = [2048, 3500, 4096, 8192]   # 3500 = build_sft.py --max-user-tokens


def parse_args():
    ap = argparse.ArgumentParser(description="Compute dataset statistics -> docs/data-stats.md")
    ap.add_argument("--train", default="data/train_public.jsonl")
    ap.add_argument("--val", default="data/val_public.jsonl")
    ap.add_argument("--tokenizer", default="models/qwen3-8b-tok",
                    help="folder containing tokenizer.json (Qwen3-8B)")
    ap.add_argument("--out", default="docs/data-stats.md")
    return ap.parse_args()


def load_run_inference():
    """Import SYSTEM_PROMPT and conversation_text() from the submitted code."""
    path = ROOT / "participant" / "run_inference.py"
    spec = importlib.util.spec_from_file_location("run_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tokenizer(folder):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("ERROR: the `tokenizers` package is missing (pip install tokenizers).")
    tok_file = Path(folder) / "tokenizer.json"
    if not tok_file.is_file():
        sys.exit(f"ERROR: {tok_file} not found. Put the Qwen3-8B tokenizer in {folder}/.")
    return Tokenizer.from_file(str(tok_file))


def count_tokens(tok, texts, batch=512):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(len(e.ids) for e in tok.encode_batch(texts[i:i + batch], add_special_tokens=False))
    return out


def analyze(path, ri, tok):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"ERROR: {path} not found.")
    s = {
        "conversations": 0, "points": 0, "turns": [], "points_per_conv": [],
        "emotions": Counter(), "all_empty": 0,
        "field_empty": Counter(), "labels": {k: Counter() for k in PROFILE_KEYS},
        "mem_points": 0, "mem_refs": [], "think_ids": [],
        "hist_chars": [], "reply_chars": [], "hist_texts": [], "replies": [],
    }
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            conv = json.loads(line)
            s["conversations"] += 1
            turns = conv.get("turns", [])
            s["turns"].append(len(turns))
            pps = conv.get("prediction_points", [])
            s["points_per_conv"].append(len(pps))
            for pp in pps:
                s["points"] += 1
                tid = pp["target_user_turn_id"]
                sid = f"{conv['conversation_id']}_t{tid}"
                tgt = pp["target"]
                s["emotions"][tgt.get("emotion_label")] += 1
                prof = tgt.get("user_profile") or {}
                empty_all = True
                for k in PROFILE_KEYS:
                    labs = prof.get(k) or []
                    if labs:
                        empty_all = False
                    else:
                        s["field_empty"][k] += 1
                    s["labels"][k].update(labs)
                s["all_empty"] += empty_all
                refs = tgt.get("memory_refs") or []
                s["mem_points"] += bool(refs)
                s["mem_refs"].append(len(refs))
                reply = tgt["response_text"]
                if THINK_RE.search(reply):
                    s["think_ids"].append(sid)
                sample = {"id": sid, "history": [
                    {"role": t["role"], "content": t["content"]} for t in turns if t["turn_id"] <= tid]}
                text = ri.conversation_text(sample)
                s["hist_texts"].append(text)
                s["hist_chars"].append(len(text))
                s["replies"].append(reply)
                s["reply_chars"].append(len(reply))
    s["hist_tok"] = count_tokens(tok, s.pop("hist_texts"))
    s["reply_tok"] = count_tokens(tok, s.pop("replies"))
    return s


def pct(values, p):
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] if f == c else v[f] * (c - k) + v[c] * (k - f)


def share(n, total):
    return f"{n / total * 100:.1f}%" if total else "0.0%"


def mean(v):
    return sum(v) / len(v) if v else 0.0


def build_md(tr, va, sys_tokens, tok_folder):
    L = []
    w = L.append
    w("# Dataset Statistics")
    w("")
    w("> Generated by `scripts/compute_stats.py`. Do not edit by hand: rerun the script.")
    w(f"> Token counts: Qwen3-8B tokenizer (`{tok_folder}`), `add_special_tokens=False`, on exactly the text the model sees "
      "(`conversation_text()` from `participant/run_inference.py`).")
    w(f"> System prompt (V1.8 `SYSTEM_PROMPT`, formatted with the label lists as at inference): **{sys_tokens} tokens**.")
    w("")
    splits = [("Train", tr), ("Val", va)]

    w("## 1. Overview")
    w("")
    w("| Split | Conversations | Prediction points | Points / conversation | Turns / conversation (mean / median / max) |")
    w("|---|--:|--:|--:|--:|")
    for n, s in splits:
        w(f"| {n} | {s['conversations']:,} | {s['points']:,} | {mean(s['points_per_conv']):.2f} | "
          f"{mean(s['turns']):.1f} / {pct(s['turns'], 50):.0f} / {max(s['turns'])} |")
    w("")

    w("## 2. Emotion distribution")
    w("")
    w("| Emotion | Train | Train % | Val | Val % |")
    w("|---|--:|--:|--:|--:|")
    for e in EMOTIONS:
        w(f"| {e} | {tr['emotions'][e]:,} | {share(tr['emotions'][e], tr['points'])} | "
          f"{va['emotions'][e]:,} | {share(va['emotions'][e], va['points'])} |")
    oov = {n: sum(c for e, c in s["emotions"].items() if e not in EMOTIONS) for n, s in splits}
    w(f"| *(outside the 16 labels)* | {oov['Train']:,} | {share(oov['Train'], tr['points'])} | "
      f"{oov['Val']:,} | {share(oov['Val'], va['points'])} |")
    w(f"| **Sum** | **{sum(tr['emotions'].values()):,}** | | **{sum(va['emotions'].values()):,}** | |")
    w("")

    w("## 3. User profile")
    w("")
    w("### 3.1 Empty rates")
    w("")
    w("| Split | All three fields empty | personality_traits empty | interests empty | style empty |")
    w("|---|--:|--:|--:|--:|")
    for n, s in splits:
        w(f"| {n} | {share(s['all_empty'], s['points'])} | " + " | ".join(
            share(s["field_empty"][k], s["points"]) for k in PROFILE_KEYS) + " |")
    w("")
    w("### 3.2 Label frequencies (% of prediction points)")
    for k in PROFILE_KEYS:
        w("")
        w(f"#### {k}")
        w("")
        w("| Label | Train | Train % | Val | Val % |")
        w("|---|--:|--:|--:|--:|")
        labels = sorted(set(tr["labels"][k]) | set(va["labels"][k]),
                        key=lambda x: (-tr["labels"][k][x], x))
        for lab in labels:
            w(f"| {lab} | {tr['labels'][k][lab]:,} | {share(tr['labels'][k][lab], tr['points'])} | "
              f"{va['labels'][k][lab]:,} | {share(va['labels'][k][lab], va['points'])} |")
    w("")

    w("## 4. memory_refs in the gold data")
    w("")
    w("| Split | Points with ≥ 1 memory id | Mean ids per point |")
    w("|---|--:|--:|")
    for n, s in splits:
        w(f"| {n} | {share(s['mem_points'], s['points'])} | {mean(s['mem_refs']):.2f} |")
    w("")
    w("The test set has no public memory bank, so our submission (like the official baseline) always outputs `[]`.")
    w("")

    w("## 5. Lengths")
    w("")
    w("### 5.1 History (the rendered `conversation_text()`, without the system prompt)")
    w("")
    w("| Split | Unit | Mean | P50 | P90 | P99 | Max |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for n, s in splits:
        for unit, key in (("chars", "hist_chars"), ("tokens", "hist_tok")):
            v = s[key]
            w(f"| {n} | {unit} | {mean(v):,.0f} | {pct(v, 50):,.0f} | {pct(v, 90):,.0f} | {pct(v, 99):,.0f} | {max(v):,} |")
    w("")
    w("### 5.2 Gold reply")
    w("")
    w("| Split | Unit | Mean | P50 | P90 | P99 | Max |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for n, s in splits:
        for unit, key in (("chars", "reply_chars"), ("tokens", "reply_tok")):
            v = s[key]
            w(f"| {n} | {unit} | {mean(v):,.0f} | {pct(v, 50):,.0f} | {pct(v, 90):,.0f} | {pct(v, 99):,.0f} | {max(v):,} |")
    w("")
    w("### 5.3 Characters per token")
    w("")
    w("| Split | History | Reply |")
    w("|---|--:|--:|")
    for n, s in splits:
        w(f"| {n} | {sum(s['hist_chars']) / sum(s['hist_tok']):.2f} | {sum(s['reply_chars']) / sum(s['reply_tok']):.2f} |")
    w("")
    w("### 5.4 Context-length thresholds (history tokens)")
    w("")
    w(f"Add {sys_tokens} tokens of system prompt plus the chat-template overhead for the full prompt. "
      "3500 is the history budget used by `train/build_sft.py` (`--max-user-tokens`). "
      "Inference runs with a 32,768-token context.")
    w("")
    w("| Split | " + " | ".join(f"> {t:,}" for t in THRESHOLDS) + " |")
    w("|---|" + "--:|" * len(THRESHOLDS))
    for n, s in splits:
        cells = [f"{sum(h > t for h in s['hist_tok']):,} ({share(sum(h > t for h in s['hist_tok']), s['points'])})"
                 for t in THRESHOLDS]
        w(f"| {n} | " + " | ".join(cells) + " |")
    w("")
    longest = {n: max(s["hist_tok"]) + sys_tokens for n, s in splits}
    w(f"Longest full prompt (history + system prompt): train {longest['Train']:,} tokens, val {longest['Val']:,} tokens.")
    w("")

    w("## 6. Data-quality notes")
    w("")
    w("Gold replies containing the generator's leaked `<think>` reasoning (skipped by `build_sft.py`):")
    w("")
    for n, s in splits:
        ids = ", ".join(f"`{i}`" for i in s["think_ids"]) or "none"
        w(f"- {n}: {len(s['think_ids'])}: {ids}")
    w("")
    return "\n".join(L)


def main():
    args = parse_args()
    ri = load_run_inference()
    tok = load_tokenizer(args.tokenizer)
    system_prompt = ri.SYSTEM_PROMPT.format(        # exactly as run_inference.main() builds it
        emotions=", ".join(sorted(ri.EMOTIONS)),
        personality=", ".join(sorted(ri.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(ri.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(ri.PROFILE_LABELS["style"])),
    )
    sys_tokens = len(tok.encode(system_prompt, add_special_tokens=False).ids)
    print(f"system prompt: {sys_tokens} tokens")
    tr = analyze(args.train, ri, tok)
    print(f"train: {tr['points']:,} points")
    va = analyze(args.val, ri, tok)
    print(f"val:   {va['points']:,} points")
    md = build_md(tr, va, sys_tokens, args.tokenizer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8", newline="\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
