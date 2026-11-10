#!/usr/bin/env python3
"""
train/make_v4_data.py -- V4 training data = T7's exact data (data/sft/train.jsonl, dev.jsonl: same split, same
truncation, same 14,942 points), with three changes:

  1. decide-first: the target JSON starts with emotion_label, then response_text, user_profile, memory_refs
     (the model commits to the emotion before it writes the reply = System One / System Two in one pass);
  2. rare-class weighting by repetition: one extra copy of every fear, mixed and shame point, and of a seeded
     half of the helplessness points (the classes #3/#4 most often turn into "anxiety" on the 600 val points);
  3. a 200-point dev slice (seeded, from the held-out dev conversations) so training logs a dev loss.

The prompt (system + user message) is unchanged and still comes from run_inference.py. stdlib only.

    python train/make_v4_data.py --sft data/sft --out data/sft_v4 [--check-sharegpt data/sft/train_sharegpt.json]
Outputs: train_v4_sharegpt.json, dev200_sharegpt.json, dataset_info.json (migu_train_v4, migu_dev200), v4_report.json
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ORDER = ("emotion_label", "response_text", "user_profile", "memory_refs")
DUPLICATE = {"fear": 1.0, "mixed": 1.0, "shame": 1.0, "helplessness": 0.5}  # share of points that get one extra copy
SEED = 42


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_sharegpt(system, user, assistant):
    return {"system": system, "conversations": [{"from": "human", "value": user}, {"from": "gpt", "value": assistant}]}


def reorder(target_text):
    obj = json.loads(target_text)
    if set(obj) != set(ORDER):
        raise ValueError(f"unexpected target keys {sorted(obj)}")
    return json.dumps({k: obj[k] for k in ORDER}, ensure_ascii=False)  # same separators as build_sft.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/sft")
    ap.add_argument("--out", default="data/sft_v4")
    ap.add_argument("--check-sharegpt", help="T7's train_sharegpt.json: verify train.jsonl is the same data")
    ap.add_argument("--dev-n", type=int, default=200)
    args = ap.parse_args()
    sft, out = Path(args.sft), Path(args.out)
    train, dev = read_jsonl(sft / "train.jsonl"), read_jsonl(sft / "dev.jsonl")

    if args.check_sharegpt:
        t7 = json.loads(Path(args.check_sharegpt).read_text(encoding="utf-8"))
        same = len(t7) == len(train) and all(
            t7[k] == to_sharegpt(*(m["content"] for m in r["messages"])) for k, r in enumerate(train))
        if not same:
            sys.exit(f"{sft}/train.jsonl is not the data T7 trained on ({args.check_sharegpt})")
        print(f"check: train.jsonl = T7's {args.check_sharegpt} ({len(train)} examples)")

    rng = random.Random(SEED)
    rows, extra = [], collections.Counter()
    for r in train:
        s, u, a = (m["content"] for m in r["messages"])
        row = to_sharegpt(s, u, reorder(a))
        rows.append(row)
        label = json.loads(a)["emotion_label"]
        share = DUPLICATE.get(label, 0.0)
        if share and (share >= 1.0 or rng.random() < share):
            rows.append(row)
            extra[label] += 1
    dev_rows = [to_sharegpt(*(m["content"] for m in r["messages"][:2]), reorder(r["messages"][2]["content"])) for r in dev]
    dev_pick = sorted(random.Random(SEED).sample(range(len(dev_rows)), min(args.dev_n, len(dev_rows))))
    dev200 = [dev_rows[k] for k in dev_pick]

    out.mkdir(parents=True, exist_ok=True)
    (out / "train_v4_sharegpt.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (out / "dev200_sharegpt.json").write_text(json.dumps(dev200, ensure_ascii=False), encoding="utf-8")
    fmt = {"formatting": "sharegpt", "columns": {"messages": "conversations", "system": "system"},
           "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}
    info = {"migu_train_v4": {"file_name": "train_v4_sharegpt.json", **fmt}, "migu_dev200": {"file_name": "dev200_sharegpt.json", **fmt}}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    labels = collections.Counter(json.loads(r["conversations"][1]["value"])["emotion_label"] for r in rows)
    report = {"source": str(sft), "field_order": ORDER, "train_points": len(train), "train_examples": len(rows),
              "extra_copies": dict(extra), "dev_examples": len(dev200), "seed": SEED,
              "emotion_share_before": {k: round(v / len(train), 4) for k, v in collections.Counter(
                  json.loads(r["messages"][2]["content"])["emotion_label"] for r in train).most_common()},
              "emotion_share_after": {k: round(v / len(rows), 4) for k, v in labels.most_common()}}
    (out / "v4_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"train {len(train)} points -> {len(rows)} examples (extra {dict(extra)}), dev {len(dev200)} -> {out}")
    print("anxiety share", report["emotion_share_before"]["anxiety"], "->", report["emotion_share_after"]["anxiety"])


if __name__ == "__main__":
    main()
