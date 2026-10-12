#!/usr/bin/env python3
"""
tests/test_truncation.py -- Z1: history truncation at inference == training.

CPU only (no GPU, no vLLM, no API). Checks participant/run_inference.py
fit_history() against train/build_sft.py fit_history() on every train point,
plus the invariants on val.

Run (repo root, venv with transformers):
    python tests/test_truncation.py --tokenizer models/qwen3-8b-tok \
        --val data/val_input.jsonl --train data/train_public.jsonl

Exit code 0 = all checks pass.
"""

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET = 3500  # = build_sft.py --max-user-tokens (data/sft/build_report.json)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="models/qwen3-8b-tok")
    ap.add_argument("--val", default="data/val_input.jsonl")
    ap.add_argument("--train", default="data/train_public.jsonl")
    ap.add_argument("--n50", default="data/val_input_n50.jsonl")
    ap.add_argument("--n50b", default="data/val_input_n50b.jsonl")
    args = ap.parse_args()

    for p in (args.tokenizer, args.val, args.train):
        if not Path(p).exists():
            sys.exit(f"missing: {p}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    count = lambda text: len(tok.encode(text, add_special_tokens=False))

    ri = load_module("run_inference", ROOT / "participant" / "run_inference.py")
    sft = load_module("build_sft", ROOT / "train" / "build_sft.py")
    counter = sft.TokenCounter(args.tokenizer)  # the exact counter training used

    val = read_jsonl(args.val)
    print(f"val points: {len(val):,}")

    # (a) huge budget: nothing changes
    print("\n(a) budget 10**9 on every val point")
    ok_a = True
    for s in val:
        out, dropped = ri.fit_history(s, count, 10**9)
        if dropped != 0 or out != s:
            ok_a = False
            break
    check("a: huge budget returns the sample unchanged, 0 dropped", ok_a)

    # (b) invariants at the training budget, (e) input not modified, (f) overhead
    print(f"\n(b, e, f) budget {BUDGET} on every val point")
    ok_b = ok_e = True
    first_bad = ""
    truncated = 0
    times = []
    for s in val:
        before = copy.deepcopy(s)
        t0 = time.perf_counter()
        out, dropped = ri.fit_history(s, count, BUDGET)
        times.append((time.perf_counter() - t0) * 1000)
        if s != before:
            ok_e = False
        h, o = s["history"], out["history"]
        n_tok = count(ri.conversation_text(out))
        good = (
            (n_tok <= BUDGET or len(o) == 1)
            and o == h[len(h) - len(o):]
            and o[0]["role"] == "user"
            and o[-1] == h[-1] and o[-1]["role"] == "user"
            and o[-1]["turn_id"] == s["target_user_turn_id"]
            and dropped == len(h) - len(o)
            and out["id"] == s["id"]
        )
        if not good and ok_b:
            ok_b, first_bad = False, s["id"]
        truncated += dropped > 0
    check("b: fits (or 1 turn), suffix, starts on user, last turn kept, dropped count right, id kept", ok_b, first_bad)
    check("e: input sample never modified", ok_e)
    print(f"  truncated on val: {truncated} / {len(val)}")
    print(f"  (f) overhead per sample: mean {sum(times) / len(times):.2f} ms, max {max(times):.2f} ms")

    # (c) train/inference match on every train point
    print(f"\n(c) train/inference match, budget {BUDGET}, every train point")
    train = read_jsonl(args.train)
    n = same = 0
    first_diff = ""
    for conv in train:
        for pp in conv["prediction_points"]:
            sample = sft.make_sample(conv, pp["target_user_turn_id"])
            sft_text, _ = sft.fit_history(sample, ri, counter, BUDGET)
            ours, _ = ri.fit_history(sample, count, BUDGET)
            n += 1
            if ri.conversation_text(ours) == sft_text:
                same += 1
            elif not first_diff:
                first_diff = sample["id"]
    check("c: identical text to build_sft.py fit_history() on 100% of train points",
          same == n, f"{same:,}/{n:,}" + (f", first diff {first_diff}" if first_diff else ""))

    # (d) counts on the eval slices
    print("\n(d) truncated points per slice")
    for name, path in (("val", args.val), ("n50", args.n50), ("n50b", args.n50b)):
        if not Path(path).exists():
            print(f"  {name}: file missing, skipped")
            continue
        rows = read_jsonl(path)
        k = sum(ri.fit_history(s, count, BUDGET)[1] > 0 for s in rows)
        print(f"  {name}: {k} / {len(rows)}")
    check("d: n50 has no truncated point (so its scores can't change)",
          not Path(args.n50).exists() or sum(ri.fit_history(s, count, BUDGET)[1] > 0 for s in read_jsonl(args.n50)) == 0)

    failed = [name for name, ok in results if not ok]
    print("\n" + ("ALL TESTS PASSED" if not failed else "FAILURES: " + "; ".join(failed)) + f"  ({len(results) - len(failed)}/{len(results)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
