#!/usr/bin/env python3
"""
scripts/emotion_logprobs.py -- collect raw answers + the emotion-label distribution.

Sends every sample to a RUNNING vLLM server with exactly the submission's prompt
and generation settings (imported from participant/run_inference.py and its config),
plus logprobs. Saves, per sample:
  - "content": the raw model answer (so any parser variant can be re-applied offline)
  - "label_top": the top-20 alternatives at the first token of the emotion label
    ({"token", "logprob"}), used by scripts/tune_emotion_bias.py

Usage (on Migu, server already started with the submission's vLLM args):
    python3 scripts/emotion_logprobs.py --input /mnt/storage/val/t10_input.jsonl \
        --out /mnt/storage/out/lp_bf16.jsonl [--base-url http://127.0.0.1:8000/v1]

--workers N sends N requests at once (default 8): vLLM batches them, so 600 points take
minutes instead of ~25 min of 2-GPU time. Batched greedy decoding can differ from the
one-at-a-time submission run in a few tokens; that does not matter for tuning the bias.
Output order = input order.

Latency of logprobs (needed only if the correction goes into the submission):
    ... --limit 20 --workers 1 --no-logprobs --out /tmp/a.jsonl   # plain
    ... --limit 20 --workers 1 --out /tmp/b.jsonl                 # with logprobs
and compare the printed "mean request time".

stdlib only (runs in conda_env next to run_inference.py).
"""

import argparse
import concurrent.futures
import importlib.util
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

LABEL_KEY = re.compile(r'"emotion_label"\s*:\s*"')


def load_ri(participant):
    path = Path(participant) / "run_inference.py"
    spec = importlib.util.spec_from_file_location("run_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def label_position(tokens, content):
    """Index of the token where the emotion label value starts, or None."""
    m = LABEL_KEY.search(content)
    if not m:
        return None
    start = m.end()
    offset = 0
    for i, t in enumerate(tokens):
        if offset == start:
            return i
        offset += len(t["token"])
        if offset > start:  # label start falls inside a token (e.g. '"an'): not a clean boundary
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--participant", default="/root/participant")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples (0 = all)")
    ap.add_argument("--no-logprobs", action="store_true",
                    help="plain request, as in the submission (with --workers 1: latency baseline)")
    args = ap.parse_args()

    ri = load_ri(args.participant)
    config = json.loads((Path(args.participant) / "configs" / "inference_config.json").read_text(encoding="utf-8"))
    generation = dict(config["generation"])
    system_prompt = ri.SYSTEM_PROMPT.format(  # exactly as run_inference.main()
        emotions=", ".join(sorted(ri.EMOTIONS)),
        personality=", ".join(sorted(ri.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(ri.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(ri.PROFILE_LABELS["style"])),
    )
    url = ri.api_endpoint(args.base_url)
    samples = ri.read_jsonl(Path(args.input))
    if args.limit:
        samples = samples[:args.limit]
    extra = {} if args.no_logprobs else {"logprobs": True, "top_logprobs": args.top}

    def one(s):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": ri.conversation_text(s)}]
        payload = {"model": "model", "messages": messages,  # = --served-model-name in the session scripts
                   **ri.api_generation_options(generation), **config.get("api_options", {}), **extra}
        req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read().decode("utf-8"))
        ms = (time.perf_counter() - t0) * 1000
        choice = resp["choices"][0]
        content = choice["message"]["content"] or ""
        tokens = (choice.get("logprobs") or {}).get("content") or []
        pos = label_position(tokens, content)
        label_top = None
        if pos is not None:
            t = tokens[pos]
            label_top = [{"token": a["token"], "logprob": a["logprob"]} for a in t.get("top_logprobs", [])]
            if not any(a["token"] == t["token"] for a in label_top):
                label_top.append({"token": t["token"], "logprob": t["logprob"]})
        return {"id": s["id"], "content": content, "label_top": label_top, "ms": round(ms, 1)}

    missing = 0
    total_ms = 0.0
    t_start = time.time()
    with open(args.out, "w", encoding="utf-8") as out, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(one, samples), 1):  # map keeps input order
            missing += rec["label_top"] is None
            total_ms += rec["ms"]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(samples):
                print(f"[{i}/{len(samples)}] {time.time() - t_start:.0f}s, label position missing: {missing}", flush=True)
    print(f"written: {args.out}")
    if samples:
        note = "" if args.workers == 1 else "  (concurrent: not a latency number; use --workers 1)"
        print(f"mean request time: {total_ms / len(samples):.0f} ms{note}")


if __name__ == "__main__":
    main()
