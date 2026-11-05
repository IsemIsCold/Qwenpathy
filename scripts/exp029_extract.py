#!/usr/bin/env python3
"""
scripts/exp029_extract.py -- EXP-029 step 1: cache the fine-tuned model's hidden states at every depth.

Offline study, no submission: does a learned read-out over *all* layers (AttnRes-style depth attention)
predict emotion / profile better than the label the model writes? This script only collects the features;
scripts/exp029_probe.py fits and scores the read-outs.

For every prediction point we build the prompt exactly like participant/run_inference.py + vLLM do
(system prompt, conversation_text(), Qwen3 chat template, enable_thinking=False, add_generation_prompt=True)
and run ONE forward pass of the merged model in bf16 with Hugging Face transformers (no lm_head, no generation).
Forward hooks keep, per point:
  last.f16   (N, 38, D)  state at the last prompt token (the token right before the model starts writing):
                         slot 0 = embeddings, slot k = output of decoder layer k (1..36), slot 37 = final norm
                         (= what lm_head reads; equals HF hidden_states[-1]).
  mean.f16   (N, M, D)   mean over the tokens of the LATEST user message, for the slots in --mean-slots.
  done.u8    (N,)        1 = row written (resume: rows already done are skipped).

Subcommands (the session script runs them in this order):
  plan   build the point list (seeded train sample + all dev + val), tokenize, create the empty arrays
  work   one worker per GPU: CUDA_VISIBLE_DEVICES=k ... work --rank k --world 2
  check  every row done, no NaN/inf, throughput summary -> extract_summary.json

    python exp029_extract.py plan --model /root/participant/model --out /root/exp029_feats \
        --sft-train /mnt/storage/sft/train_sharegpt.json --sft-dev /mnt/storage/sft/dev_sharegpt.json \
        --val-input /mnt/storage/val/t10_input.jsonl \
        --val-gold /mnt/storage/val/val_gold_n100.jsonl /mnt/storage/val/val_gold_tune500.jsonl
    CUDA_VISIBLE_DEVICES=0 python exp029_extract.py work --out /root/exp029_feats --rank 0 --world 2 &
    CUDA_VISIBLE_DEVICES=1 python exp029_extract.py work --out /root/exp029_feats --rank 1 --world 2 &
    wait; python exp029_extract.py check --out /root/exp029_feats

SFT files: either build_sft.py's *.jsonl ({"id", "messages"}) or *_sharegpt.json (id read from the
"样本 ID: ..." line). Labels: SFT points from the assistant target JSON, val points from the gold files.
Needs numpy + transformers + torch (work only). run_inference.py is imported for the prompt (never edited).
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ID_LINE = re.compile(r"^样本 ID: (\S+)")
TRAIN_CONV = re.compile(r"^(.*)_t\d+$")


def load_run_inference(path):
    spec = importlib.util.spec_from_file_location("exp029_ri", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def system_prompt(ri):
    return ri.SYSTEM_PROMPT.format(
        emotions=", ".join(sorted(ri.EMOTIONS)),
        personality=", ".join(sorted(ri.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(ri.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(ri.PROFILE_LABELS["style"])),
    )


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_sft(path):
    """[(id, system, user, assistant)] from build_sft.py jsonl or LLaMA-Factory sharegpt json."""
    path = Path(path)
    rows = []
    if path.suffix == ".jsonl":
        for r in read_jsonl(path):
            s, u, a = r["messages"]
            rows.append((r["id"], s["content"], u["content"], a["content"]))
    else:
        for r in json.loads(path.read_text(encoding="utf-8-sig")):
            u, a = r["conversations"]
            m = ID_LINE.match(u["value"])
            if not m:
                sys.exit(f"{path}: user message without a '样本 ID:' line")
            rows.append((m.group(1), r["system"], u["value"], a["value"]))
    return rows


def latest_user_span(prompt, user_text, offsets):
    """Token range [a, b) of the latest user message's text inside the rendered prompt."""
    marker = "\nuser: "  # conversation_text() renders "对话历史:\nuser: ...\nassistant: ...\nuser: <latest>"
    pos = user_text.rfind(marker)
    if pos < 0:
        return None
    content = user_text[pos + len(marker):]
    start_in_user = pos + len(marker)
    u0 = prompt.rfind(user_text)
    if u0 < 0:
        return None
    c0, c1 = u0 + start_in_user, u0 + start_in_user + len(content)
    toks = [i for i, (s, e) in enumerate(offsets) if e > c0 and s < c1]
    return (toks[0], toks[-1] + 1) if toks else None


# ------------------------------------------------------------------ plan
def cmd_plan(args):
    from transformers import AutoTokenizer

    ri = load_run_inference(args.run_inference)
    sp = system_prompt(ri)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, local_files_only=True)
    out = Path(args.out)
    if (out / "plan.json").exists() and not args.force:
        sys.exit(f"{out}/plan.json exists (resume with 'work', or pass --force to rebuild and lose the cache)")
    out.mkdir(parents=True, exist_ok=True)

    points = []  # dicts: id, split, conversation_id, system, user, emotion, profile
    rng = random.Random(args.seed)
    for split, path, n in (("train", args.sft_train, args.n_train), ("dev", args.sft_dev, args.n_dev)):
        if not path:
            continue
        rows = read_sft(path)
        if n and n < len(rows):
            rows = rng.sample(rows, n)
        bad_sp = 0
        for pid, s, u, a in rows:
            bad_sp += s != sp
            target = json.loads(a)
            points.append({
                "id": pid, "split": split, "conversation_id": TRAIN_CONV.match(pid).group(1),
                "system": s, "user": u,
                "emotion": target["emotion_label"], "profile": target["user_profile"],
            })
        if bad_sp:
            sys.exit(f"{path}: {bad_sp} system prompts differ from run_inference.py's (wrong file or old repo?)")
    if args.val_input:
        gold = {}
        for g in args.val_gold or []:
            for r in read_jsonl(g):
                gold[r["id"]] = r
        for s in read_jsonl(args.val_input):
            g = gold.get(s["id"])
            if g is None:
                sys.exit(f"no gold for val point {s['id']}")
            points.append({
                "id": s["id"], "split": "val", "conversation_id": s["conversation_id"],
                "system": sp, "user": ri.conversation_text(s),
                "emotion": g["emotion_label"], "profile": g["user_profile"],
            })
    ids = [p["id"] for p in points]
    if len(set(ids)) != len(ids):
        sys.exit("duplicate point ids")
    if not points:
        sys.exit("no points")

    all_tokens, offsets, meta = [], [0], []
    for p in points:
        messages = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids_ = enc["input_ids"]
        span = latest_user_span(prompt, p["user"], enc["offset_mapping"])
        if span is None:
            sys.exit(f"{p['id']}: latest user message not found in the prompt")
        all_tokens.extend(ids_)
        offsets.append(len(all_tokens))
        meta.append({
            "i": len(meta), "id": p["id"], "split": p["split"], "conversation_id": p["conversation_id"],
            "n_tokens": len(ids_), "user_span": list(span), "emotion": p["emotion"], "profile": p["profile"],
        })
    if args.max_tokens and max(m["n_tokens"] for m in meta) > args.max_tokens:
        sys.exit(f"a prompt has {max(m['n_tokens'] for m in meta)} tokens > --max-tokens {args.max_tokens}")

    cfg = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))
    n_layers, d = cfg["num_hidden_layers"], cfg["hidden_size"]
    n_slots = n_layers + 2
    mean_slots = [int(x) for x in args.mean_slots.split(",")] if args.mean_slots else list(range(0, n_layers + 1, 4))
    if any(not 0 <= s < n_slots for s in mean_slots):
        sys.exit(f"--mean-slots must be in 0..{n_slots - 1}")
    N = len(meta)
    np.save(out / "tokens.npy", np.asarray(all_tokens, dtype=np.int32))
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    np.lib.format.open_memmap(out / "last.npy", mode="w+", dtype=np.float16, shape=(N, n_slots, d)).flush()
    np.lib.format.open_memmap(out / "mean.npy", mode="w+", dtype=np.float16, shape=(N, len(mean_slots), d)).flush()
    np.lib.format.open_memmap(out / "done.npy", mode="w+", dtype=np.uint8, shape=(N,)).flush()
    template = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    plan = {
        "n_points": N, "splits": {s: sum(m["split"] == s for m in meta) for s in ("train", "dev", "val")},
        "n_slots": n_slots, "slot_names": ["embed"] + [f"layer{k}" for k in range(1, n_layers + 1)] + ["final_norm"],
        "mean_slots": mean_slots, "hidden_size": d, "num_hidden_layers": n_layers,
        "model": str(Path(args.model).resolve()), "seed": args.seed,
        "n_train": args.n_train, "n_dev": args.n_dev,
        "tokens_total": len(all_tokens),
        "tokens_p50": int(np.median([m["n_tokens"] for m in meta])),
        "tokens_max": max(m["n_tokens"] for m in meta),
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(sp.encode()).hexdigest(),
        "bytes_features": N * (n_slots + len(mean_slots)) * d * 2,
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"plan: {N} points {plan['splits']} | tokens {plan['tokens_total']:,} (p50 {plan['tokens_p50']}, "
          f"max {plan['tokens_max']}) | features {plan['bytes_features'] / 1e9:.2f} GB -> {out}")


# ------------------------------------------------------------------ work
def load_model(path, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": dtype, "local_files_only": True, "attn_implementation": "sdpa"}
    try:
        import accelerate  # noqa: F401  (lets weights go straight to the GPU, no 16 GB CPU copy)
        kwargs.update(device_map={"": device}, low_cpu_mem_usage=True)
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device)
    model.eval()
    return model


class Recorder:
    """Forward hooks that keep only the rows we need (last token, user-span mean) of each slot."""

    def __init__(self, base, n_slots, mean_slots):
        self.n_slots, self.mean_pos = n_slots, {s: j for j, s in enumerate(mean_slots)}
        self.last = [None] * n_slots
        self.mean = [None] * len(mean_slots)
        self.handles = [base.embed_tokens.register_forward_hook(self.hook(0))]
        for k, layer in enumerate(base.layers):
            self.handles.append(layer.register_forward_hook(self.hook(k + 1)))
        self.handles.append(base.norm.register_forward_hook(self.hook(n_slots - 1)))
        if len(self.handles) != n_slots:
            raise RuntimeError(f"model has {len(base.layers)} layers, plan expects {n_slots - 2}")

    def set_positions(self, last, span):
        self.pos, self.span = last, span

    def hook(self, slot):
        def fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output  # (1, T, D)
            self.last[slot] = h[0, self.pos].float()
            j = self.mean_pos.get(slot)
            if j is not None:
                a, b = self.span
                self.mean[j] = h[0, a:b].float().mean(0)
        return fn


def cmd_work(args):
    import torch

    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    meta = read_jsonl(out / "meta.jsonl")
    tokens, offsets = np.load(out / "tokens.npy"), np.load(out / "offsets.npy")
    last = np.load(out / "last.npy", mmap_mode="r+")
    mean = np.load(out / "mean.npy", mmap_mode="r+")
    done = np.load(out / "done.npy", mmap_mode="r+")
    mine = [m["i"] for m in meta if m["i"] % args.world == args.rank and not done[m["i"]]]
    if args.max_samples:
        mine = mine[: args.max_samples]
    if not mine:
        print(f"[rank {args.rank}] nothing to do")
        return
    longest = max(mine, key=lambda i: meta[i]["n_tokens"])  # OOM shows up in the first minute, not at 90%
    rest = [i for i in mine if i != longest]
    random.Random(args.rank).shuffle(rest)  # random order -> honest ETA
    order = [longest] + rest

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")  # the one GPU CUDA_VISIBLE_DEVICES leaves
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    t0 = time.time()
    model = load_model(plan["model"], device, dtype)
    base = model.model
    rec = Recorder(base, plan["n_slots"], plan["mean_slots"])
    print(f"[rank {args.rank}] model on {device} in {time.time() - t0:.0f} s; {len(order)} points, "
          f"{sum(meta[i]['n_tokens'] for i in order):,} tokens", flush=True)

    total_tok = sum(meta[i]["n_tokens"] for i in order)
    started, tok_done, overflow, n = time.time(), 0, 0, 0
    deadline = started + args.time_limit_min * 60 if args.time_limit_min else None
    with torch.inference_mode():
        for n, i in enumerate(order, 1):
            m = meta[i]
            ids = torch.from_numpy(tokens[offsets[i]:offsets[i + 1]].astype(np.int64))[None].to(device)
            rec.set_positions(m["n_tokens"] - 1, tuple(m["user_span"]))
            base(input_ids=ids, use_cache=False)
            L = torch.stack(rec.last)  # (S, D) fp32
            M = torch.stack(rec.mean)
            if not (torch.isfinite(L).all() and torch.isfinite(M).all()):
                raise RuntimeError(f"non-finite hidden state at {m['id']}")
            big = (L.abs() > 65000).sum().item() + (M.abs() > 65000).sum().item()
            overflow += big
            last[i] = L.clamp(-65000, 65000).cpu().numpy().astype(np.float16)
            mean[i] = M.clamp(-65000, 65000).cpu().numpy().astype(np.float16)
            done[i] = 1
            tok_done += m["n_tokens"]
            if n % 64 == 0 or n == len(order):
                last.flush(); mean.flush(); done.flush()
            if n in (1, 25, 200) or n % 500 == 0 or n == len(order):
                el = time.time() - started
                rate = tok_done / max(el, 1e-9)
                eta = (total_tok - tok_done) / max(rate, 1e-9)
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0
                print(f"[rank {args.rank}] {n}/{len(order)} points | {rate:,.0f} tok/s | elapsed {el / 60:.1f} min | "
                      f"ETA {eta / 60:.1f} min | peak mem {mem:.1f} GiB", flush=True)
            if deadline and time.time() > deadline:
                print(f"[rank {args.rank}] time limit reached after {n} points; rerun 'work' to resume", flush=True)
                break
    last.flush(); mean.flush(); done.flush()
    stats = {"rank": args.rank, "points": n, "tokens": tok_done, "seconds": round(time.time() - started, 1),
             "tok_per_s": round(tok_done / max(time.time() - started, 1e-9)), "fp16_overflow_values": overflow,
             "device": device, "torch": torch.__version__}
    (out / f"worker{args.rank}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[rank {args.rank}] done: {stats}", flush=True)


# ------------------------------------------------------------------ check
def cmd_check(args):
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    done = np.load(out / "done.npy")
    last = np.load(out / "last.npy", mmap_mode="r")
    mean = np.load(out / "mean.npy", mmap_mode="r")
    workers = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("worker*.json"))]
    rows = np.flatnonzero(done)
    sample = np.random.default_rng(0).choice(rows, size=min(200, len(rows)), replace=False) if len(rows) else []
    finite = all(np.isfinite(last[i]).all() and np.isfinite(mean[i]).all() for i in sample)
    zero_rows = sum(int(not np.any(last[i])) for i in sample)
    summary = {
        "done": int(done.sum()), "n_points": plan["n_points"], "complete": bool(done.all()),
        "sample_finite": bool(finite), "sample_zero_rows": zero_rows,
        "fp16_overflow_values": sum(w.get("fp16_overflow_values", 0) for w in workers),
        "worker_seconds": [w["seconds"] for w in workers], "tok_per_s": [w["tok_per_s"] for w in workers],
    }
    (out / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("extract check:", json.dumps(summary))
    ok = summary["complete"] and summary["sample_finite"] and summary["sample_zero_rows"] == 0
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", help="default: --model")
    p.add_argument("--out", required=True)
    p.add_argument("--run-inference", default=str(REPO / "participant/run_inference.py"))
    p.add_argument("--sft-train")
    p.add_argument("--sft-dev")
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-dev", type=int, default=0, help="0 = all dev points")
    p.add_argument("--val-input")
    p.add_argument("--val-gold", nargs="*")
    p.add_argument("--mean-slots", default="", help="comma list; default every 4th slot 0,4,..,36")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--force", action="store_true")
    w = sub.add_parser("work")
    w.add_argument("--out", required=True)
    w.add_argument("--rank", type=int, default=0)
    w.add_argument("--world", type=int, default=1)
    w.add_argument("--device", help="default cuda if available")
    w.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    w.add_argument("--max-samples", type=int, default=0, help="tests / dry runs")
    w.add_argument("--time-limit-min", type=float, default=75, help="stop cleanly after this many minutes")
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"plan": cmd_plan, "work": cmd_work, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
