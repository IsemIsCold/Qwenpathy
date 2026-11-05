#!/usr/bin/env python3
"""
tests/test_exp029.py -- EXP-029 (depth read-out) offline tests. CPU only, ~5-10 min.

    python tests/test_exp029.py [--sft-train data/sft/train.jsonl] [--tokenizer models/qwen3-8b-tok]

Needs numpy, torch, transformers (any CPU build), the Qwen3-8B tokenizer files, the val slices and
outputs/exp023/lp_fp8_t10.jsonl. Exit code 0 = all checks pass.

A. scripts/exp029_extract.py on a tiny RANDOM Qwen3 (4 layers, hidden 64, the real tokenizer):
   prompts = the exact vLLM chat-template tokens of run_inference.py's messages; the user span decodes to the
   latest user message; hook features = HF output_hidden_states (+ final norm); user-span means; 2 workers cover
   every row once; an interrupted run resumes to the same features; 'plan' refuses to overwrite a cache.
B. scripts/exp029_probe.py on SYNTHETIC features with a planted signal (emotion at slot 6 of 10, profile at
   slot 3): the probes find those slots, depth attention puts its weight there, McNemar/ECE helpers are right.
C. Probe end to end on the tiny-model features (real labels, real FP8 answers file) + exp029_session.sh run
   twice (fresh, then resume) with GPU checks off.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXTRACT = REPO / "scripts/exp029_extract.py"
PROBE = REPO / "scripts/exp029_probe.py"
SESSION = REPO / "scripts/exp029_session.sh"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def run(cmd, **kw):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false", **kw.pop("env", {})}
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, **kw)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tiny_model(path, tokenizer):
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=40960,
                      rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    with torch.no_grad():  # make depth matter: larger random residual writes
        for p in model.parameters():
            p.mul_(3.0)
    model.to(torch.bfloat16).save_pretrained(path)
    AutoTokenizer.from_pretrained(tokenizer).save_pretrained(path)


def small_inputs(tmp, sft_train):
    """24 train (jsonl), 8 dev (sharegpt json), 12 val points (incl. multi-turn and the longest)."""
    rows = []
    with open(sft_train, encoding="utf-8-sig") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= 400:
                break
    random.Random(0).shuffle(rows)
    short = [r for r in rows if len(r["messages"][1]["content"]) < 1500]
    with open(tmp / "train.jsonl", "w", encoding="utf-8") as f:
        for r in short[:24]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dev = [{"system": r["messages"][0]["content"],
            "conversations": [{"from": "human", "value": r["messages"][1]["content"]},
                              {"from": "gpt", "value": r["messages"][2]["content"]}]} for r in short[24:32]]
    (tmp / "dev_sharegpt.json").write_text(json.dumps(dev, ensure_ascii=False), encoding="utf-8")
    val = [json.loads(l) for l in open(REPO / "data/t10_input.jsonl", encoding="utf-8-sig")]
    pick = [s for s in val if len(s["history"]) == 1][:4] + [s for s in val if 3 <= len(s["history"]) <= 9][:8]
    with open(tmp / "val_input.jsonl", "w", encoding="utf-8") as f:
        for s in pick:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return [s["id"] for s in pick]


def plan_args(tmp, out, tokenizer):
    return [sys.executable, EXTRACT, "plan", "--model", tmp / "tiny", "--tokenizer", tokenizer, "--out", out,
            "--sft-train", tmp / "train.jsonl", "--sft-dev", tmp / "dev_sharegpt.json", "--n-train", 0,
            "--val-input", tmp / "val_input.jsonl", "--val-gold", REPO / "data/val_gold_n100.jsonl",
            REPO / "data/val_gold_tune500.jsonl", "--mean-slots", "0,2,4"]


def part_a(tmp, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ri = load("ri_t", REPO / "participant/run_inference.py")
    ex = load("ex_t", EXTRACT)
    feats = tmp / "feats"
    r = run(plan_args(tmp, feats, tokenizer))
    check("A1 plan runs", r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "")
    if r.returncode:
        print(r.stderr[-3000:])
        return None
    plan = json.loads((feats / "plan.json").read_text(encoding="utf-8"))
    meta = [json.loads(l) for l in open(feats / "meta.jsonl", encoding="utf-8")]
    check("A2 splits 24/8/12, 6 slots, mean slots 0,2,4", plan["splits"] == {"train": 24, "dev": 8, "val": 12}
          and plan["n_slots"] == 6 and plan["mean_slots"] == [0, 2, 4], json.dumps(plan["splits"]))

    tok = AutoTokenizer.from_pretrained(tokenizer)
    tokens, offsets = np.load(feats / "tokens.npy"), np.load(feats / "offsets.npy")
    sp = ex.system_prompt(ri)
    val = {json.loads(l)["id"]: json.loads(l) for l in open(tmp / "val_input.jsonl", encoding="utf-8")}
    train = {json.loads(l)["id"]: json.loads(l) for l in open(tmp / "train.jsonl", encoding="utf-8")}
    same, span_ok = 0, 0
    for m in meta:
        if m["split"] == "val":
            msgs = [{"role": "system", "content": sp}, {"role": "user", "content": ri.conversation_text(val[m["id"]])}]
            latest = val[m["id"]]["history"][-1]["content"]
        elif m["split"] == "train":
            msgs = train[m["id"]]["messages"][:2]
            latest = msgs[1]["content"].rsplit("\nuser: ", 1)[1]
        else:
            continue
        ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if isinstance(ref, dict):
            ref = ref["input_ids"]
        same += list(tokens[offsets[m["i"]]:offsets[m["i"] + 1]]) == list(ref)
        a, b = m["user_span"]
        span_ok += tok.decode(tokens[offsets[m["i"]] + a: offsets[m["i"]] + b]).strip() == latest.strip()
    n_chk = sum(m["split"] != "dev" for m in meta)
    check("A3 prompt tokens = chat template (enable_thinking=False) of run_inference.py messages", same == n_chk, f"{same}/{n_chk}")
    check("A4 user span decodes to the latest user message", span_ok == n_chk, f"{span_ok}/{n_chk}")
    last_tok = [tok.decode([tokens[offsets[m['i'] + 1] - 1]]) for m in meta]
    check("A5 last prompt token is the '\\n\\n' after </think>", all(t == "\n\n" for t in last_tok), repr(set(last_tok)))
    dev_ids = [m["id"] for m in meta if m["split"] == "dev"]
    check("A6 dev ids read from the sharegpt '样本 ID' line", all(i.startswith("train_conv_") for i in dev_ids), dev_ids[0])

    # two workers
    procs = [subprocess.Popen([sys.executable, EXTRACT, "work", "--out", feats, "--rank", str(k), "--world", "2"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}) for k in range(2)]
    outs = [p.communicate()[0] for p in procs]
    check("A7 two workers exit 0", all(p.returncode == 0 for p in procs), " | ".join(o.strip().splitlines()[-1][:100] for o in outs))
    r = run([sys.executable, EXTRACT, "check", "--out", feats])
    check("A8 check: complete, finite", r.returncode == 0, r.stdout.strip()[-160:])
    w = [json.loads((feats / f"worker{k}.json").read_text()) for k in range(2)]
    check("A9 workers split the points (22 + 22 = 44)", sorted(x["points"] for x in w) == [22, 22], str([x["points"] for x in w]))

    # hooks vs HF output_hidden_states
    model = AutoModelForCausalLM.from_pretrained(tmp / "tiny", torch_dtype=torch.bfloat16).eval()
    last = np.load(feats / "last.npy")
    mean = np.load(feats / "mean.npy")
    worst = worst_m = worst_n = 0.0
    for m in meta[::7]:
        ids = torch.from_numpy(tokens[offsets[m["i"]]:offsets[m["i"] + 1]].astype(np.int64))[None]
        with torch.no_grad():
            hs = model.model(input_ids=ids, output_hidden_states=True).hidden_states
        ref = [h[0, -1].float().numpy() for h in hs]  # emb, L1..L3, norm(L4)
        got = last[m["i"]].astype(np.float32)
        for j in range(4):
            worst = max(worst, np.abs(got[j] - ref[j]).max() / (np.abs(ref[j]).max() + 1e-6))
        worst = max(worst, np.abs(got[5] - ref[4]).max() / (np.abs(ref[4]).max() + 1e-6))
        with torch.no_grad():
            normed = model.model.norm(torch.from_numpy(got[4])[None].to(torch.bfloat16)).float().numpy()[0]
        worst_n = max(worst_n, np.abs(normed - got[5]).max() / (np.abs(got[5]).max() + 1e-6))
        a, b = m["user_span"]
        for jj, s in enumerate([0, 2, 4]):
            src = hs[s][0, a:b].float().numpy().mean(0) if s < 4 else None
            if src is not None:
                worst_m = max(worst_m, np.abs(mean[m["i"], jj].astype(np.float32) - src).max() / (np.abs(src).max() + 1e-6))
    check("A10 hook states = HF hidden_states (embed, layers, final norm)", worst < 1e-2, f"max rel err {worst:.2e}")
    check("A11 slot 5 = final norm of slot 4", worst_n < 2e-2, f"max rel err {worst_n:.2e}")
    check("A12 user-span means = mean of HF hidden_states over the span", worst_m < 1e-2, f"max rel err {worst_m:.2e}")
    check("A13 depth matters in the tiny model (slots differ)", np.abs(last[:, 1] - last[:, 3]).mean() > 0.05)

    # resume: interrupted single worker -> same features
    feats2 = tmp / "feats2"
    run(plan_args(tmp, feats2, tokenizer))
    r1 = run([sys.executable, EXTRACT, "work", "--out", feats2, "--max-samples", "5"])
    d1 = int(np.load(feats2 / "done.npy").sum())
    r2 = run([sys.executable, EXTRACT, "work", "--out", feats2])
    last2 = np.load(feats2 / "last.npy")
    check("A14 interrupted run (5 rows) resumes to identical features", r1.returncode == 0 and r2.returncode == 0 and d1 == 5
          and np.array_equal(last2, last) and np.array_equal(np.load(feats2 / "mean.npy"), mean), f"rows after stop {d1}")
    r = run(plan_args(tmp, feats2, tokenizer))
    check("A15 plan refuses to overwrite an existing cache", r.returncode != 0 and "exists" in (r.stdout + r.stderr))
    return feats


def make_synthetic(path, ri, n=(1500, 400, 300), S=10, D=32, seed=0):
    """Emotion signal planted at slot 6 (peak), profile signal at slot 3; the 'gen' answers are ~55% right."""
    rng = np.random.default_rng(seed)
    labels = sorted(ri.EMOTIONS)
    use = ["anxiety", "relaxed", "helplessness", "fear", "sadness", "anger"]
    prior = np.array([0.4, 0.2, 0.12, 0.1, 0.1, 0.08])
    fields = list(ri.PROFILE_LABELS)
    tags = [(f, t) for f in fields for t in sorted(ri.PROFILE_LABELS[f])]
    centers = rng.normal(size=(len(use), D)) * 1.0
    tag_dirs = rng.normal(size=(len(tags), D))
    depth_dirs = rng.normal(size=(S, D)) * 3.0
    depth_gain = np.exp(-0.5 * ((np.arange(S) - 6) / 0.8) ** 2)  # emotion lives around slot 6
    prof_gain = np.exp(-0.5 * ((np.arange(S) - 3) / 0.6) ** 2)
    N = sum(n)
    path.mkdir(parents=True, exist_ok=True)
    last = np.zeros((N, S, D), dtype=np.float16)
    mean = np.zeros((N, 3, D), dtype=np.float16)
    meta = []
    splits = ["train"] * n[0] + ["dev"] * n[1] + ["val"] * n[2]
    val_ids = []
    for i, sp in enumerate(splits):
        c = rng.choice(len(use), p=prior)
        active = [j for j, (f, t) in enumerate(tags) if f == "interests" and rng.random() < 0.12]
        x = rng.normal(size=(S, D)) * 2.0  # noisy enough that no single read-out is perfect
        x += depth_gain[:, None] * centers[c] * 1.5
        for j in active:
            x += prof_gain[:, None] * tag_dirs[j] * 1.2
        x *= (1 + np.arange(S))[:, None] * 0.5  # growing norm with depth (like a real residual stream)
        x += depth_dirs * (1 + np.arange(S))[:, None]  # each depth has its own mean direction (real streams do)
        last[i] = x.astype(np.float16)
        mean[i] = x[[0, 4, 8]].astype(np.float16)
        pid = f"{'val' if sp == 'val' else 'train'}_conv_{i:06d}_t1"
        profile = {f: [t for j, (ff, t) in enumerate(tags) if ff == f and j in active] for f in fields}
        meta.append({"i": i, "id": pid, "split": sp, "conversation_id": pid[:-3], "n_tokens": 10,
                     "user_span": [2, 5], "emotion": use[c], "profile": profile})
        if sp == "val":
            val_ids.append((pid, use[c]))
    np.save(path / "last.npy", last)
    np.save(path / "mean.npy", mean)
    np.save(path / "done.npy", np.ones(N, dtype=np.uint8))
    with open(path / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    plan = {"n_points": N, "n_slots": S, "slot_names": ["embed"] + [f"layer{k}" for k in range(1, S - 1)] + ["final_norm"],
            "mean_slots": [0, 4, 8], "hidden_size": D, "num_hidden_layers": S - 2}
    (path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    first = {v: k for k, v in ri.LABEL_FIRST_TOKEN.items()}
    with open(path / "answers.jsonl", "w", encoding="utf-8") as f:
        for pid, gold in val_ids:
            lab = gold if rng.random() < 0.55 else "anxiety"
            top = [{"token": first[lab], "logprob": -0.2}, {"token": first["relaxed" if lab != "relaxed" else "fear"], "logprob": -2.0}]
            content = json.dumps({"response_text": "好的", "emotion_label": lab,
                                  "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": []}, ensure_ascii=False)
            f.write(json.dumps({"id": pid, "content": content, "label_top": str(top), "ms": "1"}, ensure_ascii=False) + "\n")
    return path


def part_b(tmp):
    ri = load("ri_b", REPO / "participant/run_inference.py")
    pr = load("pr_b", PROBE)
    m = pr.mcnemar_exact(np.array([True] * 10 + [False] * 2 + [True] * 5), np.array([False] * 10 + [True] * 2 + [True] * 5))
    check("B1 McNemar exact (10 vs 2) p = 0.0386", abs(m["p"] - 2 * (1 + 12 + 66) / 4096) < 1e-12 and m["a_only"] == 10, f"{m}")
    probs = np.array([[0.8, 0.2]] * 10)
    y = np.array([0] * 8 + [1] * 2)
    check("B2 ECE = 0 when confidence = accuracy", abs(pr.ece(probs, y)) < 1e-12)
    check("B3 ECE = 0.3 when 80% confident and 50% right", abs(pr.ece(probs, np.array([0] * 5 + [1] * 5)) - 0.3) < 1e-12)

    feats = make_synthetic(tmp / "synth", ri)
    out = tmp / "synth_out"
    r = run([sys.executable, PROBE, "--feats", feats, "--llm-answers", feats / "answers.jsonl", "--out", out,
             "--device", "cpu", "--epochs", "20", "--seeds", "2", "--profile-baseline", "0.5"])
    check("B4 probe runs on synthetic features", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        return
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    e = res["emotion"]
    check("B5 best single layer = planted slot (layer5..layer7)", e["best_layer"]["slot"] in ("layer5", "layer6", "layer7"), e["best_layer"]["slot"])
    curve = {c["name"]: c["dev_acc"] for c in res["layer_curve"] if c["kind"] == "last"}
    check("B6 layer curve peaks in the middle, not at embed/final", curve[e["best_layer"]["slot"]] > max(curve["embed"], curve["final_norm"]) + 0.05,
          " ".join(f"{k}:{v:.2f}" for k, v in curve.items()))
    alpha = [r_ for r_ in csv_rows(out / "alpha.csv") if r_["readout"] == "depth_attn"]
    top = max(alpha, key=lambda r_: float(r_["alpha_mean_val"]))["slot"]
    check("B7 depth attention puts its largest weight near the planted slot", top in ("layer5", "layer6", "layer7"), top)
    check("B8 depth attention alpha entropy below uniform", e["depth_attn"]["alpha_entropy_val"] < e["depth_attn"]["alpha_entropy_uniform"] - 0.05,
          f"{e['depth_attn']['alpha_entropy_val']:.3f} vs {e['depth_attn']['alpha_entropy_uniform']:.3f}")
    check("B9 gen accuracy read from the answers file (~0.55 + anxiety share)", 0.55 < e["gen"]["val_acc"] < 0.85, f"{e['gen']['val_acc']:.3f}")
    p = e[res["primary"]]
    check("B10 primary beats gen with McNemar p < 0.05", p["val_acc"] > e["gen"]["val_acc"] and p["mcnemar_vs_gen"]["p"] < 0.05,
          f"{res['primary']} {p['val_acc']:.3f} p {p['mcnemar_vs_gen']['p']:.2g}")
    check("B11 primary chosen from dev accuracies by the pre-registered rule", res["primary"] == expected_primary(e), res["primary"])
    check("B12 temperature scaling does not worsen ECE much", p["val_ece_temp"] <= p["val_ece_raw"] + 0.03,
          f"{p['val_ece_raw']:.3f} -> {p['val_ece_temp']:.3f}")
    pf = res["profile"]
    check("B13 profile probe picks the planted slot (layer2..layer4)", pf["best"]["slot"] in ("layer2", "layer3", "layer4"), pf["best"]["slot"])
    check("B14 profile exact beats all-empty on interests", pf["val_field_exact"]["interests"] > pf["val_all_empty_field_exact"]["interests"] + 0.05,
          f"{pf['val_field_exact']['interests']:.3f} vs {pf['val_all_empty_field_exact']['interests']:.3f}")
    rows = [json.loads(l) for l in open(out / "val_predictions.jsonl", encoding="utf-8")]
    check("B15 val_predictions: 300 rows with every read-out", len(rows) == 300 and all(
        k in rows[0] for k in ("gold", "gen", "final", "best_layer", "user_mean", "scalar_mix", "depth_attn", "blend", "profile")))
    check("B16 report.md written", "Primary read-out" in (out / "report.md").read_text(encoding="utf-8"))


def expected_primary(e):
    pool = ["final", "best_layer", "user_mean", "scalar_mix"]
    best = max(pool, key=lambda r: e[r]["dev_acc"])
    return "depth_attn" if e["depth_attn"]["dev_acc"] >= e[best]["dev_acc"] + 0.005 else best


def csv_rows(path):
    import csv
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def part_c(tmp, feats, tokenizer):
    out = tmp / "tiny_out"
    r = run([sys.executable, PROBE, "--feats", feats, "--out", out, "--device", "cpu", "--epochs", "4", "--seeds", "1", "--max-iter", "20"])
    check("C1 probe end to end on tiny-model features + real FP8 answers file", r.returncode == 0 and (out / "results.json").exists(),
          r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-800:])
    if r.returncode == 0:
        res = json.loads((out / "results.json").read_text(encoding="utf-8"))
        ri = load("ri_c", REPO / "participant/run_inference.py")
        val_ids = [json.loads(l)["id"] for l in open(tmp / "val_input.jsonl", encoding="utf-8")]
        gold = {}
        for p in ("data/val_gold_n100.jsonl", "data/val_gold_tune500.jsonl"):
            for l in open(REPO / p, encoding="utf-8-sig"):
                g = json.loads(l)
                gold[g["id"]] = g["emotion_label"]
        ans = {json.loads(l)["id"]: json.loads(l) for l in open(REPO / "outputs/exp023/lp_fp8_t10.jsonl", encoding="utf-8")}
        exp = np.mean([ri.parse_prediction(i, ans[i]["content"])[0]["emotion_label"] == gold[i] for i in val_ids])
        pr = load("pr_c", PROBE)
        class D:  # all 600 val points: gen accuracy must be 473/600 (#4's lenient parser), not 472
            labels = sorted(ri.EMOTIONS)
        all_ids = [json.loads(l)["id"] for l in open(REPO / "data/t10_input.jsonl", encoding="utf-8-sig")]
        D.ids, D.idx = all_ids, {"val": np.arange(len(all_ids))}
        gp, glp = pr.llm_answers(REPO / "outputs/exp023/lp_fp8_t10.jsonl", ri, D)
        n_ok = int(sum(D.labels[g] == gold[i] for g, i in zip(gp, all_ids)))
        check("C2b gen on all 600 val points = 473 (0.788)", n_ok == 473 and np.allclose(np.exp(glp).sum(1), 1), f"{n_ok}/600")
        check("C2 gen accuracy on the 12 val points = parsed FP8 answers", abs(res["emotion"]["gen"]["val_acc"] - exp) < 1e-9,
              f"{res['emotion']['gen']['val_acc']:.3f}")

    # session script, fresh then resume (GPU checks off, CPU everywhere)
    env = {"U": str(REPO), "MODEL": str(tmp / "tiny"), "FEATS": str(tmp / "sess_feats"), "OUT": str(tmp / "sess_out"),
           "PY": sys.executable, "SFT_TRAIN": str(tmp / "train.jsonl"), "SFT_DEV": str(tmp / "dev_sharegpt.json"),
           "VAL_INPUT": str(tmp / "val_input.jsonl"), "N_TRAIN": "0", "WORLD": "2", "PROBE_DEVICE": "cpu",
           "RUN_EXP027": "0", "EXP029_SKIP_GPU_CHECK": "1", "EXP029_POLL_S": "2"}
    r = run(["bash", SESSION], env=env)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    check("C3 exp029_session.sh fresh run ends with DONE", r.returncode == 0 and "DONE -- shut the instance down now" in r.stdout, tail)
    if r.returncode:
        print(r.stdout[-4000:], r.stderr[-2000:])
    r2 = run(["bash", SESSION], env=env)
    check("C4 rerun resumes the cache (no re-plan) and finishes", r2.returncode == 0 and "resuming the existing cache" in r2.stdout
          and "44 / 44 rows done" in r2.stdout, r2.stdout.strip().splitlines()[-1] if r2.stdout.strip() else "")
    r3 = run(["bash", SESSION], env={**env, "MODEL": str(tmp / "nope"), "FEATS": str(tmp / "f3"), "OUT": str(tmp / "o3")})
    check("C5 preflight stops on a missing model before any work", r3.returncode == 1 and "STOP: no model" in r3.stdout
          and not (tmp / "f3").exists())
    r4 = run(["bash", SESSION], env={**env, "FEATS": str(tmp / "f4"), "OUT": str(tmp / "o4"), "RUN_EXP027": "1",
                                       "EXP027": str(tmp / "missing.sh")})
    check("C6 RUN_EXP027=1 with no EXP-027 script stops", r4.returncode == 1 and "not found" in r4.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-train", default=str(REPO / "data/sft/train.jsonl"))
    ap.add_argument("--tokenizer", default=str(REPO / "models/qwen3-8b-tok"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--parts", default="ABC")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="exp029_test_"))
    try:
        make_tiny_model(tmp / "tiny", args.tokenizer)
        small_inputs(tmp, args.sft_train)
        feats = part_a(tmp, args.tokenizer) if "A" in args.parts else None
        if "B" in args.parts:
            part_b(tmp)
        if feats is not None and "C" in args.parts:
            part_c(tmp, feats, args.tokenizer)
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print("kept", tmp)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
