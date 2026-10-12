#!/usr/bin/env python3
"""
EXP-017: teacher test. Does DeepSeek V4 write replies closer to the gold than our model?

One command does everything (Hugging Face router, key in $env:HF_TOKEN):
  1. Switch sweep on n9: tries providers x thinking-switch options until one run has
     0 reasoning chars and 0 parse failures (= thinking really off, JSON intact).
  2. Runs DeepSeek-V4-Pro and DeepSeek-V4.1-Flash on n50b with the working combo.
  3. Prints a table vs V1.8 (Qwen3-8B, n50b) and the verdict, and writes
     experiments/exp017-teacher/README.md.

Usage (PowerShell, from the repo root, VPN on):
  $env:HF_TOKEN="hf_..."
  python scripts/exp017_teacher.py

Cost: a few cents of HF credit. Each n9 try takes about 1 minute, each n50b run 3-6 minutes.
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "exp017-teacher"
BASE_URL = "https://router.huggingface.co/v1"
KEY_ENV = "HF_TOKEN"

MODELS = {
    "v4-pro": ("deepseek-ai/DeepSeek-V4-Pro-0813", ["novita", "deepinfra", "together", "baseten"]),
    "v4.1-flash": ("deepseek-ai/DeepSeek-V4.1-Flash", ["novita", "deepinfra", "fireworks-ai", "baseten"]),
}
SWITCHES = {
    "deepseek": {"thinking": {"type": "disabled"}},
    "tmpl_thinking": {"chat_template_kwargs": {"thinking": False}},
    "tmpl_enable_thinking": {"chat_template_kwargs": {"enable_thinking": False}},
}
KNOWN_BAD = {("v4-pro", "novita", "deepseek")}  # tested Oct 6: thinking stayed on, JSON cut

# V1.8 (Qwen3-8B non-thinking, greedy) on n50b = the bar (EXP-016)
BASELINE = {"acc": 0.62, "macro_f1": 0.3743, "profile_micro_f1": 0.2999, "rouge_l": 0.1996, "bleu4": 0.068}
PASS_ROUGE_L = 0.24

SLICES = {
    "n9": ("data/val_input_n9.jsonl", "data/val_gold_n9.jsonl"),
    "n50b": ("data/val_input_n50b.jsonl", "data/val_gold_n50b.jsonl"),
}


def run(model_id, provider, switch_name, slice_name, out_dir):
    """Run exp3_compare.py once. Returns its result dict (or None if the run crashed)."""
    inp, gold = SLICES[slice_name]
    cmd = [
        sys.executable, str(ROOT / "scripts" / "exp3_compare.py"),
        "--participant", str(ROOT / "participant"),
        "--input", str(ROOT / inp), "--gold", str(ROOT / gold),
        "--scripts", str(ROOT / "scripts"), "--out", str(out_dir),
        "--base-url", BASE_URL, "--key-env", KEY_ENV,
        "--model", f"{model_id}:{provider}",
        "--modes", "nothink", "--greedy", "--stream",
        "--api-options", json.dumps(SWITCHES[switch_name]),  # passed as a list: no PowerShell quoting issues
    ]
    print(f"\n>>> {model_id}:{provider} | switch={switch_name} | {slice_name}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    report = out_dir / "out_nothink" / "performance_report.json"
    if proc.returncode != 0 or not report.exists():
        print(f"    run failed (exit {proc.returncode})")
        return None
    perf = json.loads(report.read_text(encoding="utf-8"))
    res = {
        "reasoning": perf.get("avg_reasoning_chars", 0) or 0,
        "parse_failures": perf.get("parse_failure_count", 0) or 0,
        "latency_s": round(perf.get("average_latency_ms", 0) / 1000, 2),
    }
    metrics_file = out_dir / "out_nothink" / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        res.update({
            "acc": m["emotion"]["accuracy"],
            "macro_f1": m["emotion"]["macro_f1"],
            "profile_micro_f1": m["profile"]["micro_f1_avg"],
            "rouge_l": m["response"]["rouge_l"],
            "bleu4": m["response"]["bleu4"],
        })
    sub = out_dir / "out_nothink" / "submission.jsonl"
    if sub.exists():
        lens = [len(json.loads(l).get("response_text") or "") for l in sub.read_text(encoding="utf-8").splitlines() if l.strip()]
        res["reply_median"] = statistics.median(lens) if lens else 0
    print(f"    reasoning={res['reasoning']} chars, parse failures={res['parse_failures']}")
    return res


def valid(res):
    return res is not None and res["reasoning"] == 0 and res["parse_failures"] == 0


def find_switch(name, providers, switch_order):
    model_id = MODELS[name][0]
    for provider in providers:
        for switch_name in switch_order:
            if (name, provider, switch_name) in KNOWN_BAD:
                continue
            out_dir = OUT / "switch" / f"{name}_{provider}_{switch_name}"
            if valid(run(model_id, provider, switch_name, "n9", out_dir)):
                print(f"    OK: {name} works with {provider} + {switch_name}")
                return provider, switch_name
    return None


def gold_median(slice_name):
    path = ROOT / SLICES[slice_name][1]
    lens = [len(json.loads(l)["response_text"]) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    return statistics.median(lens)


def main():
    if len(os.environ.get(KEY_ENV, "").strip()) < 20:
        sys.exit(f"Set the key first: $env:{KEY_ENV}=\"hf_...\"")
    OUT.mkdir(parents=True, exist_ok=True)

    results, combos = {}, {}
    for name, (model_id, providers) in MODELS.items():
        print("\n" + "=" * 70 + f"\n{name}: finding a provider + switch with thinking OFF (n9)\n" + "=" * 70)
        # try the combo that already worked for another model first
        order = list(SWITCHES)
        prov_order = list(providers)
        if combos:
            p0, s0 = next(iter(combos.values()))
            order = [s0] + [s for s in order if s != s0]
            if p0 in prov_order:
                prov_order = [p0] + [p for p in prov_order if p != p0]
        found = find_switch(name, prov_order, order)
        if not found:
            print(f"\n!!! {name}: no provider/switch turned thinking off. Skipping it.")
            continue
        combos[name] = found
        provider, switch_name = found
        print("\n" + "=" * 70 + f"\n{name}: n50b run ({provider}, {switch_name})\n" + "=" * 70)
        res = run(model_id, provider, switch_name, "n50b", OUT / f"{name}_n50b")
        if not valid(res):
            print(f"!!! {name}: n50b run invalid (reasoning or parse failures > 0). Not scored.")
            continue
        results[name] = res

    if not results:
        sys.exit("\nNo valid n50b run. Use DeepSeek's own API (api.deepseek.com) instead; there the switch is documented.")

    gm = gold_median("n50b")
    rows = [("Emotion accuracy", "acc"), ("Emotion macro-F1", "macro_f1"), ("Profile micro-F1", "profile_micro_f1"),
            ("ROUGE-L", "rouge_l"), ("BLEU-4", "bleu4"), ("Reply median (chars)", "reply_median"),
            ("Latency (s, API)", "latency_s")]
    names = list(results)
    lines = ["# EXP-017: teacher test (n50b, greedy, thinking off)", "",
             "| Metric | V1.8 (Qwen3-8B) | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names)]
    for label, key in rows:
        base = BASELINE.get(key, "")
        lines.append(f"| {label} | {base} | " + " | ".join(str(results[n].get(key, "")) for n in names) + " |")
    lines += ["", f"Gold reply median on n50b: {gm} chars.", ""]
    for n in names:
        p, s = combos[n]
        lines.append(f"- {n}: {MODELS[n][0]}:{p}, switch `{json.dumps(SWITCHES[s])}`")
    lines.append("")
    best = max(names, key=lambda n: results[n]["rouge_l"])
    if results[best]["rouge_l"] >= PASS_ROUGE_L:
        verdict = (f"**PASS**: {best} ROUGE-L {results[best]['rouge_l']} >= {PASS_ROUGE_L}. "
                   "Its replies are closer to the gold than ours -> rewrite + rare-emotion augmentation go into P2, with this model.")
    else:
        verdict = (f"**FAIL**: best ROUGE-L {results[best]['rouge_l']} ({best}) < {PASS_ROUGE_L}. "
                   "It writes well but not like the gold -> no synthetic data; train on filtered gold only.")
    lines.append(verdict)
    report = "\n".join(lines) + "\n"
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved: {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
