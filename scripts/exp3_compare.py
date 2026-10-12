#!/usr/bin/env python3
"""
EXP-003: Qwen3 thinking vs non-thinking mode, through an OpenAI-compatible API.

For each mode it produces a submission for the given val inputs, validates and
scores it with scripts/validate.py and scripts/eval.py, and writes a
comparison table to <out>/comparison.md.

Two ways of calling the API:
  * default      runs the official participant/run_inference.py (unchanged,
                 on a temporary copy with a per-mode config)
  * --stream     calls the API itself with stream=True, reusing the official
                 script's prompt, history rendering and JSON parser. Needed for
                 providers that only allow streaming in thinking mode. Also saves
                 the raw model output per sample to out_<mode>/raw_outputs.jsonl.

Scoring: eval.py runs with --force, so a blank field counts as wrong (as in the
official scoring) instead of blocking the run; the table reports how many.

How thinking is switched:
  * default        "enable_thinking": true/false in the request body
  * --soft-switch  Qwen3's "/think" / "/no_think" tag appended to the user
                   message, for providers that ignore or reject enable_thinking.
                   Implies --stream.
  * --no-switch    nothing at all, for models without a thinking mode (Qwen2.5).

Examples (PowerShell):
  SiliconFlow:
    $env:SILICONFLOW_API_KEY="sk-..."
    python scripts/exp3_compare.py --participant <package>/participant `
        --input data/val_input_n50.jsonl --gold data/val_gold_n50.jsonl `
        --scripts scripts --out experiments/exp003-qwen3-14b-thinking `
        --base-url https://api.siliconflow.com/v1
  Hugging Face router (use a provider listed on the model page, e.g. nscale):
    $env:HF_TOKEN="hf_..."
    python scripts/exp3_compare.py ... --soft-switch --key-env HF_TOKEN `
        --base-url https://router.huggingface.co/v1 --model "Qwen/Qwen3-14B:nscale"
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# Qwen3 model-card decoding settings. Greedy decoding in thinking mode can
# repeat endlessly, so each mode uses its recommended sampling by default.
MODES = {
    "nothink": {
        "enable_thinking": False,
        "max_new_tokens": 1200,
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    },
    "think": {
        "enable_thinking": True,
        "max_new_tokens": 6000,  # reasoning tokens count against this budget
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
}

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.S)

TABLE_ROWS = [
    ("Valid submission", "valid"),
    ("Emotion accuracy", "acc"),
    ("Emotion macro-F1", "macro_f1"),
    ("Profile micro-F1 (avg)", "profile_micro_f1"),
    ("Profile exact-match (avg)", "profile_exact"),
    ("Parse failures", "parse_failures"),
    ("Blank predictions (scored as wrong)", "blank_predictions"),
    ("Avg latency per sample (s, API)", "avg_latency_s"),
    ("Avg reasoning length (chars)", "reasoning_chars"),
    ("Wall time (min)", "wall_min"),
    ("Reply length mean (chars)", "reply_len_mean"),
]


# ------------------------------------------------------------------ helpers

def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def run(cmd, env=None):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], env=env).returncode


def load_official_module(path):
    spec = importlib.util.spec_from_file_location("official_run_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config(base_cfg, args, spec):
    cfg = dict(base_cfg)
    cfg.update({
        "backend": "openai_compatible",
        "base_url": args.base_url,
        "api_model": args.model,
        "api_key": "",  # the key is passed through the environment, never written to disk
        "hardware_label": f"api:{urlparse(args.base_url).hostname}",
        "api_options": {} if (args.soft_switch or args.no_switch) else {"enable_thinking": spec["enable_thinking"]},
    })
    if args.api_options:  # provider-specific switch, e.g. DeepSeek: {"thinking": {"type": "disabled"}}
        cfg["api_options"] = json.loads(args.api_options)
    generation = {"max_new_tokens": spec["max_new_tokens"], "seed": args.seed}
    if args.greedy:
        generation["do_sample"] = False  # official script turns this into temperature 0
    else:
        generation.update({"do_sample": True, **spec["sampling"]})
    cfg["generation"] = generation
    return cfg


# ------------------------------------------------------------------ streaming client

def _stream_once(url, payload, key, timeout):
    """POST with stream=True; return (content, reasoning) strings."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    content, reasoning = [], []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        if exc.code == 429 or exc.code >= 500:
            raise _Retryable(f"HTTP {exc.code}: {detail[:200]}") from exc
        raise RuntimeError(f"API request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise _Retryable(f"Cannot reach {url}: {exc.reason}") from exc
    except OSError as exc:  # SSL EOF, connection reset, timeout while reading the stream
        raise _Retryable(f"Connection dropped ({type(exc).__name__}): {exc}") from exc
    return "".join(content), "".join(reasoning)


class _Retryable(Exception):
    """Network drop or 429/5xx: worth retrying."""


def stream_completion(url, payload, key, timeout, retries=5):
    """_stream_once with retries and backoff (5, 10, 20, 40, 60 s) for flaky VPN connections."""
    for attempt in range(retries + 1):
        try:
            return _stream_once(url, payload, key, timeout)
        except _Retryable as exc:
            if attempt == retries:
                raise RuntimeError(f"{exc} (gave up after {retries} retries)") from exc
            wait = min(60, 5 * 2 ** attempt)
            print(f"   ! {exc} -> retry {attempt + 1}/{retries} in {wait} s", flush=True)
            time.sleep(wait)


def run_streaming(official, cfg, input_file, res_dir, key, soft_tag):
    """Streaming equivalent of run_inference.py: same prompt, history format and parser."""
    system_prompt = official.SYSTEM_PROMPT.format(
        emotions=", ".join(sorted(official.EMOTIONS)),
        personality=", ".join(sorted(official.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(official.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(official.PROFILE_LABELS["style"])),
    )
    url = official.api_endpoint(cfg["base_url"])
    timeout = float(cfg.get("request_timeout_seconds", 600))
    samples = official.read_jsonl(input_file)

    res_dir.mkdir(parents=True, exist_ok=True)
    submission = res_dir / "submission.jsonl"
    raw_outputs = res_dir / "raw_outputs.jsonl"
    submission.write_text("", encoding="utf-8")
    raw_outputs.write_text("", encoding="utf-8")
    latencies, reasoning_chars, failures = [], [], []

    for index, sample in enumerate(samples, 1):
        user_text = official.conversation_text(sample)
        if soft_tag:
            user_text = f"{user_text} {soft_tag}"
        payload = {
            "model": cfg["api_model"],
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_text}],
            "stream": True,
            **official.api_generation_options(cfg["generation"]),
            **cfg["api_options"],
        }
        started = time.perf_counter()
        content, reasoning = stream_completion(url, payload, key, timeout)
        prediction, parse_failed = official.parse_prediction(sample["id"], content)
        elapsed_ms = (time.perf_counter() - started) * 1000

        inline_reasoning = "".join(THINK_BLOCK.findall(content))  # some providers inline it
        latencies.append(elapsed_ms)
        reasoning_chars.append(len(reasoning.strip()) + len(inline_reasoning.strip()))
        if parse_failed:
            failures.append(sample["id"])
        with submission.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
        with raw_outputs.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": sample["id"], "content": content, "reasoning": reasoning},
                                    ensure_ascii=False) + "\n")
        print(f"[{index}/{len(samples)}] {sample['id']}: {elapsed_ms:.0f} ms, "
              f"reasoning {reasoning_chars[-1]} chars", flush=True)

    report = {
        "version": "exp3_streaming (official prompt, history format and parser)",
        "api_model": cfg["api_model"],
        "base_url": cfg["base_url"],
        "samples": len(samples),
        "average_latency_ms": round(statistics.fmean(latencies), 3),
        "median_latency_ms": round(statistics.median(latencies), 3),
        "avg_reasoning_chars": round(statistics.fmean(reasoning_chars), 1),
        "parse_failure_count": len(failures),
        "parse_failure_ids": failures,
    }
    (res_dir / "performance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ scoring and report

def score_run(res_dir, input_file, gold_file, scripts, wall_seconds):
    submission = res_dir / "submission.jsonl"
    metrics_path = res_dir / "metrics.json"
    valid = run([sys.executable, scripts / "validate.py", submission, "--input", input_file]) == 0
    # --force: a blank field (the official parser's fallback) counts as wrong, as in the official
    # scoring, instead of blocking the whole run. Check "Valid submission" and "Blank predictions".
    run([sys.executable, scripts / "eval.py", submission, gold_file, "--out", metrics_path,
         "--history", "", "--force"])

    perf = load_json(res_dir / "performance_report.json") or {}
    metrics = load_json(metrics_path) or {}
    emotion = metrics.get("emotion", {})
    profile = metrics.get("profile", {})
    response = metrics.get("response", {})
    return {
        "failed": False,
        "valid": valid,
        "wall_min": round(wall_seconds / 60, 1),
        "avg_latency_s": round(perf.get("average_latency_ms", 0) / 1000, 2),
        "parse_failures": perf.get("parse_failure_count"),
        "blank_predictions": emotion.get("n_invalid_predictions"),
        "reasoning_chars": perf.get("avg_reasoning_chars", "n/a (use --stream)"),
        "acc": emotion.get("accuracy"),
        "macro_f1": emotion.get("macro_f1"),
        "profile_micro_f1": profile.get("micro_f1_avg"),
        "profile_exact": profile.get("exact_match_avg"),
        "reply_len_mean": response.get("length_pred", {}).get("mean"),
        "reply_len_gold": response.get("length_gold", {}).get("mean"),
        "baseline_acc": metrics.get("baselines", {}).get("always_anxiety_accuracy"),
    }


def write_report(results, args, input_file, out):
    modes = list(results)
    decoding = "greedy (temperature 0)" if args.greedy else "Qwen recommended sampling per mode"
    if args.api_options:
        switch = f"api_options {args.api_options}"
    elif args.no_switch:
        switch = "none (model without thinking mode)"
    elif args.soft_switch:
        switch = "/think, /no_think tags"
    else:
        switch = "enable_thinking parameter"
    lines = [
        f"# Results: {args.model} via {args.base_url}, {input_file.name}",
        "",
        f"Decoding: {decoding}, seed {args.seed}. Thinking switch: {switch}.",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for label, key in TABLE_ROWS:
        cells = ["FAILED" if results[m]["failed"] else str(results[m].get(key)) for m in modes]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    ok = next((r for r in results.values() if not r["failed"]), None)
    if ok:
        lines += ["", f"Reference: always-anxiety accuracy = {ok['baseline_acc']}, "
                      f"gold reply length mean = {ok['reply_len_gold']} chars."]
    lines += ["", "Check: 'think' should be clearly slower and (with --stream) show reasoning > 0 chars. "
                  "If not, the provider ignored the thinking switch."]
    report = "\n".join(lines)
    print("\n" + report)
    (out / "comparison.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: {out / 'comparison.md'}  (+ out_<mode>/ with submission, performance report, metrics)")


# ------------------------------------------------------------------ main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="official participant/ folder (contains run_inference.py)")
    ap.add_argument("--input", required=True, help="test-format input, e.g. data/val_input_n50.jsonl")
    ap.add_argument("--gold", required=True, help="matching gold file, e.g. data/val_gold_n50.jsonl")
    ap.add_argument("--scripts", required=True, help="folder with validate.py and eval.py")
    ap.add_argument("--out", required=True, help="output folder for this experiment")
    ap.add_argument("--modes", nargs="+", default=["nothink", "think"], choices=list(MODES))
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    ap.add_argument("--key-env", default="SILICONFLOW_API_KEY", help="environment variable holding the API key")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--greedy", action="store_true", help="temperature 0 for both modes")
    ap.add_argument("--stream", action="store_true", help="use the built-in streaming client")
    ap.add_argument("--soft-switch", action="store_true",
                    help="switch thinking with /think and /no_think tags (implies --stream)")
    ap.add_argument("--no-switch", action="store_true",
                    help="send no thinking switch at all, for models without a thinking mode "
                         "(e.g. Qwen2.5); only valid with --modes nothink")
    ap.add_argument("--api-options", help="JSON merged into every request body instead of enable_thinking, "
                    "e.g. DeepSeek non-thinking: '{\"thinking\": {\"type\": \"disabled\"}}'; only with --modes nothink")
    args = ap.parse_args()
    if args.api_options:
        try:
            if not isinstance(json.loads(args.api_options), dict):
                raise ValueError
        except ValueError:
            ap.error("--api-options must be a JSON object")
        if "think" in args.modes:
            ap.error("--api-options only works with --modes nothink")
    if args.no_switch and (args.soft_switch or "think" in args.modes):
        ap.error("--no-switch can't be combined with --soft-switch or the 'think' mode")
    return args


def main():
    args = parse_args()
    key = os.environ.get(args.key_env, "").strip()
    if len(key) < 20:
        sys.exit(f'Set your API key first:  $env:{args.key_env}="..."')

    participant = Path(args.participant).resolve()
    input_file = Path(args.input).resolve()
    gold_file = Path(args.gold).resolve()
    scripts = Path(args.scripts).resolve()
    out = Path(args.out).resolve()
    for path in [participant / "run_inference.py", participant / "configs" / "inference_config.json",
                 input_file, gold_file, scripts / "validate.py", scripts / "eval.py"]:
        if not path.exists():
            sys.exit(f"Not found: {path}")
    base_cfg = load_json(participant / "configs" / "inference_config.json")
    if not isinstance(base_cfg, dict):
        sys.exit("participant/configs/inference_config.json is not valid JSON")
    out.mkdir(parents=True, exist_ok=True)

    # Work on a copy so the official participant/ folder is never modified.
    work = out / "_participant_copy"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(participant, work, ignore=shutil.ignore_patterns("model", "__pycache__", "~"))
    official = load_official_module(work / "run_inference.py")
    use_stream = args.stream or args.soft_switch

    # Other EVALUATION_* variables would override the config file, so drop them.
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVALUATION_")}
    env["EVALUATION_API_KEY"] = key
    env["PYTHONIOENCODING"] = "utf-8"

    results = {}
    try:
        for mode in args.modes:
            spec = MODES[mode]
            cfg = build_config(base_cfg, args, spec)
            config_text = json.dumps(cfg, ensure_ascii=False, indent=2)
            (work / "configs" / "inference_config.json").write_text(config_text, encoding="utf-8")
            (out / f"config_{mode}.json").write_text(config_text, encoding="utf-8")

            res_dir = out / f"out_{mode}"
            print("\n" + "=" * 64 + f"\nRUN: {mode} (thinking={spec['enable_thinking']})\n" + "=" * 64)
            started = time.time()
            if use_stream:
                soft_tag = None
                if args.soft_switch:
                    soft_tag = "/think" if spec["enable_thinking"] else "/no_think"
                try:
                    run_streaming(official, cfg, input_file, res_dir, key, soft_tag)
                    ok = True
                except RuntimeError as exc:
                    print(f"!! {exc}")
                    ok = False
            else:
                ok = run([sys.executable, work / "run_inference.py", input_file, res_dir], env=env) == 0

            if not ok:
                print(f"!! {mode}: inference failed, see the error above.")
                results[mode] = {"failed": True}
                continue
            results[mode] = score_run(res_dir, input_file, gold_file, scripts, time.time() - started)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    write_report(results, args, input_file, out)


if __name__ == "__main__":
    main()
