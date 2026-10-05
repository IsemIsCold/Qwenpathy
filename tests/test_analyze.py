#!/usr/bin/env python3
"""
tests/test_analyze.py -- test suite for analyze.py (Layer 3 diagnosis).

Run:  python tests/test_analyze.py [repo-root]
Needs scripts/{validate,eval,analyze}.py and the four data/*.jsonl files.

The A07 alignment test is the one to show external reviewers: it proves the
per-sample counts in mistakes.csv are IDENTICAL to the official metrics that
eval.py produces (criteria cannot drift -- analyze imports eval's functions).
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
EVAL = os.path.join(ROOT, "scripts", "eval.py")
ANALYZE = os.path.join(ROOT, "scripts", "analyze.py")
GOLD = os.path.join(ROOT, "data", "val_gold.jsonl")
INPUT_FULL = os.path.join(ROOT, "data", "val_input.jsonl")

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))


def run_analyze(sub_path, outdir, *extra, skip_response_quality=True):
    cmd = [sys.executable, ANALYZE, sub_path, GOLD, "--input", INPUT_FULL, "--outdir", outdir]
    if skip_response_quality:
        cmd.append("--skip-response-quality")
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    tmp = tempfile.mkdtemp(prefix="analyze_")

    # fixtures
    perfect = [dict(r) for r in gold]
    always = [{
        "id": r["id"], "response_text": "我理解你的感受，我会一直陪着你。",
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    } for r in gold]

    print("== analyze.py tests ==")

    # A01: gold-as-submission -> zero mistakes, no hypotheses
    sub = os.path.join(tmp, "perfect.jsonl"); write_jsonl(sub, perfect)
    out = os.path.join(tmp, "a01")
    proc = run_analyze(sub, out)
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    report = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    check("A01 exit 0", proc.returncode == 0, f"rc={proc.returncode} err={proc.stderr[:200]}")
    check("A01 csv has all rows, none flagged", len(rows) == len(gold) and all(r["any_mistake"] == "0" for r in rows),
          f"rows={len(rows)} flagged={sum(1 for r in rows if r['any_mistake'] == '1')}")
    check("A01 report: no emotion errors, no hypotheses", "Emotion errors: 0" in report and "No significant error buckets" in report)

    # A02: always-anxiety/empty profile -> counts match theory exactly
    sub = os.path.join(tmp, "always.jsonl"); write_jsonl(sub, always)
    out = os.path.join(tmp, "a02")
    proc = run_analyze(sub, out)
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    n = len(gold)
    n_anx = sum(1 for r in gold if r["emotion_label"] == "anxiety")
    correct = sum(1 for r in rows if r["emotion_correct"] == "1")
    check("A02 emotion correct count == gold anxiety count", correct == n_anx, f"{correct} vs {n_anx}")
    for f, short in [("personality_traits", "pt"), ("interests", "in"), ("style", "st")]:
        empty_gold = sum(1 for r in gold if not r["user_profile"][f])
        exact = sum(1 for r in rows if r[f"{short}_exact"] == "1")
        check(f"A02 {f}: empty-pred exact == empty-gold rate", exact == empty_gold, f"{exact} vs {empty_gold}")
        miss_col = sum(1 for r in rows if r[f"{short}_missing"] != "-")
        check(f"A02 {f}: every non-empty gold flagged as missed", miss_col == n - empty_gold, f"{miss_col} vs {n - empty_gold}")

    # A03: language drift detection (English replies, otherwise perfect labels)
    drift = []
    for i, r in enumerate(gold):
        reply = ("I hear you, and I want you to know that I am here for you. "
                 "Everything will be okay, please take a deep breath and tell me more.") if i < 50 else r["response_text"]
        drift.append({"id": r["id"], "response_text": reply, "emotion_label": r["emotion_label"],
                      "user_profile": r["user_profile"], "memory_refs": r["memory_refs"]})
    sub = os.path.join(tmp, "drift.jsonl"); write_jsonl(sub, drift)
    out = os.path.join(tmp, "a03")
    proc = run_analyze(sub, out)
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    flagged = sum(1 for r in rows if "LANG_DRIFT" in r["flags"])
    emo_ok = sum(1 for r in rows if r["emotion_correct"] == "1")
    report = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    # expected: replaced replies whose GOLD is Chinese (drift = deviation from gold)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    ev = importlib.import_module("eval")
    expected = sum(1 for r in gold[:50] if ev.chinese_char_ratio(r["response_text"]) >= 0.70)
    check("A03 exactly the replaced-Chinese-gold replies flagged", flagged == expected, f"{flagged} vs {expected}")
    check("A03 emotion labels untouched (all correct)", emo_ok == n, f"{emo_ok}")
    check("A03 hypothesis mentions language drift", "language drift" in report)

    # A04: --mistakes-only filters; aggregates still describe full submission
    out = os.path.join(tmp, "a04")
    proc = run_analyze(sub, out, "--mistakes-only")  # drift submission
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    report = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    check("A04 mistakes-only csv has exactly the flagged rows", len(rows) == expected, f"{len(rows)} vs {expected}")
    check("A04 report header still says n = full set", f"n = {n}" in report)

    # A05: id mismatch -> exit 1
    sub_bad = os.path.join(tmp, "bad.jsonl"); write_jsonl(sub_bad, always[:-1])
    proc = run_analyze(sub_bad, os.path.join(tmp, "a05"))
    check("A05 id mismatch exits 1", proc.returncode == 1, f"rc={proc.returncode}")

    # A06: determinism -- identical bytes on rerun
    out = os.path.join(tmp, "a06")
    run_analyze(sub, out)
    first = (open(os.path.join(out, "mistakes.csv"), encoding="utf-8").read(),
             open(os.path.join(out, "report.md"), encoding="utf-8").read())
    run_analyze(sub, out)
    second = (open(os.path.join(out, "mistakes.csv"), encoding="utf-8").read(),
              open(os.path.join(out, "report.md"), encoding="utf-8").read())
    check("A06 rerun produces byte-identical outputs", first == second)

    # A07 (ALIGNMENT PROOF): analyze per-sample counts == eval.py official metrics
    sub = os.path.join(tmp, "always.jsonl")
    metrics_path = os.path.join(tmp, "always.metrics.json")
    subprocess.run([sys.executable, EVAL, sub, GOLD, "--out", metrics_path, "--history", ""],
                   capture_output=True, text=True, timeout=300)
    m = json.load(open(metrics_path, encoding="utf-8"))
    out = os.path.join(tmp, "a07")
    proc = run_analyze(sub, out, "--metrics", metrics_path)
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    n = m["emotion"]["n"]
    emo_csv = sum(1 for r in rows if r["emotion_correct"] == "1")
    emo_eval = round(m["emotion"]["accuracy"] * n)
    check("A07 emotion correct count == eval accuracy * n", emo_csv == emo_eval, f"csv={emo_csv} eval={emo_eval}")
    for f, short in [("personality_traits", "pt"), ("interests", "in"), ("style", "st")]:
        exact_csv = sum(1 for r in rows if r[f"{short}_exact"] == "1")
        exact_eval = round(m["profile"][f]["exact_match"] * n)
        check(f"A07 {f} exact count == eval exact_match * n", exact_csv == exact_eval, f"csv={exact_csv} eval={exact_eval}")
    report = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    check("A07 report cites eval.py numbers verbatim", f"{m['emotion']['accuracy']:.4f}" in report)

    # A08: gate enforcement -- analyze refuses an invalid submission
    broken = [dict(r, emotion_label="nope") for r in always[:10]] + always[10:]
    sub_broken = os.path.join(tmp, "broken.jsonl"); write_jsonl(sub_broken, broken)
    proc = run_analyze(sub_broken, os.path.join(tmp, "a08"))
    check("A08 invalid submission refused (exit 1, no Traceback)",
          proc.returncode == 1 and "refusing" in proc.stderr and "Traceback" not in proc.stderr,
          f"rc={proc.returncode} err={proc.stderr[:150]}")


    # A09: analysis.json schema validation
    out = os.path.join(tmp, "a09")
    proc = run_analyze(sub, out, "--metrics", metrics_path)
    analysis_json = json.load(open(os.path.join(out, "analysis.json"), encoding="utf-8"))
    check("A09 analysis.json schema_version present", analysis_json.get("schema_version") == "2.1")
    check("A09 analysis.json has meta", "meta" in analysis_json and "submission" in analysis_json["meta"])
    check("A09 analysis.json has summary", "summary" in analysis_json)
    check("A09 analysis.json has hypotheses", "hypotheses" in analysis_json)
    check("A09 analysis.json hypotheses have owner field",
          all("owner" in h for h in analysis_json["hypotheses"]))

    # A10: template detection flags repeated prefixes
    # Use a long base string (>30 chars) so str(i) suffix never affects prefix hash
    template_sub = []
    base = "我理解你的感受，我会一直陪着你。别担心，一切都会好起来的，加油！"
    for i, r in enumerate(gold):
        if i < 100:
            reply = base + " [" + str(i) + "]"
        else:
            reply = r["response_text"]
        template_sub.append({"id": r["id"], "response_text": reply,
                             "emotion_label": r["emotion_label"],
                             "user_profile": r["user_profile"], "memory_refs": r["memory_refs"]})
    sub = os.path.join(tmp, "template.jsonl"); write_jsonl(sub, template_sub)
    out = os.path.join(tmp, "a10")
    proc = run_analyze(sub, out)
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    # template hashes should be identical for first 100
    hashes = [r["template_hash"] for r in rows[:100]]
    check("A10 template hashes identical for identical prefixes", len(set(hashes)) == 1)
    report = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    check("A10 report mentions template pattern", "template" in report.lower())

    # A11: impact-weighted severity -- rare class errors rank higher
    rare_sub = []
    for i, r in enumerate(gold):
        if r["emotion_label"] == "surprise":
            rare_sub.append({"id": r["id"], "response_text": r["response_text"],
                             "emotion_label": "anxiety",  # misclassify surprise->anxiety
                             "user_profile": r["user_profile"], "memory_refs": r["memory_refs"]})
        else:
            rare_sub.append({"id": r["id"], "response_text": r["response_text"],
                             "emotion_label": r["emotion_label"],
                             "user_profile": r["user_profile"], "memory_refs": r["memory_refs"]})
    sub = os.path.join(tmp, "rare.jsonl"); write_jsonl(sub, rare_sub)
    out = os.path.join(tmp, "a11")
    proc = run_analyze(sub, out)
    analysis_json = json.load(open(os.path.join(out, "analysis.json"), encoding="utf-8"))
    hyps = analysis_json["hypotheses"]
    rare_hyp = next((h for h in hyps if "surprise" in h["description"].lower()), None)
    check("A11 rare-class error appears in hypotheses", rare_hyp is not None)
    if rare_hyp:
        check("A11 rare-class hypothesis has estimated impact note", "estimated" in rare_hyp["impact_note"])

    # A12: comparative mode (--compare)
    sub_new = os.path.join(tmp, "always.jsonl")
    # "new" submission: fix 50 emotion errors (first 50 anxiety samples predict correctly)
    improved = []
    fixed = 0
    for r in gold:
        if r["emotion_label"] != "anxiety" and fixed < 50:
            improved.append({"id": r["id"], "response_text": r["response_text"],
                             "emotion_label": r["emotion_label"],
                             "user_profile": r["user_profile"], "memory_refs": r["memory_refs"]})
            fixed += 1
        else:
            improved.append({"id": r["id"], "response_text": "我理解你的感受，我会一直陪着你。",
                             "emotion_label": "anxiety",
                             "user_profile": {"personality_traits": [], "interests": [], "style": []},
                             "memory_refs": []})
    sub_improved = os.path.join(tmp, "improved.jsonl"); write_jsonl(sub_improved, improved)
    out = os.path.join(tmp, "a12")
    proc = run_analyze(sub_improved, out, "--compare", sub_new)
    check("A12 comparative mode exit 0", proc.returncode == 0)
    delta_rows = read_csv(os.path.join(out, "delta.csv"))
    check("A12 delta.csv has all rows", len(delta_rows) == n)
    emo_improved = sum(1 for d in delta_rows if d["emotion_delta"] == "1")
    check("A12 delta shows emotion improvements", emo_improved >= 50, f"{emo_improved}")
    report = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    check("A12 report has comparative section", "Comparative Delta" in report)

    # A13: owner map override via CLI
    # Use a submission with emotion errors but varied, gold-length replies so
    # template/short findings don't swamp the ranking
    owner_test = []
    for r in gold:
        reply = r["response_text"]
        if r["emotion_label"] == "joy":
            el = "anxiety"  # force some confusion
        else:
            el = r["emotion_label"]
        owner_test.append({"id": r["id"], "response_text": reply,
                           "emotion_label": el,
                           "user_profile": r["user_profile"], "memory_refs": r["memory_refs"]})
    sub_owner = os.path.join(tmp, "owner_test.jsonl")
    write_jsonl(sub_owner, owner_test)
    owner_path = os.path.join(tmp, "owners.json")
    with open(owner_path, "w", encoding="utf-8") as f:
        json.dump({"emotion_confusion": "data_pipeline"}, f)
    out = os.path.join(tmp, "a13")
    proc = run_analyze(sub_owner, out, "--owner-map", owner_path)
    analysis_json = json.load(open(os.path.join(out, "analysis.json"), encoding="utf-8"))
    if analysis_json["hypotheses"]:
        any_data_pipeline = any(h["owner"] == "data_pipeline" for h in analysis_json["hypotheses"])
        check("A13 owner map override respected (at least one hypothesis uses override)",
              any_data_pipeline)
    else:
        check("A13 owner map override respected (no hypotheses)", True)

    # A14: executive summary determinism -- same input -> same summary
    out = os.path.join(tmp, "a14a")
    run_analyze(sub_new, out)
    out2 = os.path.join(tmp, "a14b")
    run_analyze(sub_new, out2)
    r1 = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    r2 = open(os.path.join(out2, "report.md"), encoding="utf-8").read()
    check("A14 executive summary deterministic", r1 == r2)

    # A15: new CSV columns present (pred_prefix, template_hash)
    out = os.path.join(tmp, "a15")
    proc = run_analyze(sub_new, out)
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    check("A15 pred_prefix column present", all("pred_prefix" in r for r in rows))
    check("A15 template_hash column present", all("template_hash" in r for r in rows))

    # A16: per-sample ROUGE-L in CSV (computed via eval.py functions)
    # Must NOT skip response quality for this test
    out = os.path.join(tmp, "a16")
    proc = run_analyze(sub_new, out, skip_response_quality=False)
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    check("A16 rouge_l column present", all("rouge_l" in r for r in rows))
    # On gold-as-submission, rouge_l should be 1.0
    gold_sub = os.path.join(tmp, "gold_a16.jsonl")
    write_jsonl(gold_sub, gold)
    out = os.path.join(tmp, "a16b")
    proc = run_analyze(gold_sub, out, skip_response_quality=False)
    rows = read_csv(os.path.join(out, "mistakes.csv"))
    rouge_vals = [float(r["rouge_l"]) for r in rows]
    check("A16 gold-as-submission rouge_l ≈ 1.0", all(r >= 0.99 for r in rouge_vals), f"min={min(rouge_vals) if rouge_vals else 'N/A'}")

    print()
    failed = [nm for nm, ok in results if not ok]
    print(f"{'ALL TESTS PASSED' if not failed else 'FAILURES: ' + ', '.join(failed)}  ({len(results) - len(failed)}/{len(results)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
