#!/usr/bin/env python3
"""
tests/run_tests.py -- attack-test suite for validate.py + smoke tests for eval.py.

Every test builds a deliberately broken (or valid) submission from the real n50
data, runs the script as a subprocess, and asserts the exit code + summary.

Run:  python tests/run_tests.py [repo-root]
With no argument, the repo root is inferred from this file's location
(parent of tests/). With an argument, that path is used instead -- this is
how you run the suite against a different checkout.

Needs: scripts/validate.py, scripts/eval.py, and in data/:
  val_input.jsonl, val_gold.jsonl, val_input_n50.jsonl, val_gold_n50.jsonl
"""

import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Priority 1 fix: honor an explicit repo-root argument; default to inferring it.
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
VALIDATE = os.path.join(ROOT, "scripts", "validate.py")
EVAL = os.path.join(ROOT, "scripts", "eval.py")

DATA_INPUT_N50 = os.path.join(ROOT, "data", "val_input_n50.jsonl")
DATA_GOLD_N50 = os.path.join(ROOT, "data", "val_gold_n50.jsonl")
DATA_GOLD_FULL = os.path.join(ROOT, "data", "val_gold.jsonl")

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))


def run_validate(sub_path, input_path):
    return subprocess.run(
        [sys.executable, VALIDATE, sub_path, "--input", input_path],
        capture_output=True, text=True, timeout=120,
    )


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_raw(path, text, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


def main():
    gold_n50 = [json.loads(l) for l in open(DATA_GOLD_N50, encoding="utf-8")]
    gold_full = [json.loads(l) for l in open(DATA_GOLD_FULL, encoding="utf-8")]
    tmp = tempfile.mkdtemp(prefix="attack_")

    def valid_sub():
        return [
            {**r, "user_profile": {k: list(v) for k, v in r["user_profile"].items()}}
            for r in gold_n50
        ]

    def mutate(records, fn, which=0):
        r = [dict(x) for x in records]
        fn(r[which])
        return r

    print("== validate.py attack tests (n50) ==")
    cases = []

    cases.append(("T01 valid file passes", valid_sub(), 0, ""))
    cases.append(("T02 invalid JSON line", None, 1, "1 json errors"))
    cases.append(("T04 duplicate JSON key", None, 1, "1 json errors"))
    cases.append(("T05 NaN constant", None, 1, "1 json errors"))
    cases.append(("T06 non-object line", None, 1, "1 json errors"))
    cases.append(("T07 missing response_text", mutate(valid_sub(), lambda r: r.pop("response_text")), 1, "1 structural"))
    cases.append(("T08 extra top-level field", mutate(valid_sub(), lambda r: r.update({"cheat": 1})), 1, "1 structural"))

    def bad_emo_upper(r): r["emotion_label"] = "JOY"
    cases.append(("T09 emotion uppercase JOY", mutate(valid_sub(), bad_emo_upper), 1, "1 bad labels"))

    def bad_emo_type(r): r["emotion_label"] = ["anxiety"]
    cases.append(("T10 emotion wrong type", mutate(valid_sub(), bad_emo_type), 1, "1 bad labels"))

    def empty_resp(r): r["response_text"] = "   "
    cases.append(("T11 whitespace response_text", mutate(valid_sub(), empty_resp), 1, "1 structural"))

    def prof_missing(r):
        r["user_profile"] = {k: v for k, v in r["user_profile"].items() if k != "style"}
    cases.append(("T12 profile missing subfield", mutate(valid_sub(), prof_missing), 1, "1 structural"))

    def prof_extra(r): r["user_profile"]["mood"] = ["happy"]
    cases.append(("T13 profile extra subfield", mutate(valid_sub(), prof_extra), 1, "1 structural"))

    def prof_nonstr(r): r["user_profile"]["style"] = ["rational", 42]
    cases.append(("T14 profile non-string value", mutate(valid_sub(), prof_nonstr), 1, "1 structural"))

    # T15: vocabulary confirmed via official participant schema (Oct 4 team review)
    # -> OOV profile values are HARD REJECTED (PROFILE_OOV_HARD_FAIL in validate.py)
    def prof_oov(r): r["user_profile"]["style"] = ["rational", "very_happy"]
    cases.append(("T15 profile OOV value -> hard reject (official vocab confirmed)",
                  mutate(valid_sub(), prof_oov), 1, "1 bad profile values"))

    def bad_mem(r): r["memory_refs"] = ["mem_123", "memory_000001"]
    cases.append(("T16 malformed memory_refs", mutate(valid_sub(), bad_mem), 1, "2 bad memory_refs"))

    def mem_nolist(r): r["memory_refs"] = "mem_000001"
    cases.append(("T17 memory_refs not a list", mutate(valid_sub(), mem_nolist), 1, "1 bad memory_refs"))

    dup = valid_sub(); dup[1] = dict(dup[0])
    cases.append(("T18 duplicate prediction id", dup, 1, "1 duplicate ids"))

    cases.append(("T19 missing prediction id", valid_sub()[1:], 1, "1 missing ids"))

    extra = valid_sub() + [dict(valid_sub()[-1], id="val_conv_999999_t1")]
    cases.append(("T20 extra prediction id", extra, 1, "1 extra ids"))

    swapped = valid_sub(); swapped[0], swapped[1] = swapped[1], swapped[0]
    cases.append(("T21 swapped order -> WARN only, exit 0", swapped, 0, "1 order warnings"))

    crlf = valid_sub()
    cases.append(("T22 blank lines + CRLF pass", crlf, 0, ""))

    cases.append(("T23 count mismatch", valid_sub()[:-1], 1, "1 count mismatch"))

    def dup_profile_label(r): r["user_profile"]["style"] = ["rational", "rational"]
    cases.append(("T26 duplicate profile label tolerated", mutate(valid_sub(), dup_profile_label), 0, ""))

    def dup_mem(r): r["memory_refs"] = ["mem_000001", "mem_000001"]
    cases.append(("T27 duplicate memory_ref tolerated", mutate(valid_sub(), dup_mem), 0, ""))

    def id_null(r): r["id"] = None
    cases.append(("T28 id = null rejected", mutate(valid_sub(), id_null), 1, "1 structural"))

    def id_num(r): r["id"] = 123
    cases.append(("T29 id = 123 rejected", mutate(valid_sub(), id_num), 1, "1 structural"))

    def ctrl_char(r): r["response_text"] = "\x00hidden control char"
    cases.append(("T30 control char in response tolerated", mutate(valid_sub(), ctrl_char), 0, ""))

    def prof_null(r): r["user_profile"] = None
    cases.append(("T31 user_profile = null rejected", mutate(valid_sub(), prof_null), 1, "1 structural"))

    def traits_null(r): r["user_profile"]["personality_traits"] = None
    cases.append(("T32 personality_traits = null rejected", mutate(valid_sub(), traits_null), 1, "1 structural"))

    def mem_null(r): r["memory_refs"] = [None]
    cases.append(("T33 memory_refs = [null] rejected", mutate(valid_sub(), mem_null), 1, "1 bad memory_refs"))

    def emo_empty(r): r["emotion_label"] = ""
    cases.append(("T34 emotion_label = empty rejected", mutate(valid_sub(), emo_empty), 1, "1 bad labels"))

    def prof_nested_extra(r): r["user_profile"]["mood"] = ["happy"]
    cases.append(("T35 extra nested profile field rejected", mutate(valid_sub(), prof_nested_extra), 1, "1 structural"))

    for name, records, want_code, want_snippet in cases:
        sub_path = os.path.join(tmp, "sub.jsonl")
        if name.startswith("T02"):
            write_raw(sub_path, '{"id": "x", "response_text": "y"}\n{not json}\n')
        elif name.startswith("T04"):
            write_raw(sub_path, '{"id": "a", "id": "b", "response_text": "x"}\n')
        elif name.startswith("T05"):
            write_raw(sub_path, '{"id": "a", "response_text": NaN}\n')
        elif name.startswith("T06"):
            write_raw(sub_path, '["i am an array"]\n')
        elif name.startswith("T22"):
            lines = [json.dumps(r, ensure_ascii=False) for r in records]
            write_raw(sub_path, "\n\r\n".join(lines[:10]) + "\r\n\n" + "\n".join(lines[10:]) + "\n")
        else:
            write_jsonl(sub_path, records)

        proc = run_validate(sub_path, DATA_INPUT_N50)
        ok = proc.returncode == want_code
        if want_snippet and want_snippet not in proc.stdout:
            ok = False
        check(name, ok, f"rc={proc.returncode}, want {want_code}; stdout={proc.stdout.strip()[-200:] if not ok else ''}; stderr={proc.stderr.strip()[:200] if not ok else ''}")

    # T03 UTF-16
    sub_path = os.path.join(tmp, "sub_utf16.jsonl")
    with open(sub_path, "w", encoding="utf-16") as f:
        for r in valid_sub()[:5]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    proc = run_validate(sub_path, DATA_INPUT_N50)
    check("T03 UTF-16 file rejected with targeted message", proc.returncode == 2 and "UTF-16" in proc.stderr,
          f"rc={proc.returncode} stderr={proc.stderr[:200]}")

    # T24 BOM tolerated with note
    sub_path = os.path.join(tmp, "sub_bom.jsonl")
    with open(sub_path, "wb") as f:
        f.write(b"\xef\xbb\xbf")
        f.write(("\n".join(json.dumps(r, ensure_ascii=False) for r in valid_sub()) + "\n").encode("utf-8"))
    proc = run_validate(sub_path, DATA_INPUT_N50)
    check("T24 UTF-8 BOM stripped, still passes", proc.returncode == 0, f"rc={proc.returncode}")

    # T25 full 2354-line validation of a valid submission (scale check)
    sub_full = os.path.join(tmp, "sub_full.jsonl")
    write_jsonl(sub_full, [json.loads(l) for l in open(DATA_GOLD_FULL, encoding="utf-8")])
    proc = run_validate(sub_full, os.path.join(ROOT, "data", "val_input.jsonl"))
    check("T25 full 2354-line valid submission passes", proc.returncode == 0, f"rc={proc.returncode} {proc.stdout[-200:]}")

    # T36 duplicate id in the INPUT file -> exit 2 (pipeline bug, not submission bug)
    input_dup = os.path.join(tmp, "input_dup.jsonl")
    src_lines = open(DATA_INPUT_N50, encoding="utf-8").read().splitlines()
    write_raw(input_dup, src_lines[0] + "\n" + "\n".join(src_lines) + "\n")
    sub_ok = os.path.join(tmp, "sub_ok.jsonl")
    write_jsonl(sub_ok, valid_sub())
    proc = run_validate(sub_ok, input_dup)
    check("T36 duplicate input id -> exit 2", proc.returncode == 2 and "duplicate ids" in proc.stderr,
          f"rc={proc.returncode} stderr={proc.stderr[:150]}")

    # T37 empty submission -> count mismatch -> exit 1
    sub_empty = os.path.join(tmp, "sub_empty.jsonl")
    write_raw(sub_empty, "")
    proc = run_validate(sub_empty, DATA_INPUT_N50)
    check("T37 empty submission rejected (count mismatch)", proc.returncode == 1 and "1 count mismatch" in proc.stdout,
          f"rc={proc.returncode}")

    # T39 unhashable submission id (list) -> structural violation, NOT a crash
    def id_list(r): r["id"] = ["a", "b"]
    sub_bad = os.path.join(tmp, "sub_bad.jsonl")
    write_jsonl(sub_bad, mutate(valid_sub(), id_list))
    proc = run_validate(sub_bad, DATA_INPUT_N50)
    check("T39 list-as-id rejected without crash (exit 1)", proc.returncode == 1 and "1 structural" in proc.stdout,
          f"rc={proc.returncode} stderr={proc.stderr[:200]}")

    # T40 non-string id in INPUT -> exit 2 (pipeline bug)
    input_bad = os.path.join(tmp, "input_bad.jsonl")
    src_lines = open(DATA_INPUT_N50, encoding="utf-8").read().splitlines()
    first = json.loads(src_lines[0]); first["id"] = ["x"]
    write_raw(input_bad, json.dumps(first, ensure_ascii=False) + "\n" + "\n".join(src_lines[1:]) + "\n")
    proc = run_validate(sub_ok, input_bad)
    check("T40 non-string input id -> exit 2", proc.returncode == 2 and "non-string ids" in proc.stderr,
          f"rc={proc.returncode} stderr={proc.stderr[:200]}")

    # T38 empty input -> exit 2
    input_empty = os.path.join(tmp, "input_empty.jsonl")
    write_raw(input_empty, "")
    proc = run_validate(sub_ok, input_empty)
    check("T38 empty input -> exit 2", proc.returncode == 2 and "no records" in proc.stderr,
          f"rc={proc.returncode} stderr={proc.stderr[:150]}")

    # E15: history header gains bleu4/rouge_l/bertscore; old-format table archived
    old_hist = os.path.join(tmp, "old-history.md")
    with open(old_hist, "w", encoding="utf-8") as f:
        f.write("| date | submission | emotion acc | macro-F1 | profile micro-F1 (avg) | always-anxiety baseline | delta |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        f.write("| 2026-10-01 | old_sub.jsonl | 0.3000 | 0.1000 | 0.2000 | 0.3190 | -0.0190 |\n")
    sub_e15 = os.path.join(tmp, "e15.jsonl")
    write_jsonl(sub_e15, valid_sub())
    out_h = os.path.join(tmp, "e15.metrics.json")
    proc = subprocess.run([sys.executable, EVAL, sub_e15, DATA_GOLD_N50, "--out", out_h, "--history", old_hist],
                          capture_output=True, text=True, timeout=300)
    head = open(old_hist, encoding="utf-8").readline()
    archived_ok = os.path.exists(old_hist + ".v1.md") and "old_sub.jsonl" in open(old_hist + ".v1.md", encoding="utf-8").read()
    check("E15 new history header has bleu4/rouge_l/bertscore columns",
          proc.returncode == 0 and "bleu4" in head and "bertscore" in head, f"head={head.strip()}")
    check("E15 old-format table archived intact (no experiment lost)",
          archived_ok)

    # E16: order invariance -- shuffled submission must give identical metrics
    # (excluding the path metadata fields, which legitimately differ)
    always_fixture = [{
        "id": r["id"], "response_text": "我理解你的感受，我会一直陪着你。",
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    } for r in gold_full]
    import random as _rnd
    _rnd.seed(7)
    shuffled = always_fixture[:]
    _rnd.shuffle(shuffled)
    sub_base = os.path.join(tmp, "e16_base.jsonl")
    write_jsonl(sub_base, always_fixture)
    out_base = os.path.join(tmp, "e16_base.metrics.json")
    subprocess.run([sys.executable, EVAL, sub_base, DATA_GOLD_FULL, "--out", out_base, "--history", ""],
                   capture_output=True, text=True, timeout=300)
    sub_shuf = os.path.join(tmp, "shuffled.jsonl")
    write_jsonl(sub_shuf, shuffled)
    out_shuf = os.path.join(tmp, "shuffled.metrics.json")
    subprocess.run([sys.executable, EVAL, sub_shuf, DATA_GOLD_FULL, "--out", out_shuf, "--history", ""],
                   capture_output=True, text=True, timeout=300)
    ma = json.load(open(out_base, encoding="utf-8"))
    mb = json.load(open(out_shuf, encoding="utf-8"))
    for mdict in (ma, mb):
        mdict.pop("submission", None)
        mdict.pop("gold", None)
    check("E16 metrics byte-identical under submission reordering", ma == mb)

    print()
    print("== eval.py smoke tests ==")

    # E01: gold-as-submission -> perfect scores
    sub_path = os.path.join(tmp, "perfect.jsonl")
    write_jsonl(sub_path, gold_full)
    out_json = os.path.join(tmp, "perfect.metrics.json")
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", ""],
                          capture_output=True, text=True, timeout=300)
    m = json.load(open(out_json, encoding="utf-8"))
    check("E01 gold-as-submission: accuracy 1.0", proc.returncode == 0 and m["emotion"]["accuracy"] == 1.0,
          f"rc={proc.returncode} acc={m['emotion']['accuracy']}")
    check("E01 gold-as-submission: macro-F1 1.0", m["emotion"]["macro_f1"] == 1.0, f"macro={m['emotion']['macro_f1']}")
    check("E01 gold-as-submission: profile micro-F1 1.0", m["profile"]["micro_f1_avg"] == 1.0,
          f"profile={m['profile']['micro_f1_avg']}")
    check("E01 gold-as-submission: profile macro-F1 1.0 (zero-support labels skipped)",
          m["profile"]["macro_f1_avg"] == 1.0, f"macro={m['profile']['macro_f1_avg']}")
    check("E01 per-class F1 all 1.0", all(s["f1"] == 1.0 for s in m["emotion"]["per_class"].values()))
    check("E01 gold-as-submission: BLEU-4 1.0 and ROUGE-L 1.0",
          m["response"]["bleu4"] == 1.0 and m["response"]["rouge_l"] == 1.0,
          f"bleu={m['response']['bleu4']} rouge={m['response']['rouge_l']}")

    # E02: always-anxiety + empty profile -> accuracy == baseline, delta 0
    sub_path = os.path.join(tmp, "anxiety.jsonl")
    always = [{
        "id": r["id"], "response_text": "我理解你的感受。",
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    } for r in gold_full]
    write_jsonl(sub_path, always)
    out_json = os.path.join(tmp, "anxiety.metrics.json")
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", ""],
                          capture_output=True, text=True, timeout=300)
    m = json.load(open(out_json, encoding="utf-8"))
    expected_acc = sum(1 for r in gold_full if r["emotion_label"] == "anxiety") / len(gold_full)
    check("E02 always-anxiety accuracy matches theory", abs(m["emotion"]["accuracy"] - round(expected_acc, 4)) < 1e-9,
          f"acc={m['emotion']['accuracy']} expected={expected_acc:.4f}")
    check("E02 delta vs baseline is 0", abs(m["emotion"]["accuracy"] - m["baselines"]["always_anxiety_accuracy"]) < 1e-9)
    # empty predictions only match empty gold -> exact-match MUST equal empty_gold_rate
    check("E02 empty profile exact-match == empty_gold_rate (style field)",
          abs(m["profile"]["style"]["exact_match"] - m["profile"]["style"]["empty_gold_rate"]) < 1e-9,
          f"exact={m['profile']['style']['exact_match']} empty_rate={m['profile']['style']['empty_gold_rate']}")
    check("E02 report prints confusion pairs", "top confusion pairs" in proc.stdout)
    check("E02 fixed short reply: BLEU-4 low but non-negative / ROUGE-L bounded",
          0.0 <= m["response"]["bleu4"] < 0.15 and 0.0 <= m["response"]["rouge_l"] < 0.6,
          f"bleu={m['response']['bleu4']} rouge={m['response']['rouge_l']}")

    # E03: id mismatch -> exit 1
    bad = always[:-1]
    write_jsonl(sub_path, bad)
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", ""],
                          capture_output=True, text=True, timeout=300)
    check("E03 id mismatch refuses to score (exit 1)", proc.returncode == 1 and "validate.py" in proc.stderr,
          f"rc={proc.returncode}")

    # E04: gate refuses to score invalid emotion labels without --force
    broken = [dict(r, emotion_label="not_an_emotion") for r in always[:200]]
    write_jsonl(sub_path, broken + always[200:])
    out_e04 = os.path.join(tmp, "e04.metrics.json")
    if os.path.exists(out_e04):
        os.remove(out_e04)
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_e04, "--history", ""],
                          capture_output=True, text=True, timeout=300)
    check("E04 gate refuses invalid labels (exit 1, no metrics written)",
          proc.returncode == 1 and "refusing to score" in proc.stderr and not os.path.exists(out_e04),
          f"rc={proc.returncode} stderr={proc.stderr[:200]}")
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_e04, "--history", "", "--force"],
                          capture_output=True, text=True, timeout=300)
    m = json.load(open(out_e04, encoding="utf-8"))
    check("E04 --force: invalid labels counted as wrong (200 flagged)",
          proc.returncode == 0 and m["emotion"]["n_invalid_predictions"] == 200,
          f"rc={proc.returncode} n={m['emotion']['n_invalid_predictions']}")

    # E06: gate refuses malformed user_profile without --force; --force scores it
    broken = [dict(r, user_profile="garbage") for r in always[:200]]
    write_jsonl(sub_path, broken + always[200:])
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", ""],
                          capture_output=True, text=True, timeout=300)
    check("E06 gate refuses malformed user_profile (exit 1)", proc.returncode == 1,
          f"rc={proc.returncode}")
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", "", "--force"],
                          capture_output=True, text=True, timeout=300)
    check("E06 --force scores through malformed profiles (exit 0)", proc.returncode == 0, f"rc={proc.returncode}")

    # E10: BERTScore degrades gracefully when weights are missing/unavailable
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json,
                           "--history", "", "--force", "--bert-model", os.path.join(tmp, "no_such_model")],
                          capture_output=True, text=True, timeout=300)
    m = json.load(open(out_json, encoding="utf-8"))
    check("E10 BERTScore skipped cleanly on missing weights (exit 0, null, notice printed)",
          proc.returncode == 0 and m["response"]["bertscore"] is None and "skipped" in proc.stdout,
          f"rc={proc.returncode} bs={m['response']['bertscore']}")

    # E11/E12/E13: pure-math checks (need numpy; skipped gracefully without it)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    ev = importlib.import_module("eval")
    try:
        import numpy as np
    except ImportError:
        np = None
        print("  [SKIP] E11/E12/E13 -- numpy not installed on this machine")
    if np is not None:
        # synthetic matrices carry no [CLS]/[SEP] semantics -> drop_special=False
        ident = ev._greedy_cosine_f1(np.eye(2), np.eye(2), drop_special=False)
        # implementation carries a deliberate 1e-8 normalization epsilon -> tolerance 1e-6
        check("E11 identical token sets -> P=R=F1=1.0",
              abs(ident[0]-1) < 1e-6 and abs(ident[1]-1) < 1e-6 and abs(ident[2]-1) < 1e-6, f"{ident}")
        orth = ev._greedy_cosine_f1(np.array([[1.0, 0]]), np.array([[0, 1.0]]), drop_special=False)
        check("E11 orthogonal tokens -> 0.0",
              orth == (0.0, 0.0, 0.0), f"{orth}")
        mixed = ev._greedy_cosine_f1(np.array([[1.0, 0], [1.0, 0]]), np.eye(2), drop_special=False)
        # P: both pred rows best-match gold[0] -> 1.0 ; R: gold[0] matched, gold[1] not -> 0.5 ; F1 = 2/3
        check("E11 mixed case -> P 1.0, R 0.5, F1 2/3",
              abs(mixed[0]-1) < 1e-6 and abs(mixed[1]-0.5) < 1e-6 and abs(mixed[2]-2/3) < 1e-6, f"{mixed}")

        bleu_short = ev.bleu4_corpus(["我很好啊"], ["我很好啊我很好啊"])
        check("E12 BLEU exact hand case (brevity penalty only) = exp(-1) = 0.3679",
              abs(bleu_short - round(math.exp(-1), 4)) < 1e-9, f"{bleu_short}")

        # E13: [CLS],A,[SEP] vs [CLS],C,[SEP] (real tokenizer order) with A
        # orthogonal to C: including specials gives F1 2/3; excluding gives 0.0
        emb_p = np.array([[1., 0, 0, 0], [0, 0, 1., 0], [0, 1., 0, 0]])
        emb_g = np.array([[1., 0, 0, 0], [0, 0, 0, 1.], [0, 1., 0, 0]])
        full = ev._greedy_cosine_f1(emb_p, emb_g, drop_special=False)
        stripped = ev._greedy_cosine_f1(emb_p, emb_g, drop_special=True)
        check("E13 special tokens included -> F1 2/3 (the inflation artifact)",
              abs(full[2] - 2/3) < 1e-6, f"{full}")
        check("E13 special tokens excluded -> F1 0.0 (bert-score behavior)",
              stripped == (0.0, 0.0, 0.0), f"{stripped}")

        # E17: a blank reply is only [CLS][SEP]; it must score 0, not match on specials
        blank = ev._greedy_cosine_f1(np.array([[1., 0], [0, 1.]]),
                                     np.array([[1., 0], [.5, .5], [0, 1.]]), drop_special=True)
        check("E17 blank reply ([CLS][SEP] only) -> BERTScore 0.0", blank == (0.0, 0.0, 0.0), f"{blank}")

    # E18: bit-parallel LCS == textbook DP (random strings, incl. empty and repeats)
    def lcs_dp(a, b):
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b):
                cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
            prev = cur
        return prev[-1]
    import random as _r
    _r.seed(11)
    alphabet = "我你他好的了是不在有"
    lcs_ok = all(
        ev._lcs_len(a, b) == lcs_dp(a, b)
        for a, b in [("", ""), ("我", ""), ("", "我")] + [
            ("".join(_r.choice(alphabet) for _ in range(_r.randint(0, 80))),
             "".join(_r.choice(alphabet) for _ in range(_r.randint(0, 80))))
            for _ in range(300)])
    check("E18 bit-parallel LCS equals the DP on 303 random pairs", lcs_ok)

    # E19: a second header change must not overwrite the first archive
    hist19 = os.path.join(tmp, "h19.md")
    old_table = "| date | submission | emotion acc |\n|---|---|---|\n| d | {} | 0.1 |\n"
    write_raw(hist19, old_table.format("first"))
    write_raw(hist19 + ".v1.md", old_table.format("archived_earlier"))
    proc = subprocess.run([sys.executable, EVAL, sub_e15, DATA_GOLD_N50, "--out", out_h, "--history", hist19],
                          capture_output=True, text=True, timeout=300)
    check("E19 archive never overwrites an older archive (.v1 kept, .v2 created)",
          proc.returncode == 0
          and "archived_earlier" in open(hist19 + ".v1.md", encoding="utf-8").read()
          and "first" in open(hist19 + ".v2.md", encoding="utf-8").read(),
          f"rc={proc.returncode} stderr={proc.stderr[:200]}")

    # E07/E08: list-as-ID through eval.py must exit cleanly (no traceback)
    bad_id = [dict(r, id=["a", "b"]) for r in always[:5]]
    write_jsonl(sub_path, bad_id + always[5:])
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", ""],
                          capture_output=True, text=True, timeout=300)
    check("E07 gate refuses list-ID cleanly (exit 1, no traceback)",
          proc.returncode == 1 and "refusing to score" in proc.stderr and "Traceback" not in proc.stderr,
          f"rc={proc.returncode} stderr={proc.stderr[:200]}")
    proc = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", "", "--force"],
                          capture_output=True, text=True, timeout=300)
    check("E08 --force with list-ID exits 1 via load guard (no traceback)",
          proc.returncode == 1 and "must be a string" in proc.stderr and "Traceback" not in proc.stderr,
          f"rc={proc.returncode} stderr={proc.stderr[:200]}")

    # E05: history append
    hist = os.path.join(tmp, "score-history.md")
    write_jsonl(sub_path, always)
    p1 = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", hist],
                        capture_output=True, text=True, timeout=300)
    p2 = subprocess.run([sys.executable, EVAL, sub_path, DATA_GOLD_FULL, "--out", out_json, "--history", hist],
                        capture_output=True, text=True, timeout=300)
    check("E05 both eval invocations succeed (rc=0)", p1.returncode == 0 and p2.returncode == 0,
          f"rc={p1.returncode},{p2.returncode}")
    content = open(hist, encoding="utf-8").read()
    data_rows = [ln for ln in content.splitlines() if ln.startswith("|") and sub_path in ln]
    check("E05 history: header once + exactly 2 data rows for this submission",
          content.count("emotion acc") == 1 and len(data_rows) == 2,
          f"header_count={content.count('emotion acc')} data_rows={len(data_rows)}")

    print()
    failed = [n for n, ok in results if not ok]
    print(f"{'ALL TESTS PASSED' if not failed else 'FAILURES: ' + ', '.join(failed)}  ({len(results) - len(failed)}/{len(results)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
