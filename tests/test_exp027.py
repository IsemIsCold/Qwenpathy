#!/usr/bin/env python3
"""
tests/test_exp027.py -- EXP-027 (submission #4 candidate) offline tests. CPU only, stdlib + numpy, ~3 min.

    python tests/test_exp027.py [--old experiments/exp027-early-stop/run_inference_sub3.py]

--old = submission #3's run_inference.py (sha256 ec3cfdf5...). Needs participant/profile_clf/, the saved answers
outputs/exp022/lp_bf16.jsonl + outputs/exp023/lp_fp8_t10.jsonl, data/t10_input.jsonl and the n100/tune500 gold.
Exit code 0 = all checks pass. The session-script part (E) needs bash.

A. Lenient parse: on all 1,200 saved answers the new parser = #3's parser, except the 2 broken FP8 answers,
   which now keep their reply and label.
B. Cut at "user_profile": on all 1,200 answers the closed cut keeps reply + label exactly.
C. Replay of the FP8 answers through run_inference.py on the stub vLLM (600 points):
   C1 #3's config: new file = #3's file except the 2 recovered answers;
   C2 candidate (exp027_install.py --variant exact): every request carries the stop, 600/600 cut, 0 fallbacks,
      reply + emotion = C1, profile = classifier alone at the exact thresholds (per-field exact 0.6522);
   C3 20% of answers write the profile first -> exactly those fall back to a full answer, same reply + emotion.
D. Config checks; exp027_install.py backup / restore / refusal.
E. scripts/exp027_session.sh with a fake vLLM: PASS keeps the candidate; FAIL restores #3 byte for byte.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FP8 = ROOT / "outputs/exp023/lp_fp8_t10.jsonl"
BF16 = ROOT / "outputs/exp022/lp_bf16.jsonl"
T10 = ROOT / "data/t10_input.jsonl"
GOLDS = [ROOT / "data/val_gold_n100.jsonl", ROOT / "data/val_gold_tune500.jsonl"]
BROKEN = {"val_conv_000383_t7", "val_conv_000755_t7"}
STUB = ROOT / "tests/stub_vllm.py"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Stub:
    def __init__(self, env):
        self.port = free_port()
        self.proc = subprocess.Popen([sys.executable, str(STUB), "--port", str(self.port)],
                                     env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("stub did not start")

    def stop(self):
        self.proc.terminate()
        self.proc.wait()


def make_participant(tmp, name, run_inference, config):
    d = tmp / name
    (d / "configs").mkdir(parents=True)
    shutil.copy2(run_inference, d / "run_inference.py")
    shutil.copytree(ROOT / "participant/profile_clf", d / "profile_clf")
    (d / "configs/inference_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return d


def run_ri(pdir, test_file, out, stub, quiet=False):
    env = {**os.environ, "EVALUATION_BACKEND": "openai_compatible", "EVALUATION_BASE_URL": f"http://127.0.0.1:{stub.port}/v1",
           "EVALUATION_API_MODEL": "model", "EVALUATION_HARDWARE_LABEL": "test", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(pdir / "run_inference.py"), str(test_file), str(out)],
                       capture_output=True, text=True, env=env)
    if r.returncode and not quiet:
        print(r.stdout[-2000:], r.stderr[-3000:])
    return r


def sub3_config():
    cfg = json.loads((ROOT / "participant/configs/inference_config.json").read_text(encoding="utf-8"))
    for k in ("stop_after_emotion", "profile_blend"):
        cfg.pop(k, None)
    cfg["profile_classifier"] = "profile_clf"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(ROOT / "experiments/exp027-early-stop/run_inference_sub3.py"))
    args = ap.parse_args()
    new = load("ri_new", ROOT / "participant/run_inference.py")
    old = load("ri_old", args.old)
    ev = load("ev", ROOT / "scripts/eval.py")
    print(f"old run_inference {sha(args.old)[:12]}, new {sha(ROOT / 'participant/run_inference.py')[:12]}")
    gold = {r["id"]: r for g in GOLDS for r in read_jsonl(g)}
    answers = {"fp8": {r["id"]: r["content"] for r in read_jsonl(FP8)}, "bf16": {r["id"]: r["content"] for r in read_jsonl(BF16)}}

    print("A. lenient parse")
    diff = {}
    for name, rows in answers.items():
        for i, c in rows.items():
            if new.parse_prediction(i, c) != old.parse_prediction(i, c):
                diff[(name, i)] = (old.parse_prediction(i, c)[0], new.parse_prediction(i, c)[0])
    check("A1 new parser = #3's on 1,200 answers except the 2 broken FP8 ones", set(diff) == {("fp8", i) for i in BROKEN},
          f"differences: {sorted(i for _, i in diff)}")
    rec = {i: diff[("fp8", i)][1] for i in BROKEN if ("fp8", i) in diff}
    check("A2 the 2 broken answers keep reply + label (755: mixed = gold)", len(rec) == 2
          and all(len(p["response_text"]) > 100 for p in rec.values())
          and rec["val_conv_000755_t7"]["emotion_label"] == "mixed" and rec["val_conv_000383_t7"]["emotion_label"] == "anxiety",
          ", ".join(f"{i[-10:]}: {p['emotion_label']} / {len(p['response_text'])} chars" for i, p in rec.items()))
    check("A3 under #3's parser they had lost the reply", all(diff[("fp8", i)][0]["response_text"] == "" for i in BROKEN))
    n_ok = sum(new.parse_prediction(i, c)[0]["emotion_label"] == gold[i]["emotion_label"] for i, c in answers["fp8"].items())
    check("A4 FP8 emotion on 600 with the new parser = 473", n_ok == 473, f"{n_ok}/600")

    print("B. cut at the stop string")
    bad = n = 0
    for name, rows in answers.items():
        for i, c in rows.items():
            k = c.find(new.STOP_AFTER_EMOTION)
            closed = new.close_after_emotion(c[:k]) if k >= 0 else None
            full = new.parse_prediction(i, c)[0]
            cut = new.parse_prediction(i, closed)[0] if closed else None
            n += 1
            bad += not (cut and cut["response_text"] == full["response_text"] and cut["emotion_label"] == full["emotion_label"])
    check("B1 cut + close keeps reply and label on all 1,200 answers", bad == 0, f"{n - bad}/{n}")
    check("B2 a cut without reply or label returns None", new.close_after_emotion('{"response_text": "好", ') is None
          and new.close_after_emotion('{"emotion_label": "joy", ') is None)

    tmp = Path(tempfile.mkdtemp(prefix="exp027_test_"))
    try:
        print("C. replay on the stub vLLM (600 points)")
        stub_log = tmp / "stub_log.jsonl"
        stub = Stub({"STUB_REPLAY": str(FP8), "STUB_GOLD": ",".join(map(str, GOLDS)), "STUB_LOG": str(stub_log)})
        try:
            p_old = make_participant(tmp, "p_old", args.old, sub3_config())
            p_new = make_participant(tmp, "p_new", ROOT / "participant/run_inference.py", sub3_config())
            r1, r2 = run_ri(p_old, T10, tmp / "o_old", stub), run_ri(p_new, T10, tmp / "o_new", stub)
            s_old = {r["id"]: r for r in read_jsonl(tmp / "o_old/submission.jsonl")}
            s_new = {r["id"]: r for r in read_jsonl(tmp / "o_new/submission.jsonl")}
            d = {i for i in s_old if s_old[i] != s_new[i]}
            check("C1 #3's config: new file = #3's file except the 2 recovered answers", r1.returncode == 0 and r2.returncode == 0
                  and d <= BROKEN and len(s_new) == 600, f"differ: {sorted(d)}")
            rep_old = json.loads((tmp / "o_old/performance_report.json").read_text())
            rep_new = json.loads((tmp / "o_new/performance_report.json").read_text())
            check("C1b performance_report.json has the same keys", set(rep_old) == set(rep_new))

            inst = subprocess.run([sys.executable, str(ROOT / "scripts/exp027_install.py"), "--participant", str(p_new),
                                   "--backup", str(tmp / "bk"), "--new-run-inference", str(ROOT / "participant/run_inference.py"),
                                   "--variant", "exact"], capture_output=True, text=True)
            check("C2a install --variant exact", inst.returncode == 0, inst.stdout.strip().splitlines()[-1] if inst.stdout.strip() else inst.stderr[-300:])
            stub_log.write_text("")
            r3 = run_ri(p_new, T10, tmp / "o_cand", stub)
            s_c = {r["id"]: r for r in read_jsonl(tmp / "o_cand/submission.jsonl")}
            reqs = read_jsonl(stub_log)
            check("C2b every request carries the stop string", len(reqs) == 600 and all("stop" in q["keys"] for q in reqs), f"{len(reqs)} requests")
            check("C2c 600/600 cut, 0 fallbacks", "stop after emotion: cut 600/600, full-output fallbacks 0" in r3.stdout,
                  next((l for l in r3.stdout.splitlines() if l.startswith("stop after")), "line missing"))
            same = all(s_c[i]["response_text"] == s_new[i]["response_text"] and s_c[i]["emotion_label"] == s_new[i]["emotion_label"] for i in s_new)
            check("C2d reply + emotion identical to the full answers", same)
            ids = list(s_c)
            prof = ev.score_profile({f: [(ev.to_label_set(s_c[i]["user_profile"][f]), ev.to_label_set(gold[i]["user_profile"][f])) for i in ids]
                                     for f in ev.PROFILE_FIELDS})
            check("C2e profile = classifier alone at the exact thresholds: per-field exact 0.6522", abs(prof["exact_match_avg"] - 0.6522) < 1e-4,
                  f"{prof['exact_match_avg']:.4f}")
            out = json.loads((tmp / "o_cand/performance_report.json").read_text())
            check("C2f parse failures 0, complete", out["parse_failure_count"] == 0 and out["complete"])
        finally:
            stub.stop()
        stub = Stub({"STUB_REPLAY": str(FP8), "STUB_GOLD": ",".join(map(str, GOLDS)), "STUB_PROFILE_FIRST": "0.2"})
        try:
            r4 = run_ri(p_new, T10, tmp / "o_pf", stub)
            s_pf = {r["id"]: r for r in read_jsonl(tmp / "o_pf/submission.jsonl")}
            line = next((l for l in r4.stdout.splitlines() if l.startswith("stop after")), "")
            n_fb = int(line.rsplit(" ", 1)[-1]) if line else -1
            same = all(s_pf[i]["response_text"] == s_new[i]["response_text"] and s_pf[i]["emotion_label"] == s_new[i]["emotion_label"] for i in s_new)
            check("C3 profile-first answers fall back to a full answer (≈ 20%), reply + emotion unchanged", 80 < n_fb < 160 and same, line)
        finally:
            stub.stop()

        print("D. config and install checks")
        for bad_cfg, why in (({"stop_after_emotion": "yes"}, "stop_after_emotion"), ({"profile_blend": {"style": [0, 0.3]}}, "profile_blend"),
                             ({"profile_blend": {"personality_traits": [0, 2], "interests": [0, 0.3], "style": [0, 0.3]}}, "profile_blend.personality")):
            pd = make_participant(tmp, f"p_bad_{why}", ROOT / "participant/run_inference.py", {**sub3_config(), **bad_cfg})
            stub = Stub({"STUB_REPLAY": str(FP8), "STUB_GOLD": ",".join(map(str, GOLDS))})
            try:
                rb = run_ri(pd, ROOT / "data/val_input_n50.jsonl", tmp / f"o_bad_{why}", stub, quiet=True)
            finally:
                stub.stop()
            check(f"D1 bad {why} rejected at start-up", rb.returncode != 0 and why.split(".")[0] in rb.stderr, rb.stderr.strip().splitlines()[-1][:120] if rb.stderr.strip() else "")
        no_clf = {k: v for k, v in sub3_config().items() if k != "profile_classifier"}
        pd = make_participant(tmp, "p_noclf", ROOT / "participant/run_inference.py", {**no_clf, "stop_after_emotion": True})
        rb = subprocess.run([sys.executable, str(pd / "run_inference.py"), str(ROOT / "data/val_input_n50.jsonl"), str(tmp / "o_noclf")],
                            capture_output=True, text=True, env={**os.environ, "EVALUATION_BACKEND": "openai_compatible",
                                                                 "EVALUATION_BASE_URL": "http://127.0.0.1:9/v1", "EVALUATION_API_MODEL": "model"})
        check("D2 stop_after_emotion without the classifier rejected", rb.returncode != 0 and "profile_classifier" in rb.stderr)
        before = {rel: sha(tmp / "bk" / rel) for rel in ("run_inference.py", "configs/inference_config.json")}
        again = subprocess.run([sys.executable, str(ROOT / "scripts/exp027_install.py"), "--participant", str(p_new), "--backup", str(tmp / "bk"),
                                "--new-run-inference", str(ROOT / "participant/run_inference.py"), "--variant", "f1"], capture_output=True, text=True)
        check("D3 second install keeps the existing backup", again.returncode == 0 and "backup exists" in again.stdout
              and before == {rel: sha(tmp / "bk" / rel) for rel in before})
        rs = subprocess.run([sys.executable, str(ROOT / "scripts/exp027_install.py"), "--participant", str(p_new), "--backup", str(tmp / "bk"), "--restore"],
                            capture_output=True, text=True)
        check("D4 restore = #3's files byte for byte", rs.returncode == 0 and sha(p_new / "run_inference.py") == sha(ROOT / "participant/run_inference.py")
              and json.loads((p_new / "configs/inference_config.json").read_text()) == sub3_config())
        p_x = make_participant(tmp, "p_x", ROOT / "participant/run_inference.py", no_clf)
        rx = subprocess.run([sys.executable, str(ROOT / "scripts/exp027_install.py"), "--participant", str(p_x), "--backup", str(tmp / "bk_x"),
                             "--new-run-inference", str(ROOT / "participant/run_inference.py"), "--variant", "exact"], capture_output=True, text=True)
        check("D5 install refuses files that are not #3 (no classifier)", rx.returncode != 0 and not (tmp / "bk_x").exists())

        if shutil.which("bash"):
            print("E. exp027_session.sh with a fake vLLM")
            session_tests(tmp, args.old)
        else:
            print("E. skipped (no bash)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


def session_tests(tmp, old):
    bin_ = tmp / "bin"
    bin_.mkdir()
    (bin_ / "activate").write_text("# fake conda activate\n")
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 'GPU 0: fake'\necho 'GPU 1: fake'\n")
    (bin_ / "vllm").write_text(f"#!/bin/sh\nexec {sys.executable} {STUB} \"$@\"\n")
    for f in ("nvidia-smi", "vllm"):
        (bin_ / f).chmod(0o755)
    V = tmp / "val"
    V.mkdir()
    shutil.copy2(ROOT / "data/val_input_n100.jsonl", V / "t9_input.jsonl")
    shutil.copy2(ROOT / "data/val_gold_n100.jsonl", V / "val_gold_n100.jsonl")
    U = tmp / "U"
    for rel in ("scripts/exp027_session.sh", "scripts/exp027_install.py", "scripts/validate.py", "scripts/eval.py", "participant/run_inference.py"):
        (U / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, U / rel)

    def fresh_p(name):
        p = make_participant(tmp, name, old, sub3_config())
        shutil.copy2(ROOT / "participant/start.sh", p / "start.sh")
        (p / "model").mkdir()
        (p / "model/config.json").write_text("{}")
        return p

    def run(p, out, extra):
        env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "P": str(p), "U": str(U), "V": str(V), "OUT": str(out),
               "BACKUP": str(out) + "_bk", "PY": sys.executable, "PYTHON_BIN": sys.executable, "EXP027_TEST": "1",
               "EXP027_EXPECT_SHA": "0", "STUB_REPLAY": str(FP8), "STUB_GOLD": ",".join(map(str, GOLDS)),
               "VLLM_PORT": str(free_port()), "VLLM_SERVED_MODEL_NAME": "model", **extra}
        return subprocess.run(["bash", str(U / "scripts/exp027_session.sh")], capture_output=True, text=True, env=env)

    p = fresh_p("sess_pass")
    r = run(p, tmp / "sess_pass_out", {"STUB_SLEEP_PER_CHAR": "0.00002"})
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:]
    check("E1 PASS path ends with DONE -- PASS", "DONE -- PASS" in r.stdout, tail)
    cfg = json.loads((p / "configs/inference_config.json").read_text())
    check("E2 PASS keeps the candidate installed (stop + exact thresholds on #3's config)", cfg.get("stop_after_emotion") is True
          and cfg.get("profile_blend", {}).get("interests") == [0.0, 0.30] and cfg.get("profile_classifier") == "profile_clf")
    gate = json.loads((tmp / "sess_pass_out/gate.json").read_text()) if (tmp / "sess_pass_out/gate.json").exists() else {}
    check("E3 gate: same emotion as run A, higher profile exact, faster", gate.get("pass") and gate["run_b"]["acc"] == gate["run_a"]["acc"]
          and gate["run_b"]["exact"] > gate["run_a"]["exact"] and gate["run_b"]["lat"] < gate["run_a"]["lat"],
          f"A {gate.get('run_a', {}).get('exact')} / {gate.get('run_a', {}).get('lat')} s, B {gate.get('run_b', {}).get('exact')} / {gate.get('run_b', {}).get('lat')} s" if gate else "no gate.json")
    if "DONE -- PASS" not in r.stdout:
        print(r.stdout[-3000:], r.stderr[-1500:])

    p = fresh_p("sess_fail")
    h_before = {f: sha(p / f) for f in ("run_inference.py", "configs/inference_config.json")}
    r = run(p, tmp / "sess_fail_out", {"STUB_PROFILE_FIRST": "0.5"})
    check("E4 FAIL path (50% fallbacks) ends with DONE -- FAIL and restores #3 byte for byte", "DONE -- FAIL (submission #3 restored)" in r.stdout
          and h_before == {f: sha(p / f) for f in h_before}, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
    p = fresh_p("sess_missing")
    (p / "profile_clf/profile_clf.json").unlink()
    r = run(p, tmp / "sess_missing_out", {})
    check("E5 missing file stops before anything changes", "MISSING" in r.stdout and "nothing changed" in r.stdout)


if __name__ == "__main__":
    main()
