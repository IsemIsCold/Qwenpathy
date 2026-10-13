#!/usr/bin/env python3
"""
tests/test_profile_blend.py -- EXP-025: the profile classifier + blend inside participant/run_inference.py.

CPU only, no GPU, no sklearn needed. Needs numpy, the EXP-022 outputs (outputs/exp022/) and the
data slices. Run from the repo root:
    python tests/test_profile_blend.py [--old <submission #2 run_inference.py>]
Exit code 0 = all checks pass.

1. numpy classifier == the sklearn model it was exported from (reference probabilities saved
   by EXP-024 on the 600 scored val points), and no blend decision flips.
2. Replay of submission #2's real raw answers (lp_bf16.jsonl) through run_inference.py on a stub
   server: with the key, profile micro-F1 on the 600 points = the EXP-024/025 number; without it,
   identical to the old parser (and byte-identical to --old if given).
3. Config checks, per-sample overhead.
4. scripts/exp025_session.sh with a fake vLLM: PASS path keeps the candidate, FAIL path restores #2.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLF_DIR = ROOT / "participant" / "profile_clf"
REF = ROOT / "experiments" / "exp024-cpu-classifier" / "P_val_reference.npy"
LP = ROOT / "outputs" / "exp022" / "lp_bf16.jsonl"
OLD_RI = Path(sys.argv[sys.argv.index("--old") + 1]).resolve() if "--old" in sys.argv else None
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def load_module(name, path):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def profile_scores(ev, sub_path, gold_rows):
    sub = {r["id"]: r for r in read_jsonl(sub_path)}
    pairs = {f: [(ev.to_label_set(sub[g["id"]]["user_profile"].get(f)), ev.to_label_set((g.get("user_profile") or {}).get(f)))
                 for g in gold_rows] for f in ev.PROFILE_FIELDS}
    s = ev.score_profile(pairs)
    return s["micro_f1_avg"], s["exact_match_avg"]


def main():
    ri = load_module("ri", ROOT / "participant/run_inference.py")
    ev = load_module("ev", ROOT / "scripts/eval.py")
    inputs = {r["id"]: r for f in ("val_input_tune500.jsonl", "val_input_n100.jsonl") for r in read_jsonl(DATA / f)}
    gold_t500, gold_n100 = read_jsonl(DATA / "val_gold_tune500.jsonl"), read_jsonl(DATA / "val_gold_n100.jsonl")
    ids = [g["id"] for g in gold_t500] + [g["id"] for g in gold_n100]  # = order of the reference file

    print("\n1. numpy classifier vs the exported sklearn model")
    clf = ri.ProfileClassifier(CLF_DIR)
    t0 = time.perf_counter()
    P = np.vstack([clf.probabilities(inputs[i]) for i in ids])
    ms = (time.perf_counter() - t0) * 1000 / len(ids)
    ref = np.load(REF)
    diff = float(np.abs(P - ref).max())
    check("probabilities match the sklearn reference", diff < 1e-5, f"max |diff| {diff:.2e} over {P.size} values")
    flips = 0
    for f in ri.PROFILE_LABELS:
        a, t = clf.blend[f]
        for j, (ff, _) in enumerate(clf.tags):
            if ff != f:
                continue
            for llm_has in (0, 1):
                flips += int(((a * llm_has + (1 - a) * P[:, j]) >= t).sum() != ((a * llm_has + (1 - a) * ref[:, j]) >= t).sum())
    check("no blend decision flips vs the reference", flips == 0, f"{flips}")
    check("per-sample classifier cost is small", ms < 30, f"{ms:.1f} ms per sample (inside the timed region)")
    check("blend settings = the EXP-024 CV picks",
          clf.blend == {"personality_traits": (0.2, 0.25), "interests": (0.5, 0.15), "style": (0.0, 0.15)}, clf.blend)

    print("\n2. replay of submission #2's raw answers through run_inference.py (stub server)")
    tmp = Path(tempfile.mkdtemp(prefix="exp025_"))
    port = 8021
    env = dict(os.environ, STUB_REPLAY=str(LP), STUB_GOLD=f"{DATA / 'val_gold_n100.jsonl'},{DATA / 'val_gold_tune500.jsonl'}")
    proc = subprocess.Popen([sys.executable, str(ROOT / "tests/stub_vllm.py"), "--port", str(port)], env=env)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        test_file = tmp / "val600.jsonl"
        test_file.write_text("".join(json.dumps(inputs[i], ensure_ascii=False) + "\n" for i in ids), encoding="utf-8")

        def run(name, participant_ri, extra):
            p = tmp / name
            (p / "configs").mkdir(parents=True)
            shutil.copy(participant_ri, p / "run_inference.py")
            if (p / "run_inference.py").read_text(encoding="utf-8").count("ProfileClassifier"):
                shutil.copytree(CLF_DIR, p / "profile_clf")
            cfg = json.loads((ROOT / "participant/configs/inference_config.json").read_text(encoding="utf-8"))
            cfg.update(extra)
            (p / "configs/inference_config.json").write_text(json.dumps(cfg), encoding="utf-8")
            r = subprocess.run([sys.executable, str(p / "run_inference.py"), str(test_file), str(p / "out")],
                               env=dict(os.environ, EVALUATION_BASE_URL=f"http://127.0.0.1:{port}/v1", EVALUATION_API_MODEL="model"),
                               capture_output=True, text=True)
            return r, p / "out" / "submission.jsonl"

        r0, sub0 = run("nokey", ROOT / "participant/run_inference.py", {})
        r1, sub1 = run("blend", ROOT / "participant/run_inference.py", {"profile_classifier": "profile_clf"})
        check("both runs complete", r0.returncode == 0 and r1.returncode == 0, (r0.stderr + r1.stderr)[-300:])
        f0, e0 = profile_scores(ev, sub0, gold_t500 + gold_n100)
        f1, e1 = profile_scores(ev, sub1, gold_t500 + gold_n100)
        check("no key: profile = submission #2's parser (EXP-022 F1 0.4609)", abs(f0 - 0.4609) < 5e-4, f"{f0:.4f}")
        check("blend: profile F1 on 600 = EXP-025 in-sample 0.5055", abs(f1 - 0.5055) < 5e-4, f"{f1:.4f}, exact {e1:.4f} (no key {e0:.4f})")
        t5 = profile_scores(ev, sub1, gold_t500)[0]
        n1 = profile_scores(ev, sub1, gold_n100)[0]
        print(f"         blend tune500 {t5:.4f}  n100 {n1:.4f}")
        s0, s1 = read_jsonl(sub0), read_jsonl(sub1)
        same_rest = all(a["id"] == b["id"] and a["emotion_label"] == b["emotion_label"] and a["response_text"] == b["response_text"]
                        for a, b in zip(s0, s1))
        check("blend changes only user_profile (ids, order, emotion, reply identical)", same_rest and len(s0) == len(s1) == 600)
        check("summary line printed", "profile classifier exp025-v1: tags added" in r1.stdout,
              [l for l in r1.stdout.splitlines() if "profile classifier" in l])
        val = subprocess.run([sys.executable, str(ROOT / "scripts/validate.py"), "--input", str(test_file), str(sub1)],
                             capture_output=True, text=True)
        check("validate.py: 0 errors on the blend output", val.returncode == 0, val.stdout.strip().splitlines()[-1] if val.stdout else val.stderr[-200:])
        rep0 = json.loads((sub0.parent / "performance_report.json").read_text(encoding="utf-8"))
        rep1 = json.loads((sub1.parent / "performance_report.json").read_text(encoding="utf-8"))
        check("performance_report keys unchanged", set(rep0) == set(rep1))
        if OLD_RI:
            ro, subo = run("old", OLD_RI, {})
            check(f"no key: byte-identical to {OLD_RI.name} (old)", ro.returncode == 0 and subo.read_bytes() == sub0.read_bytes())

        print("\n3. config checks")
        for bad in ({"profile_classifier": ""}, {"profile_classifier": 3}, {"profile_classifier": "missing_dir"}):
            rb, _ = run(f"bad{len(results)}", ROOT / "participant/run_inference.py", bad)
            check(f"rejected at start-up: {bad}", rb.returncode != 0)
        print("\n4. exp025_session.sh end to end (fake vllm / nvidia-smi / conda)")
        for name, stub_env, expect in (("pass", {"STUB_REPLAY": str(LP)}, "PASS"), ("fail", {"STUB_REPLAY": ""}, "FAIL")):
            root = tmp / f"session_{name}"
            bindir = root / "bin"
            bindir.mkdir(parents=True)
            (bindir / "vllm").write_text(f'#!/usr/bin/env bash\nexec {sys.executable} {ROOT / "tests/stub_vllm.py"} "$@"\n')
            (bindir / "nvidia-smi").write_text('#!/usr/bin/env bash\necho "GPU 0: fake"; echo "GPU 1: fake"\n')
            (bindir / "activate").write_text("# fake conda\n")
            for f in bindir.iterdir():
                f.chmod(0o755)
            part = root / "participant"
            (part / "configs").mkdir(parents=True)
            (part / "model").mkdir()
            (part / "model/config.json").write_text("{}")
            shutil.copy(OLD_RI or ROOT / "participant/run_inference.py", part / "run_inference.py")
            shutil.copy(ROOT / "participant/start.sh", part / "start.sh")
            shutil.copy(ROOT / "participant/configs/inference_config.json", part / "configs/inference_config.json")
            before = {rel: hashlib.sha256((part / rel).read_bytes()).hexdigest() for rel in ("run_inference.py", "configs/inference_config.json")}
            v = root / "val"
            v.mkdir()
            shutil.copy(DATA / "val_input_n100.jsonl", v / "t9_input.jsonl")
            shutil.copy(DATA / "val_gold_n100.jsonl", v / "val_gold_n100.jsonl")
            r = subprocess.run(["bash", str(ROOT / "scripts/exp025_session.sh")], capture_output=True, text=True,
                               env={**env, "PATH": f"{bindir}:{os.environ['PATH']}", "P": str(part), "U": str(ROOT), "V": str(v),
                                    "OUT": str(root / "out"), "BACKUP": str(root / "backup"), "VLLM_PORT": "8031", **stub_env})
            out = r.stdout
            check(f"{name}: session ends with DONE -- {expect}", r.returncode == 0 and f"DONE -- {expect}" in out, (r.stderr or out)[-500:])
            after = {rel: hashlib.sha256((part / rel).read_bytes()).hexdigest() for rel in before}
            if expect == "PASS":
                check("pass: candidate stays installed (profile_clf/ + key)", after != before and (part / "profile_clf/profile_clf_1.npz").exists()
                      and json.loads((part / "configs/inference_config.json").read_text())["profile_classifier"] == "profile_clf")
                print("         " + "\n         ".join(l for l in out.splitlines() if l.startswith("  ok") or l.startswith("  FAIL")))
            else:
                check("fail: submission #2 restored exactly, profile_clf/ removed", after == before and not (part / "profile_clf").exists())
    finally:
        proc.kill()
        proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
