#!/usr/bin/env python3
"""
tests/test_emotion_bias.py -- EXP-023: emotion_bias in run_inference.py, the start.sh
quantisation patch, exp023_install.py, and the whole exp023_session.sh on a fake vLLM.

CPU only, stdlib only, Linux/macOS (bash). Needs the EXP-022 outputs and data slices:
    python tests/test_emotion_bias.py [--old <submission #2 run_inference.py>]
Exit code 0 = all checks pass.
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

ROOT = Path(__file__).resolve().parent.parent
STUB = ROOT / "tests" / "stub_vllm.py"
DATA = ROOT / "data"
LP = [ROOT / "outputs/exp022/lp_bf16.jsonl", ROOT / "outputs/exp022/lp_fp8.jsonl"]
GOLDS = [DATA / "val_gold_n100.jsonl", DATA / "val_gold_tune500.jsonl"]

results = []
# Submission #2's run_inference.py, to check that "no emotion_bias" changes nothing:
#   python tests/test_emotion_bias.py --old path/to/old_run_inference.py
OLD_RI = Path(sys.argv[sys.argv.index("--old") + 1]).resolve() if "--old" in sys.argv else None


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_health(port, proc):
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError("stub died")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("stub not ready")


def make_fake_bin(tmp):
    bindir = tmp / "bin"
    bindir.mkdir()
    (bindir / "vllm").write_text(f'#!/usr/bin/env bash\nexec {sys.executable} {STUB} "$@"\n')
    (bindir / "nvidia-smi").write_text('#!/usr/bin/env bash\necho "GPU 0: fake"; echo "GPU 1: fake"\n')
    (bindir / "activate").write_text("# fake conda activate\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    return bindir


def sub2_start():
    """Submission #2's start.sh = the repo's start.sh without the quantisation block."""
    install = load_module("install_", ROOT / "scripts/exp023_install.py")
    text = (ROOT / "participant/start.sh").read_text(encoding="utf-8")
    assert text.count(install.QUANT_BLOCK) == 1
    return text.replace(install.QUANT_BLOCK, "")


def make_participant(tmp, name="participant", run_inference=None):
    p = tmp / name
    (p / "configs").mkdir(parents=True)
    (p / "model").mkdir()
    (p / "model" / "config.json").write_text("{}")
    shutil.copy(run_inference or ROOT / "participant/run_inference.py", p / "run_inference.py")
    (p / "start.sh").write_text(sub2_start(), encoding="utf-8", newline="")
    shutil.copy(ROOT / "participant/configs/inference_config.json", p / "configs/inference_config.json")
    return p


def test_equivalence(ri, tuner):
    print("\n1. run_inference.py scoring == tune_emotion_bias.py on all EXP-022 points")
    for lp_path in LP:
        recs = [r for r in read_jsonl(lp_path) if r["label_top"]]
        for k in (None, 5):
            diffs = 0
            for r in recs:
                alts = r["label_top"] if k is None else sorted(r["label_top"], key=lambda a: -a["logprob"])[:k]
                s_ri, s_t = ri.label_scores(alts), tuner.label_scores(r["label_top"], k)
                label = max(s_t, key=s_t.get) if s_t else "anxiety"
                for b in (0, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0):
                    diffs += ri.apply_emotion_bias(label, s_ri, b) != tuner.corrected(label, s_t, b)
                    diffs += s_ri.keys() != s_t.keys()
            check(f"{lp_path.name} top-{k or 'all'}: identical decisions", diffs == 0, f"{len(recs)} points x 7 b")
    tokens = [{"token": t, "logprob": -0.1, "top_logprobs": []} for t in ['{"', "emotion", '_label', '":', ' "', "an", "xiety", '"}']]
    content = "".join(t["token"] for t in tokens)
    check("label_alternatives finds the label token", ri.label_alternatives(tokens, content)[0]["token"] == "an")
    check("no label key -> None", ri.label_alternatives(tokens[:2], '{"emotion') is None)


def run_ri(participant, test_file, outdir, port, config_extra):
    cfg_path = participant / "configs/inference_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in ("emotion_bias", "emotion_bias_top_logprobs"):
        cfg.pop(key, None)
    cfg.update(config_extra)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    env = dict(os.environ, EVALUATION_BASE_URL=f"http://127.0.0.1:{port}/v1", EVALUATION_API_MODEL="model")
    return subprocess.run([sys.executable, str(participant / "run_inference.py"), str(test_file), str(outdir)],
                          env=env, capture_output=True, text=True)


def test_run_inference(tmp, stub_mod):
    print("\n2. run_inference.py against the stub server (n100)")
    port, log = 8011, tmp / "stub_log.jsonl"
    env = dict(os.environ, STUB_GOLD=",".join(map(str, GOLDS)), STUB_LOG=str(log))
    proc = subprocess.Popen([sys.executable, str(STUB), "--port", str(port)], env=env)
    try:
        wait_health(port, proc)
        old = make_participant(tmp, "old", OLD_RI)
        new = make_participant(tmp, "new")
        test = DATA / "val_input_n100.jsonl"
        r_old = run_ri(old, test, tmp / "out_old", port, {})
        log.unlink(missing_ok=True)
        r_new = run_ri(new, test, tmp / "out_new0", port, {})
        keys0 = read_jsonl(log)
        check("no emotion_bias: runs", r_new.returncode == 0, r_new.stderr[-300:])
        check("no emotion_bias: no logprobs in any request", all(not k["logprobs"] for k in keys0), f"{len(keys0)} requests")
        if OLD_RI:
            check(f"no emotion_bias: submission identical to {OLD_RI.name} (old)",
                  (tmp / "out_old/submission.jsonl").read_bytes() == (tmp / "out_new0/submission.jsonl").read_bytes())
        check("no emotion_bias: no bias line printed", "emotion bias" not in r_new.stdout)

        gold = {g["id"]: g for g in read_jsonl(DATA / "val_gold_n100.jsonl")}
        for b in (0.5, 1.0):
            log.unlink(missing_ok=True)
            r = run_ri(new, test, tmp / f"out_b{b}", port, {"emotion_bias": b})
            keys = read_jsonl(log)
            sub = {s["id"]: s for s in read_jsonl(tmp / f"out_b{b}/submission.jsonl")}
            want_switch = 0
            wrong = 0
            for sid, s in sub.items():
                said, alts = stub_mod.plan(sid)
                g = gold[sid]["emotion_label"]
                expect = g if said == "anxiety" and g != "anxiety" and alts[g] > alts["anxiety"] - b else said
                want_switch += expect != said
                wrong += s["emotion_label"] != expect
            check(f"b={b}: runs, every request asks logprobs + top_logprobs 5",
                  r.returncode == 0 and all(k["logprobs"] is True and k["top_logprobs"] == 5 for k in keys),
                  r.stderr[-300:])
            check(f"b={b}: every label = expected switch rule", wrong == 0, f"{want_switch} switches expected")
            check(f"b={b}: summary line", f"switched {want_switch}/100, no label scores 0" in r.stdout,
                  [l for l in r.stdout.splitlines() if "emotion bias" in l])
            rep = json.loads((tmp / f"out_b{b}/performance_report.json").read_text(encoding="utf-8"))
            check(f"b={b}: performance_report keys unchanged",
                  set(rep) == set(json.loads((tmp / "out_old/performance_report.json").read_text(encoding="utf-8"))))
        r = run_ri(new, test, tmp / "out_top3", port, {"emotion_bias": 0.5, "emotion_bias_top_logprobs": 3})
        check("emotion_bias_top_logprobs 3 is sent", all(k["top_logprobs"] == 3 for k in read_jsonl(log)[-100:]))
        for bad in ({"emotion_bias": -1}, {"emotion_bias": True}, {"emotion_bias": 0.5, "emotion_bias_top_logprobs": 21}):
            r = run_ri(new, test, tmp / "out_bad", port, bad)
            check(f"bad config rejected: {bad}", r.returncode != 0)
    finally:
        proc.kill()
        proc.wait()


def test_start_and_install(tmp, install):
    print("\n3. start.sh patch + exp023_install.py (fake vllm / nvidia-smi / conda)")
    bindir = make_fake_bin(tmp)
    argslog = tmp / "vllm_args.jsonl"
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", FAKE_VLLM_ARGS_LOG=str(argslog),
               STUB_GOLD=",".join(map(str, GOLDS)), VLLM_PORT="8012")
    p = make_participant(tmp, "inst", OLD_RI)
    before = {rel: sha(p / rel) for rel in install.FILES}
    backup = tmp / "sub2_backup"
    base = [sys.executable, str(ROOT / "scripts/exp023_install.py"), "--participant", str(p), "--backup", str(backup)]
    r = subprocess.run(base + ["--new-run-inference", str(ROOT / "participant/run_inference.py"), "--bias", "0.5"],
                       capture_output=True, text=True)
    check("install runs", r.returncode == 0, r.stderr[-300:])
    cfg = json.loads((p / "configs/inference_config.json").read_text(encoding="utf-8"))
    check("config: bias 0.5, top 5, fp8", (cfg["emotion_bias"], cfg["emotion_bias_top_logprobs"], cfg["vllm_quantization"]) == (0.5, 5, "fp8"))
    check("backup holds submission #2", {rel: sha(backup / rel) for rel in install.FILES} == before)
    patched = (p / "start.sh").read_text(encoding="utf-8")
    r2 = subprocess.run(base + ["--new-run-inference", str(ROOT / "participant/run_inference.py"), "--bias", "0.75"],
                        capture_output=True, text=True)
    check("install twice: start.sh patched once, backup kept",
          r2.returncode == 0 and (p / "start.sh").read_text(encoding="utf-8") == patched
          and patched.count("VLLM_ARGS+=(--quantization") == 1
          and {rel: sha(backup / rel) for rel in install.FILES} == before)

    r = subprocess.run(["bash", str(p / "start.sh"), str(DATA / "val_input_n100.jsonl"), str(tmp / "chk")],
                       env=env, capture_output=True, text=True)
    args = read_jsonl(argslog)[-1]
    check("patched start.sh end to end", r.returncode == 0 and (tmp / "chk/submission.jsonl").exists(), r.stderr[-300:])
    check("vLLM got --quantization fp8 (+ TP2)", "--quantization" in args and args[args.index("--quantization") + 1] == "fp8"
          and args[args.index("--tensor-parallel-size") + 1] == "2", args)
    check("bias line in start.sh output", "emotion bias 0.75 (top_logprobs 5)" in r.stdout)

    cfg["vllm_quantization"] = ""
    (p / "configs/inference_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    subprocess.run(["bash", str(p / "start.sh"), str(DATA / "val_input_n100.jsonl"), str(tmp / "chk2")],
                   env=env, capture_output=True, text=True)
    check("empty vllm_quantization -> no --quantization", "--quantization" not in read_jsonl(argslog)[-1])
    unpatched = tmp / "unpatched"
    unpatched.mkdir()
    shutil.copy(backup / "start.sh", unpatched / "start.sh")
    text, changed = install.patch_start((unpatched / "start.sh").read_text(encoding="utf-8"))
    check("patch = original + one inserted block", changed and text.replace(install.QUANT_BLOCK, "") ==
          (unpatched / "start.sh").read_text(encoding="utf-8"))
    check("repo participant/start.sh = patched submission #2 start.sh",
          text == (ROOT / "participant/start.sh").read_text(encoding="utf-8"))

    r = subprocess.run(base + ["--restore"], capture_output=True, text=True)
    check("restore -> exactly submission #2's files", r.returncode == 0 and {rel: sha(p / rel) for rel in install.FILES} == before)


def run_session(tmp, name, stub_env):
    root = tmp / name
    root.mkdir()
    bindir = make_fake_bin(root)
    p = make_participant(root, "participant", OLD_RI)
    v = root / "val"
    v.mkdir()
    shutil.copy(DATA / "t10_input.jsonl", v / "t10_input.jsonl")
    shutil.copy(DATA / "val_input_n100.jsonl", v / "t9_input.jsonl")
    for g in ("val_gold_n50.jsonl", "val_gold_n50b.jsonl", "val_gold_n100.jsonl"):
        shutil.copy(DATA / g, v / g)
    before = {rel: sha(p / rel) for rel in ("run_inference.py", "start.sh", "configs/inference_config.json")}
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", STUB_GOLD=",".join(map(str, GOLDS)),
               P=str(p), U=str(ROOT), V=str(v), OUT=str(root / "out"), BACKUP=str(root / "sub2_backup"), **stub_env)
    r = subprocess.run(["bash", str(ROOT / "scripts/exp023_session.sh")], env=env, capture_output=True, text=True)
    return r, p, before, root


def test_session(tmp):
    print("\n4. exp023_session.sh end to end on the fake vLLM")
    r, p, before, root = run_session(tmp, "go", {})
    out = r.stdout
    check("GO session runs to DONE", r.returncode == 0 and "DONE" in out, (r.stderr or out)[-600:])
    check("gate says GO, candidate installed", "\nGO\n" in out and "installed: emotion_bias" in out)
    gate = json.loads((root / "out/gate.json").read_text())
    check("tuned b recovers the planted gaps (b >= 1.5 region)", gate["b"] >= 1.5, gate["b"])
    check("check run: validate + eval + latency lines",
          "accuracy (OFFICIAL)" in out and "latency first 100" in out and "parse failures 0" in out, out[-800:])
    check("check run used fp8 and the bias", "--quantization fp8" in out and "emotion bias" in out)
    check("backup = submission #2", all(sha(root / "sub2_backup" / rel) == h for rel, h in before.items()))
    no_server = subprocess.run(["bash", "-c", "pgrep -f '[s]tub_vllm.py serve' || true"], capture_output=True, text=True)
    check("no fake vLLM left running", no_server.stdout.strip() == "", no_server.stdout)

    # bf16 reference = gold replies (ROUGE-L 1.0) and stub replies cut in half -> ROUGE-L fails the gate
    ref = tmp / "ref_gold_tune500.jsonl"
    shutil.copy(DATA / "val_gold_tune500.jsonl", ref)
    r, p, before, root = run_session(tmp, "nogo", {"STUB_REPLY_FRACTION": "0.5", "BF16_REF": str(ref)})
    out = r.stdout
    check("NO-GO session exits 0 at the gate", r.returncode == 0 and "NO-GO" in out and "DONE" in out, (r.stderr or out)[-600:])
    check("NO-GO: participant untouched, no backup made",
          all(sha(p / rel) == h for rel, h in before.items()) and not (root / "sub2_backup").exists())


def main():
    ri = load_module("ri_new", ROOT / "participant/run_inference.py")
    tuner = load_module("tuner", ROOT / "scripts/tune_emotion_bias.py")
    install = load_module("install", ROOT / "scripts/exp023_install.py")
    os.environ["STUB_GOLD"] = ",".join(map(str, GOLDS))
    stub_mod = load_module("stub", STUB)
    tmp = Path(tempfile.mkdtemp(prefix="exp023_"))
    try:
        test_equivalence(ri, tuner)
        test_run_inference(tmp, stub_mod)
        test_start_and_install(tmp, install)
        test_session(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
