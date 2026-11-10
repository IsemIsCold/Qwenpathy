#!/usr/bin/env python3
"""
tests/test_v4.py -- EXP-030 (V4 retrain) offline tests with fake llamafactory-cli / vLLM / GPUs. Needs bash, numpy.

    python tests/test_v4.py

A. v4_gate.py: a candidate identical to #4 FAILS (no emotion gain); a candidate with +3 pts and same replies PASSES;
   a slow one FAILS on latency.
B. v4_train_session.sh: writes the config with the base model + V4 data, runs "training", reports dev loss, DONE PASS;
   refuses reply-first data and a finished run.
C. v4_eval_session.sh: merge -> 600 points -> gate -> swap -> check run. PASS swaps the model folders (#4's kept);
   FAIL leaves #4 untouched.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FP8 = ROOT / "outputs/exp023/lp_fp8_t10.jsonl"
GOLDS = [ROOT / "data/val_gold_n100.jsonl", ROOT / "data/val_gold_tune500.jsonl"]
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def read_jsonl(p):
    with open(p, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_upload(tmp):
    U = tmp / "U"
    for rel in ("participant/run_inference.py", "scripts/eval.py", "scripts/validate.py", "scripts/v4_gate.py",
                "scripts/v4_train_session.sh", "scripts/v4_eval_session.sh", "data/t10_input.jsonl",
                "data/val_gold_n100.jsonl", "data/val_gold_tune500.jsonl", "outputs/exp023/lp_fp8_t10.jsonl",
                "experiments/exp030-v4-decide-first/configs/v4_train.yaml", "experiments/exp030-v4-decide-first/configs/v4_merge.yaml"):
        (U / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, U / rel)
    return U


def v4_answers(tmp, better):
    """Fake V4 raw answers, decide-first order, replies = #4's. better=True fixes ~18 of #4's emotion errors."""
    gold = {r["id"]: r for g in GOLDS for r in read_jsonl(g)}
    out, fixed = tmp / f"v4_answers_{better}.jsonl", 0
    with open(out, "w", encoding="utf-8") as f:
        for r in read_jsonl(FP8):
            obj = json.loads(r["content"], strict=False) if r["id"] not in ("val_conv_000755_t7",) else None
            if obj is None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            lab = obj["emotion_label"]
            if better and lab != gold[r["id"]]["emotion_label"] and fixed < 18:
                lab = gold[r["id"]]["emotion_label"]
                fixed += 1
            first = {"emotion_label": lab, **{k: v for k, v in obj.items() if k != "emotion_label"}}
            f.write(json.dumps({"id": r["id"], "content": json.dumps(first, ensure_ascii=False)}, ensure_ascii=False) + "\n")
    return out


def fake_bin(tmp):
    b = tmp / "bin"
    b.mkdir()
    (b / "activate").write_text("# fake conda activate\n")
    (b / "nvidia-smi").write_text("#!/bin/sh\necho 'GPU 0: fake'\necho 'GPU 1: fake'\n")
    (b / "vllm").write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests/stub_vllm.py'} \"$@\"\n")
    lf = f"""#!/usr/bin/env bash
# fake llamafactory-cli: train -> progress + eval loss + adapter; export -> config.json + 1 shard
cfg="$2"
get() {{ grep "^$1:" "$cfg" | head -1 | sed "s/^$1: *//; s/ *#.*//"; }}
if [[ "$1" == train ]]; then
  out=$(get output_dir); mkdir -p "$out"
  echo "  5%|5/100 [00:01<00:20, 1.0it/s]"; echo "{{'eval_loss': 1.234, 'epoch': 0.5}}"; echo " 100%|100/100 [00:20<00:00, 1.0it/s]"
  echo '{{"current_steps": 50, "eval_loss": 1.234}}' > "$out/trainer_log.jsonl"
  echo '{{"train_loss": 1.1, "train_runtime": 120, "train_samples_per_second": 2.3}}' > "$out/all_results.json"
  echo x > "$out/adapter_model.safetensors"; echo '{{}}' > "$out/adapter_config.json"
  cp "$cfg" "$out/used_config.yaml"
elif [[ "$1" == export ]]; then
  d=$(get export_dir); mkdir -p "$d"; echo '{{}}' > "$d/config.json"; echo x > "$d/model-00001-of-00001.safetensors"
fi
"""
    (b / "llamafactory-cli").write_text(lf)
    (b / "python").write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
    for f in ("nvidia-smi", "vllm", "llamafactory-cli", "python"):
        (b / f).chmod(0o755)
    return b


def main():
    tmp = Path(tempfile.mkdtemp(prefix="v4_test_"))
    try:
        U = make_upload(tmp)
        b = fake_bin(tmp)
        venv = tmp / "venv"
        (venv / "bin").mkdir(parents=True)
        for f in ("llamafactory-cli", "python"):
            shutil.copy2(b / f, venv / "bin" / f)

        print("A. gate")
        stub_env = {"STUB_GOLD": ",".join(map(str, GOLDS))}

        def gate(answers, extra_env=None, sleep="0"):
            P = tmp / f"pg_{len(os.listdir(tmp))}"
            (P / "configs").mkdir(parents=True)
            shutil.copy2(ROOT / "participant/run_inference.py", P / "run_inference.py")
            shutil.copytree(ROOT / "participant/profile_clf", P / "profile_clf")
            cfg = json.loads((ROOT / "participant/configs/inference_config.json").read_text())
            cfg.update({"profile_classifier": "profile_clf", "stop_after_emotion": True, "profile_blend": {
                "personality_traits": [0.0, 0.45], "interests": [0.0, 0.30], "style": [0.0, 0.35]}})
            (P / "configs/inference_config.json").write_text(json.dumps(cfg))
            port = free_port()
            stub = subprocess.Popen([sys.executable, str(ROOT / "tests/stub_vllm.py"), "--port", str(port)],
                                    env={**os.environ, **stub_env, "STUB_REPLAY": str(answers), "STUB_SLEEP_PER_CHAR": sleep},
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time, urllib.request
            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read(); break
                except OSError:
                    time.sleep(0.1)
            out = P / "out"
            r = subprocess.run([sys.executable, str(P / "run_inference.py"), str(U / "data/t10_input.jsonl"), str(out)], capture_output=True, text=True,
                               env={**os.environ, "EVALUATION_BACKEND": "openai_compatible", "EVALUATION_BASE_URL": f"http://127.0.0.1:{port}/v1",
                                    "EVALUATION_API_MODEL": "model", "EVALUATION_HARDWARE_LABEL": "t"})
            stub.terminate(); stub.wait()
            (out.parent / "start.log").write_text(r.stdout)
            g = subprocess.run([sys.executable, str(U / "scripts/v4_gate.py"), "--sub", str(out / "submission.jsonl"), "--report",
                                str(out / "performance_report.json"), "--start-log", str(out.parent / "start.log"), "--valid", "1", "--u", str(U)],
                               capture_output=True, text=True)
            return g
        g = gate(v4_answers(tmp, False))
        check("A1 candidate = #4 (decide-first order, same labels) FAILS on the emotion rule", g.returncode == 3 and "FAIL emotion >= #4 + 12" in g.stdout,
              next((l.strip() for l in g.stdout.splitlines() if "emotion: V4" in l), g.stderr[-300:]))
        g = gate(v4_answers(tmp, True))
        check("A2 candidate with +18 correct, same replies PASSES", g.returncode == 0, next((l.strip() for l in g.stdout.splitlines() if "emotion: V4" in l), g.stdout[-500:] + g.stderr[-500:]))
        last = sorted(tmp.glob("pg_*"))[-1] / "out"
        rep = json.loads((last / "performance_report.json").read_text())
        rep["average_latency_ms"] = 1100.0  # > 1.10 x 923 ms
        (last / "performance_report.json").write_text(json.dumps(rep))
        g = subprocess.run([sys.executable, str(U / "scripts/v4_gate.py"), "--sub", str(last / "submission.jsonl"), "--report",
                            str(last / "performance_report.json"), "--start-log", str(last.parent / "start.log"), "--valid", "1", "--u", str(U)],
                           capture_output=True, text=True)
        check("A3 same candidate at 1.10 s mean FAILS on latency", g.returncode == 3 and "FAIL latency" in g.stdout)

        print("B. train session")
        data = tmp / "sft_v4"
        data.mkdir()
        row = {"system": "s", "conversations": [{"from": "human", "value": "u"}, {"from": "gpt", "value": '{"emotion_label": "joy", "response_text": "x"}'}]}
        (data / "train_v4_sharegpt.json").write_text(json.dumps([row] * 3))
        (data / "dev200_sharegpt.json").write_text(json.dumps([row]))
        (data / "dataset_info.json").write_text("{}")
        base = tmp / "base"
        base.mkdir()
        (base / "config.json").write_text("{}")
        env = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "U": str(U), "DATA": str(data), "RUN": str(tmp / "runs/v4"),
               "OUT": str(tmp / "out_v4"), "BASE": str(base), "VENV": str(venv), "V4_SKIP_GPU_CHECK": "1", "V4_POLL_S": "1"}
        r = subprocess.run(["bash", str(U / "scripts/v4_train_session.sh")], capture_output=True, text=True, env=env)
        if "DONE -- PASS" not in r.stdout:
            print(r.stdout[-2000:], r.stderr[-1000:])
        used = (tmp / "runs/v4/used_config.yaml").read_text() if (tmp / "runs/v4/used_config.yaml").exists() else ""
        check("B1 train session ends DONE -- PASS with dev loss reported", r.returncode == 0 and "DONE -- PASS" in r.stdout and "50: 1.2340" in r.stdout,
              r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
        check("B2 config used: base model, V4 data, dev eval, resume-safe", f"model_name_or_path: {base}" in used and f"dataset_dir: {data}" in used
              and "dataset: migu_train_v4" in used and "eval_dataset: migu_dev200" in used and "overwrite_output_dir: false" in used)
        r = subprocess.run(["bash", str(U / "scripts/v4_train_session.sh")], capture_output=True, text=True, env=env)
        check("B3 refuses to overwrite a finished run", r.returncode == 1 and "already holds a finished run" in r.stdout)
        row2 = {**row, "conversations": [row["conversations"][0], {"from": "gpt", "value": '{"response_text": "x", "emotion_label": "joy"}'}]}
        (data / "train_v4_sharegpt.json").write_text(json.dumps([row2]))
        r = subprocess.run(["bash", str(U / "scripts/v4_train_session.sh")], capture_output=True, text=True, env={**env, "RUN": str(tmp / "runs/v4b")})
        check("B4 refuses reply-first data", r.returncode == 1 and "not decide-first" in r.stdout)

        print("C. eval session")
        V = tmp / "val"
        V.mkdir()
        shutil.copy2(ROOT / "data/val_input_n100.jsonl", V / "t9_input.jsonl")

        def eval_run(name, answers):
            P = tmp / name
            (P / "configs").mkdir(parents=True)
            shutil.copy2(ROOT / "participant/run_inference.py", P / "run_inference.py")
            shutil.copy2(ROOT / "participant/start.sh", P / "start.sh")
            shutil.copytree(ROOT / "participant/profile_clf", P / "profile_clf")
            cfg = json.loads((ROOT / "participant/configs/inference_config.json").read_text())
            cfg.update({"profile_classifier": "profile_clf", "stop_after_emotion": True, "profile_blend": {
                "personality_traits": [0.0, 0.45], "interests": [0.0, 0.30], "style": [0.0, 0.35]}})
            (P / "configs/inference_config.json").write_text(json.dumps(cfg))
            (P / "model").mkdir()
            (P / "model/config.json").write_text('{"t7": true}')
            e = {**env, "P": str(P), "V": str(V), "OUT": str(tmp / f"{name}_out"), "MERGED": str(tmp / f"{name}_merged"),
                 "KEEP": str(tmp / f"{name}_keep"), "PY": sys.executable, "PYTHON_BIN": sys.executable, "V4_TEST": "1",
                 "V4_EXPECT_SHA": "0", "STUB_REPLAY": str(answers), "STUB_GOLD": ",".join(map(str, GOLDS)),
                 "VLLM_PORT": str(free_port()), "RUN": str(tmp / "runs/v4")}
            r = subprocess.run(["bash", str(U / "scripts/v4_eval_session.sh")], capture_output=True, text=True, env=e)
            return r, P
        r, P = eval_run("ev_pass", v4_answers(tmp, True))
        check("C1 PASS path: DONE -- PASS", "DONE -- PASS" in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
        check("C2 PASS swaps the model folders, #4's weights kept", (tmp / "ev_pass_keep/config.json").read_text() == '{"t7": true}'
              and (P / "model/config.json").read_text().strip() == "{}" if (tmp / "ev_pass_keep").exists() else False)
        if "DONE -- PASS" not in r.stdout:
            print(r.stdout[-3000:], r.stderr[-1000:])
        r, P = eval_run("ev_fail", v4_answers(tmp, False))
        check("C3 FAIL path: DONE -- FAIL, #4's model untouched, no swap", "DONE -- FAIL (V4 rejected" in r.stdout
              and (P / "model/config.json").read_text() == '{"t7": true}' and not (tmp / "ev_fail_keep").exists(),
              r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
