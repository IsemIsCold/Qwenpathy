#!/usr/bin/env python3
"""
scripts/exp023_install.py -- install (or undo) the submission #3 candidate in /root/participant.

Install = 3 changes, after backing up submission #2's files (once; an existing backup is never overwritten):
  1. run_inference.py  <- the new one (emotion_bias support; identical behaviour without the config key)
  2. start.sh          <- patched in place: reads "vllm_quantization" from the config and adds
                          --quantization <q> to the vLLM args (absent key = old behaviour)
  3. configs/inference_config.json  <- emotion_bias, emotion_bias_top_logprobs, vllm_quantization

    python3 exp023_install.py --participant /root/participant --backup /root/sub2_backup \
        --new-run-inference /mnt/storage/exp023_upload/participant/run_inference.py \
        --bias 0.5 --top-logprobs 5 --quantization fp8
    python3 exp023_install.py --participant /root/participant --backup /root/sub2_backup --restore

stdlib only.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

FILES = ["run_inference.py", "start.sh", "configs/inference_config.json"]
ANCHOR = 'echo "GPUs detected:'
MARKER = "vllm_quantization"
QUANT_BLOCK = '''# Optional weight quantisation from the config ("vllm_quantization": "fp8", EXP-020/023).
# Absent or empty = bf16 weights, exactly as before.
QUANTIZATION="$("$PYTHON_CMD" -c \\
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("vllm_quantization") or "")' \\
  "$CONFIG_FILE")"
if [[ -n "$QUANTIZATION" ]]; then
  VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_start(text):
    """start.sh text with the quantisation block inserted before the 'GPUs detected' echo."""
    if MARKER in text:
        return text, False
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(ANCHOR)]
    if len(hits) != 1:
        raise SystemExit(f"start.sh: expected exactly one line starting with {ANCHOR!r}, found {len(hits)}")
    i = hits[0]
    if "PYTHON_CMD=" not in "".join(lines[:i]) or "CONFIG_FILE=" not in "".join(lines[:i]):
        raise SystemExit("start.sh: PYTHON_CMD / CONFIG_FILE are not defined before the anchor line")
    return "".join(lines[:i]) + QUANT_BLOCK + "".join(lines[i:]), True


def backup(participant, dest):
    if dest.exists():
        print(f"backup exists, kept as is: {dest}")
        return
    for rel in FILES:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(participant / rel, dest / rel)
    (dest / "SHA256SUMS").write_text("".join(f"{sha(dest / rel)}  {rel}\n" for rel in FILES), encoding="utf-8")
    print(f"backup of the current files -> {dest}")


def restore(participant, src):
    lines = (src / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    sums = {rel: digest for digest, rel in (line.split("  ", 1) for line in lines if line)}
    for rel in FILES:
        if sha(src / rel) != sums[rel]:
            raise SystemExit(f"backup file changed since it was made: {src / rel}")
    for rel in FILES:
        shutil.copy2(src / rel, participant / rel)
        print(f"restored {rel}  {sha(participant / rel)[:12]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--backup", required=True)
    ap.add_argument("--restore", action="store_true", help="put the backed-up files back")
    ap.add_argument("--new-run-inference")
    ap.add_argument("--bias", type=float)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--quantization", default="fp8", help="'' = bf16")
    args = ap.parse_args()
    participant, dest = Path(args.participant), Path(args.backup)

    if args.restore:
        restore(participant, dest)
        return
    if args.new_run_inference is None or args.bias is None:
        sys.exit("install needs --new-run-inference and --bias")
    for rel in FILES:
        if not (participant / rel).is_file():
            sys.exit(f"missing: {participant / rel}")
    print("before:", "  ".join(f"{rel} {sha(participant / rel)[:12]}" for rel in FILES))
    backup(participant, dest)

    start = participant / "start.sh"
    text, changed = patch_start(start.read_text(encoding="utf-8"))
    if changed:
        start.write_text(text, encoding="utf-8", newline="")
    subprocess.run(["bash", "-n", str(start)], check=True)  # syntax check

    shutil.copy2(args.new_run_inference, participant / "run_inference.py")

    cfg_path = participant / "configs" / "inference_config.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    config["emotion_bias"] = args.bias
    config["emotion_bias_top_logprobs"] = args.top_logprobs
    config["vllm_quantization"] = args.quantization
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("after: ", "  ".join(f"{rel} {sha(participant / rel)[:12]}" for rel in FILES))
    print(f"installed: emotion_bias {args.bias:g}, top_logprobs {args.top_logprobs}, "
          f"quantization {args.quantization or 'none (bf16)'}; start.sh patched: {changed}")


if __name__ == "__main__":
    main()
