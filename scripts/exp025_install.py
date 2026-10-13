#!/usr/bin/env python3
"""
scripts/exp025_install.py -- install (or undo) submission #3 = #2 + the EXP-024/025 profile blend.

Install = 3 changes in /root/participant, after backing up submission #2's files once
(an existing backup is never overwritten):
  1. run_inference.py               <- the new one (identical output without the config key)
  2. profile_clf/                   <- the classifier files (profile_clf.json + profile_clf_1.npz + profile_clf_2.npz)
  3. configs/inference_config.json  <- "profile_classifier": "profile_clf"
start.sh and the model are not touched.

    python3 exp025_install.py --participant /root/participant --backup /root/sub2_backup_exp025 \
        --new-run-inference U/participant/run_inference.py --classifier-dir U/participant/profile_clf
    python3 exp025_install.py --participant /root/participant --backup /root/sub2_backup_exp025 --restore

stdlib only.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

FILES = ["run_inference.py", "configs/inference_config.json"]
KEY = "profile_classifier"
DIRNAME = "profile_clf"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--backup", required=True)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--new-run-inference")
    ap.add_argument("--classifier-dir")
    args = ap.parse_args()
    p, b = Path(args.participant), Path(args.backup)

    if args.restore:
        sums = {rel: h for h, rel in (l.split("  ", 1) for l in (b / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if l)}
        for rel in FILES:
            if sha(b / rel) != sums[rel]:
                sys.exit(f"backup file changed since it was made: {b / rel}")
        for rel in FILES:
            shutil.copy2(b / rel, p / rel)
        if (p / DIRNAME).exists() and not (b / "had_profile_clf").exists():
            shutil.rmtree(p / DIRNAME)
        print("restored:", "  ".join(f"{rel} {sha(p / rel)[:12]}" for rel in FILES), f"| {DIRNAME}/ present: {(p / DIRNAME).exists()}")
        return

    if not args.new_run_inference or not args.classifier_dir:
        sys.exit("install needs --new-run-inference and --classifier-dir")
    src = Path(args.classifier_dir)
    names = ["profile_clf.json", *json.loads((src / "profile_clf.json").read_text(encoding="utf-8"))["weight_files"]]
    for f in (*(src / n for n in names), Path(args.new_run_inference), *(p / rel for rel in FILES)):
        if not f.is_file():
            sys.exit(f"missing: {f}")
    print("before:", "  ".join(f"{rel} {sha(p / rel)[:12]}" for rel in FILES))
    if b.exists():
        print(f"backup exists, kept as is: {b}")
    else:
        for rel in FILES:
            (b / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p / rel, b / rel)
        if (p / DIRNAME).exists():
            (b / "had_profile_clf").write_text("1")
        (b / "SHA256SUMS").write_text("".join(f"{sha(b / rel)}  {rel}\n" for rel in FILES), encoding="utf-8")
        print(f"backup of submission #2 -> {b}")

    shutil.copy2(args.new_run_inference, p / "run_inference.py")
    (p / DIRNAME).mkdir(exist_ok=True)
    for name in names:
        shutil.copy2(src / name, p / DIRNAME / name)
    cfg_path = p / "configs" / "inference_config.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    config[KEY] = DIRNAME
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("after: ", "  ".join(f"{rel} {sha(p / rel)[:12]}" for rel in FILES),
          "|", "  ".join(f"{DIRNAME}/{n} {sha(p / DIRNAME / n)[:12]}" for n in names))
    print(f'installed: config "{KEY}": "{DIRNAME}"')


if __name__ == "__main__":
    main()
