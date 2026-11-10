#!/usr/bin/env python3
"""
scripts/exp027_install.py -- install (or undo) submission #4 = #3 + stop after emotion_label.

#3 must be installed in /root/participant (config has "profile_classifier"). Install = 2 changes, after
backing up #3's files once (an existing backup is never overwritten):
  1. run_inference.py               <- the EXP-027 one (same output as #3's when the new keys are absent,
                                       except 2 broken-JSON answers that the lenient parse now keeps)
  2. configs/inference_config.json  <- #3's config + "stop_after_emotion": true + "profile_blend" (variant)
start.sh, the model and profile_clf/ are not touched.

    python3 exp027_install.py --participant /root/participant --backup /root/sub3_backup_exp027 \
        --new-run-inference U/participant/run_inference.py --variant exact        # or f1
    python3 exp027_install.py --participant /root/participant --backup /root/sub3_backup_exp027 --restore

Variants = classifier alone (a = 0) with thresholds fitted on the 600 val points (EXP-027, Oct 16):
  exact  personality 0.45, interests 0.30, style 0.35  -> per-field exact 0.652 in-sample, 0.647 5-fold CV
  f1     personality 0.25, interests 0.10, style 0.15  -> micro-F1 0.471 / exact 0.567 (CV)
The organisers confirmed the profile is scored by exact match -> `exact`. stdlib only.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

FILES = ["run_inference.py", "configs/inference_config.json"]
VARIANTS = {
    "exact": {"personality_traits": [0.0, 0.45], "interests": [0.0, 0.30], "style": [0.0, 0.35]},
    "f1": {"personality_traits": [0.0, 0.25], "interests": [0.0, 0.10], "style": [0.0, 0.15]},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--backup", required=True)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--new-run-inference")
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    args = ap.parse_args()
    p, b = Path(args.participant), Path(args.backup)

    if args.restore:
        sums = {rel: h for h, rel in (line.split("  ", 1) for line in (b / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if line)}
        for rel in FILES:
            if sha(b / rel) != sums[rel]:
                sys.exit(f"backup file changed since it was made: {b / rel} (not restoring)")
        for rel in FILES:
            shutil.copy2(b / rel, p / rel)
            if sha(p / rel) != sums[rel]:
                sys.exit(f"restore failed for {rel}")
        print("restored #3:", " ".join(f"{rel}={sums[rel][:12]}" for rel in FILES))
        return

    if not (args.new_run_inference and args.variant):
        sys.exit("install needs --new-run-inference and --variant")
    config = json.loads((p / FILES[1]).read_text(encoding="utf-8"))
    if not b.exists():
        if not config.get("profile_classifier") or "stop_after_emotion" in config:
            sys.exit("the installed files are not submission #3 (need profile_classifier, no stop_after_emotion): no backup made")
        (b / "configs").mkdir(parents=True)
        for rel in FILES:
            shutil.copy2(p / rel, b / rel)
        (b / "SHA256SUMS").write_text("".join(f"{sha(b / rel)}  {rel}\n" for rel in FILES), encoding="utf-8")
        print(f"backup of #3 -> {b}")
    else:
        print(f"backup exists, kept: {b}")
    base = json.loads((b / FILES[1]).read_text(encoding="utf-8"))  # always build on #3's own config
    base["stop_after_emotion"] = True
    base["profile_blend"] = VARIANTS[args.variant]
    shutil.copy2(args.new_run_inference, p / FILES[0])
    (p / FILES[1]).write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"installed EXP-027 variant {args.variant}: run_inference {sha(p / FILES[0])[:12]}, config {sha(p / FILES[1])[:12]}")


if __name__ == "__main__":
    main()
