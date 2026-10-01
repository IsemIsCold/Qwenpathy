# Brief: Ismail F. (captain: pipeline + submissions)

**Tasks:** B1 local baseline (Oct 2), B7 submission #1 (Oct 4), I4 long conversations, and all submissions.

## B1: local baseline

1. Set up the venv, install CPU torch, and download Qwen2.5-0.5B-Instruct into `participant/model` (see the README).
2. Read `start.sh` and copy the `python run_inference.py ...` command it builds for the `transformers` backend into the README.
3. Run it on 5 samples (from Yahya's `convert_val.py`, or a hand-made UTF-8 file), and check the output.

## B7: submission #1

1. On Migu, use 无卡开机 (wú kǎ kāijī, start without GPU) to copy the code and model. Then switch to GPU only to run.
2. Use a bigger Qwen that fits 2×24 GB, run it on full val, and check that `validate.py` passes and record the `eval.py` scores.
3. Go through the checklist in CONTRIBUTING.md, log the run in LOG.md, and submit.

The goal of #1 is **to make sure the pipeline works**, not to get a good score.

## Ongoing

- Merge PRs. Keep `main` runnable.
- Chase the organisers' answers, and record them in `docs/organiser-answers.md`.
- Post the evening sync summary.
