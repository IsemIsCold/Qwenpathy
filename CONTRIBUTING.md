# Team rules

There are five of us, we get 5 successful submissions, and our GPU budget is limited. These rules keep us from wasting any of it.

## 1. Git workflow

- `main` must **always run end to end**: input file → `submission.jsonl` → `validate.py` passes.
- Work on your own branch: `name/short-task`, e.g. `zakaria/eval-script`.
- Open a pull request into `main`. **The captain merges.** Every PR needs:
  - a description of what changed and how you tested it
  - if it touches inference: the `validate.py` output and the `eval.py` scores on val
- Keep PRs small: one task per PR.

## 2. Never commit

- Model weights (`*.safetensors`, `*.bin`, `participant/model/`)
- Contest data (`data/`). It's the organisers' data, and it's big.
- API keys, Migu or AutoDL passwords, or tokens
- Run outputs (`outputs/`)

## 3. Submissions: captain only

We get **5 successful test runs**, and the **last successful one counts**. A failed run doesn't use up an attempt. Before every submission:

1. [ ] `main` is up to date, and the image is built from `main`.
2. [ ] `start.sh` has run end to end inside the official base image on sample data.
3. [ ] `validate.py` passes on the full val set: every line has all 4 fields and the labels are from the allowed lists.
4. [ ] `eval.py` on val is **at least as good** as the last submission.
5. [ ] The latency for the first 100 samples has been checked.
6. [ ] The planned submission is logged in [LOG.md](LOG.md) **before** it goes out.

After **Oct 18** there are no risky changes. A worse final submission replaces a better one.

## 4. Compute

- **Migu (咪咕 Mīgū):** 20 free card-hours for the whole team. A 2-GPU instance probably burns **2 card-hours per hour**. Use 无卡开机 (wú kǎ kāijī, start without GPU) for setup and file copying.
- **AutoDL:** team-paid, for training.
- **Before any GPU job**, write the estimated hours in LOG.md. Afterwards, write the actual hours.
- Debug first on CPU, with a tiny model (Qwen2.5-0.5B) and 5–50 samples. A GPU is only for jobs that already work on a small scale.

## 5. Daily sync (WeChat, every evening, about 10 min)

Each person posts three lines:

```
Done:    ...
Blocked: ...
Next:    ...
```

If you're blocked for more than half a day, say so in the group straight away. Don't wait for the sync.

## 6. Questions about the rules

Don't guess. Post the question in our group, and the captain asks the organisers in the official WeChat group. Record the answers in `docs/organiser-answers.md`.
