# Brief: Elly (infrastructure)

**You own:** everything that makes our code run on the organisers' machine. That's `participant/` (`start.sh`, `run_inference.py`), the Docker image, the Migu (咪咕 Mīgū) platform, the vLLM backend and latency.

**Tasks:** B1 local baseline (Oct 2), B7 Migu pipeline (Oct 4), I5 vLLM + latency (Oct 10), F3 buffer.

## The contract we must meet

```bash
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

This runs in conda env `conda_env`, **offline**, and must write `RESULT_DIR/submission.jsonl`. No human steps. If it crashes, we get nothing.

## B1: local baseline (CPU)

1. Make a venv, install CPU torch, transformers and modelscope (see the README), and download Qwen2.5-0.5B-Instruct into `participant/model`.
2. Read `start.sh`. Find the `python run_inference.py ...` command it builds for the `transformers` backend, and run it yourself on 5 samples.
3. Add the exact command to the README.

**Hint:** if you're on Windows, you can't run `start.sh` directly. Read it, don't run it. Also, PowerShell `>` writes UTF-16, so create test files from Python.

## B7: Migu pipeline

- Use 无卡开机 (wú kǎ kāijī, start without GPU) for setup and copying files, so it doesn't burn card-hours. Switch to GPUs only to run.
- Run the official image on the full val set with a bigger Qwen. Record the time and the GPU memory used.
- Work out exactly how we submit: save a private image, or upload a Docker image? Write it step by step in `docs/submission-howto.md`.
- Log every GPU hour in LOG.md.

## I5: speed + long histories

- Latency of the first 100 samples is scored. Compare the `transformers` and `openai_compatible` (vLLM) backends on the same 100 samples.
- Some histories are about 40k characters. Make sure inference never crashes on them. Agree the truncation rule with Zakaria (keep the system prompt + the most recent turns).
- The model must load from local files with nothing downloaded at runtime. Test with networking switched off.
