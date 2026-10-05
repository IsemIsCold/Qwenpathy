# How to submit on Migu (咪咕仝学 Mīgū Tóngxué)

First done for submission #1 on 2026-10-04 (Ismail F. + Yahya). Every step below was actually run. Steps marked **TODO** still need confirming.

Platform: https://aigpu.migu.cn → log in → **console**. The left menu is called "AI computing power…".

## 0. Before you touch a GPU

- The `participant/` you submit must match `main`. Check `participant/configs/inference_config.json`:
  `"backend": "openai_compatible"` and `"api_options": {"chat_template_kwargs": {"enable_thinking": false}}`.
  (On Oct 4 the committed config was still the organisers' default `transformers`. That runs Qwen3 *with thinking* and without vLLM.)
- Card-hours: the voucher showed **30** card-hours (29.99 after creating the instance). The 2-GPU instance costs 500 AIQ/h, i.e. 2 card-hours per hour.

## 1. Create the instance (once)

**Create an instance** with:

| Field | Value |
|---|---|
| Computer room | Southwest District 2 |
| Model | RTX 4090 |
| Billing method | Billing by volume (can't be changed later; uses the voucher) |
| Scheduling method | Seize (default) |
| Specification | **2 GPUs** (32 cores / 128 GB RAM / 100 GB disk) |
| Mirror type | Public Mirror → `vLLM 0.19.0-ubuntu22.04-p3.12-cuda12.8` (**only the competition's public mirror can upload entries**) |
| Container path | `/mnt/storage` (default) |

**Create and power on** starts the GPUs right away → **Shutdown** at once, then **Power on without card**.

Our instance: `b7-sub1`, ID `30834204-f664-4d25-a721-031ab5094984`.

## 2. Work in no-GPU mode (Power on without card)

Access: **SSH** (shown under System Services). The port and hostname can change after every power-on, so check them each time.

```powershell
ssh root@183.222.230.12 -p <port>           # small -p
scp -P <port> <files> root@183.222.230.12:/root/...   # capital -P
```

Image facts (checked Oct 4):
- `conda_env` exists and is active by default. Python, vLLM 0.19.0, transformers 4.57.6, torch 2.10.0+cu128.
- **No git** → copy `participant\` with `scp -r` from a laptop checkout of `main`.
- `/` has ~93 GB free and **is saved into the image**. `/mnt/storage` and `/data` are **not** saved.
- `/root` survives a shutdown and power-on.

```bash
# on the instance
sed -i 's/\r$//' /root/participant/start.sh
/opt/conda/bin/pip install -q modelscope          # into base, keeps conda_env clean
nohup /opt/conda/bin/modelscope download --model Qwen/Qwen3-8B --local_dir /root/participant/model > /root/dl.log 2>&1 &
# ~8.5 min, 16 GB, 5 safetensors shards
```

## 3. GPU test run (Turn on)

Click **Turn on** and set a **Timed switch** as a safety net. Then run the test as a script under `nohup`, with timestamps (see `run_b7.sh` in LOG.md / this doc's history):

```bash
bash /root/participant/start.sh <input.jsonl> <result_dir> 2>&1 | while IFS= read -r l; do echo "$(date +%s.%N) $l"; done | tee run.log
```

Before shutting down, copy `/tmp/competition-vllm.*.log` to `/root/` (`/tmp` may be wiped).

Oct 4 results (Qwen3-8B, TP2, 32k context, V0 prompt):
- vLLM start-up 109 s cold / 51 s warm (compile cache in `/root/.cache`; keep it)
- latency on 50 val points: mean 1.41 s, median 1.40 s, max 2.01 s; 0 parse failures
- KV cache 187k tokens (5.7× concurrency at 32k)
- val50: emotion acc 0.48 (API 0.50), ROUGE-L 0.173

## 4. Validate on a laptop (no GPU)

```powershell
scp -P <port> -r root@183.222.230.12:/root/out/* outputs\migu_val50\
python scripts\validate.py outputs\migu_val50\val50\submission.jsonl --input data\val_input_n50.jsonl
python scripts\eval.py outputs\migu_val50\val50\submission.jsonl data\val_gold_n50.jsonl --out outputs\migu_val50\metrics.json --history=
```

PowerShell drops `--history ""`, so use `--history=`.

## 5. Clean up and save the image

In no-GPU mode:

```bash
rm -rf /root/out /root/data /root/*.sh /root/*.log /root/nohup.out /root/.cache/pip /root/.cache/modelscope
chmod -R a+rX /root/participant && chmod +x /root/participant/start.sh
```

Then **Shutdown** → **Save image**:
- Mirror Name: `npu-ic-qwen3-8b-v0`
- Version Number: `v20261004225439` (the default timestamp)
- Notes: everything on the system disk is saved except `/mnt/storage` and `/data`. Fees apply if your saved images total more than 30 GB (ours is ~44 GB). **TODO:** record the actual fee.
- After saving, **restart the instance** (the platform says so), in no-GPU mode.

## 6. Upload and submit — TODO

1. **Upload competition mirror** (button next to Mirror type on the instance card). The console says you must upload the entry image before you can submit for evaluation.
2. Submit for evaluation on the competition page.
3. Record in LOG.md: date, commit, image name and version, official score.
4. **Shutdown** the instance.
