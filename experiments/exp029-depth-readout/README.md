# EXP-029: depth read-out probes (Attention Residuals idea, offline)

| | |
|---|---|
| **Date** | 2026-11-05 (built); GPU run pending |
| **Author** | Ismail F. |
| **Goal** | Test the one idea from Attention Residuals (Kimi / Moonshot, arXiv 2603.15031) that fits our deadline: can a learned read-out over **all** layers of the fine-tuned Qwen3-8B (#3/#4's model) predict emotion and profile better than the label the model writes from its last layer? |
| **Status** | Built and tested offline (`tests/test_exp029.py` 38/38). Needs one 2-GPU boot (≈ 0.7–1.2 card-hours; + ≈ 0.7 if the EXP-027 check run goes first). **No submission comes out of this experiment directly.** |

## Why not AttnRes itself

AttnRes replaces the residual sum `h_l = h_{l-1} + f(h_{l-1})` with attention over depth: `h_l = Σ_i α_{i→l} v_i`, `α = softmax(w_lᵀ RMSNorm(v_i))`, one learned pseudo-query `w_l` per layer, zero-initialised. It is a pre-training architecture: putting it into a pretrained Qwen breaks it, and there is no time to retrain. What we can take is the **read-out**: the same attention rule applied once, on top of the frozen model, to decide the emotion (and the profile) from every depth.

Earlier work on probes says: on 7–8B models, intermediate layers (≈ 60–80% depth) often hold task information better than the last layer, but the headroom over a fine-tuned model's own answer is small (≈ 2 pp). So the honest expectation is val accuracy ≈ 0.79–0.81 against the 0.788 we have; a PASS is possible, not likely. The layer curve and the depth weights are useful either way (defence and research material).

## Design

1. **Features** (`scripts/exp029_extract.py`, GPU): for each point, the exact prompt `run_inference.py` sends to vLLM (system prompt, `conversation_text()`, Qwen3 chat template, `enable_thinking=False`, generation prompt), one bf16 forward pass of `/root/participant/model` in Hugging Face transformers (no generation, no lm_head), forward hooks keep:
   - the state at the **last prompt token** (`\n\n` after `</think>`, the position right before the model starts writing) at 38 depths: embeddings, the outputs of layers 1–36, the final norm;
   - the **mean over the latest user message's tokens** at every 4th depth (embeddings, 4, 8, …, 36).
   Points: 6,000 SFT train points (seeded sample, fit), all 768 SFT dev points (held out from fine-tuning: **every choice is made here**), the 600 val points (scored once). ≈ 9.5M tokens, ≈ 2.9 GB of fp16 features in `/root/exp029_feats`. Two workers (one per GPU, no NCCL), resumable.
2. **Read-outs** (`scripts/exp029_probe.py`, GPU or CPU), all linear on z-scored features:
   - `gen`: the label the FP8 model writes (`outputs/exp023/lp_fp8_t10.jsonl`; #4's lenient parser keeps the 2 broken-JSON labels) = **473/600 = 0.788**
   - `final`: logistic regression on the final-norm state (what lm_head reads)
   - `best_layer`: the same on the best single depth (picked on dev)
   - `user_mean`: the same on the best user-message-mean depth
   - `scalar_mix`: one softmax weight per depth, shared by all points (ELMo-style)
   - `depth_attn`: AttnRes rule, one zero-initialised query `w`, `α_l = softmax_l(w · RMSNorm(v_l))` per point (keys from the raw states, which keep each depth's own direction), `h = Σ α_l z_l`, linear head
   - `blend`: `(1 − b)·log p_primary + b·log p_gen` with the model's own label log-probs, `b` by 5-fold CV over val conversations (**exploratory only**)
   Picked on dev: L2 (1e-4, 1e-3, 1e-2), class weights (none / √inverse frequency), the depth, the epoch (lowest dev NLL). The two mixes use 3 seeds and average their probabilities.
3. **Profile**: one-vs-rest logistic per depth (last token and user mean), one threshold per field tuned on dev for exact match.
4. **Diagnostics**: dev accuracy per depth (`layer_curve.csv`), α per depth and per gold class (`alpha.csv`), train in-sample vs grouped-CV vs dev at the best depth (did LoRA memorise the train labels into the features?), ECE before / after one temperature fitted on dev, per-class recall, bootstrap CI over conversations.

## Pass rules (fixed before any val number exists)

- **Primary read-out** = the dev-best of `final`, `best_layer`, `user_mean`, `scalar_mix`; `depth_attn` replaces it only if its dev accuracy is ≥ +0.5 pp higher.
- **Emotion PASS** = primary val accuracy ≥ **0.81** AND exact McNemar p < **0.05** against `gen` on the same 600 points.
- **Profile PASS** = val per-field exact match ≥ **0.672** (#4's CPU classifier with `exact` thresholds, 0.652, + 2 pts).
- Val is scored once, in the same run. Any change after seeing val numbers becomes EXP-029b and is reported as val-tuned.
- **A PASS does not ship anything.** The features come from HF bf16; the submission serves FP8 in vLLM, which would need a hidden-state path in vLLM (research flagged `extract_hidden_states` as risky: chunked prefill must be off, a prefix-cache bug, TP ≥ 2 crash, FP8 undocumented). A PASS opens A2 (that path + dry-run gates) for submission #5 only, never replacing #4.

## Run (one 2-GPU boot on the submission instance)

Upload `exp029_upload.tgz` in JupyterLab (it lands in `/`), then:

```bash
mkdir -p /mnt/storage/exp029_upload /mnt/storage/out && tar -xzf /exp029_upload.tgz -C /mnt/storage/exp029_upload
bash /mnt/storage/exp029_upload/scripts/exp029_session.sh 2>&1 | tee /mnt/storage/out/exp029_session.log
```

- Preflight first (scripts, model, data, python, 2 GPUs, disk); it stops in seconds if anything is missing.
- If `/mnt/storage/exp027_upload/scripts/exp027_session.sh` exists, the EXP-027 check run (#4 candidate, `VARIANT=exact`) runs first in the same boot (`RUN_EXP027=0` skips it). Its "shut the instance down" line can be ignored; EXP-029 continues.
- Progress and ETA every minute. If the boot dies, run the same command again: finished rows are kept.
- Workers stop cleanly after 75 min (`TIME_LIMIT_MIN`).
- Ends with the report, the wall time and the card-hours used, then `DONE -- shut the instance down now`.
- Bring back `/mnt/storage/out/exp029/` (report.md, results.json, layer_curve.csv, alpha.csv, val_predictions.jsonl, logs; ≈ 1 MB). The features stay in `/root/exp029_feats` for a possible EXP-029b.

Settings (environment variables): `N_TRAIN` (6000), `MODEL`, `FEATS`, `OUT`, `PY` (`/opt/train_venv/bin/python`), `SFT_TRAIN` / `SFT_DEV` (`/mnt/storage/sft/*_sharegpt.json`), `WORLD` (2), `PROBE_DEVICE` (cuda, CPU fallback), `RUN_EXP027` (auto/0/1), `VARIANT` (exact).

## Tests (`tests/test_exp029.py`, CPU, ≈ 2 min, 38/38)

- **A. Extraction on a tiny random Qwen3** with the real tokenizer:
  - prompt tokens equal the chat template of `run_inference.py`'s messages;
  - the user span decodes to the latest user message, and the last token is `\n\n`;
  - hook states equal HF `output_hidden_states` (rel. error 0), slot 37 equals the final norm, and the span means match;
  - 2 workers cover 44/44 points; an interrupted run resumes to identical features; `plan` refuses to overwrite a cache.
- **B. Probes on synthetic features** (emotion planted at depth 6 of 10, profile at depth 3):
  - the probes find both depths, and depth attention puts ≈ all its weight on depth 6;
  - the McNemar and ECE helpers match hand-computed values;
  - the primary is chosen by the rule above.
- **C. End to end:**
  - the probes run on the tiny-model features with the real FP8 answers file (`gen` = 473/600 on all 600);
  - `exp029_session.sh` runs fresh, then resumes;
  - preflight stops on a missing model, and on `RUN_EXP027=1` without its script.

## Results

(pending)
