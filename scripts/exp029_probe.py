#!/usr/bin/env python3
"""
scripts/exp029_probe.py -- EXP-029 step 2: fit read-outs on the cached hidden states and score them once on val.

Question: the fine-tuned Qwen3-8B writes emotion_label from its LAST layer. Is the emotion (and the profile)
easier to read from a learned mix of ALL layers, the way Attention Residuals (Kimi, arXiv 2603.15031) lets
each layer attend over the outputs of all earlier layers?

Splits (from exp029_extract.py's meta.jsonl): train = seeded sample of the SFT train set (fit),
dev = the SFT dev set (held out from fine-tuning; ALL choices are made here), val = the 600 val points
(scored once at the end, never used for a choice except the CV-on-val blend, reported separately).

Emotion read-outs (all linear on top of frozen features, features z-scored with train statistics):
  gen          the label the FP8 model generates (outputs/exp023/lp_fp8_t10.jsonl, = submission #3/#4's model)
  final        multinomial logistic regression on the final-norm state (what lm_head sees)
  best_layer   the same on the single best slot (picked on dev)
  user_mean    the same on the best user-message-mean slot (picked on dev)
  scalar_mix   one softmax weight per depth, shared by all points (ELMo-style), + linear head
  depth_attn   AttnRes-style: one learned pseudo-query w (zero-initialised => starts as the plain mean of
               depths), alpha_l = softmax_l(w . RMSNorm(v_l)) per point with v_l the raw state at depth l,
               h = sum_l alpha_l z_l (z = z-scored v), + linear head
  blend        (1-b) log p_primary + b log p_gen-logprobs, b by 5-fold CV over val conversations (exploratory)
Hyper-parameters (L2 strength, class weights none|sqrt-inverse; for the two mixes: L2 and the epoch, with the
class weighting of the best single layer) are picked on dev.
The stochastic mixes use 3 seeds and average their probabilities.

Pre-registered primary read-out (fixed before any val number exists): the dev-best of final / best_layer /
user_mean / scalar_mix; depth_attn replaces it only if its dev accuracy is >= +0.5 pp higher.
PASS (emotion) = primary val accuracy >= 0.81 AND exact McNemar p < 0.05 vs gen on the same 600 points.
Profile: per-field exact match (mean of the 3 fields), one-vs-rest logistic per slot, one threshold per field
tuned on dev for exact match; PASS = val per-field exact >= CPU classifier (0.652, submission #4 'exact') + 0.02.

    python exp029_probe.py --feats /root/exp029_feats --llm-answers outputs/exp023/lp_fp8_t10.jsonl \
        --out /mnt/storage/out/exp029 [--device cuda]

Outputs in --out: results.json (everything), report.md (human summary), layer_curve.csv, alpha.csv,
val_predictions.jsonl. Needs numpy + torch; run_inference.py is imported for labels and the answer parser.
"""

import argparse
import ast
import csv
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
PASS_ACC, PASS_P, DEPTH_MARGIN = 0.81, 0.05, 0.005
PROFILE_BASELINE, PROFILE_MARGIN = 0.652, 0.02
L2_GRID = (1e-4, 1e-3, 1e-2)
CW_GRID = ("none", "sqrt")
TH_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
LABEL_RE = re.compile(r'"emotion_label"\s*:\s*"([a-z_]+)"')


def load_ri(path):
    spec = importlib.util.spec_from_file_location("exp029_ri", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ stats helpers
def mcnemar_exact(a_ok, b_ok):
    """Two-sided exact McNemar: a right / b wrong vs a wrong / b right."""
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"a_only": b, "b_only": c, "p": min(1.0, 2 * tail)}


def ece(probs, y, bins=10):
    conf = probs.max(1)
    ok = probs.argmax(1) == y
    total = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins) if lo > 0 else (conf <= 1 / bins)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits, y):
    """One temperature on dev by NLL (grid over log T, then refine)."""
    lt = torch.as_tensor(logits, dtype=torch.float64)
    yt = torch.as_tensor(y)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    nll = [F.cross_entropy(lt / t, yt).item() for t in grid]
    return float(grid[int(np.argmin(nll))])


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_softmax_np(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def macro_f1(pred, y, k):
    f = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fn == 0:
            continue
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def boot_diff(a_ok, b_ok, groups, n=1000, seed=0):
    """95% CI of acc(a) - acc(b), resampling conversations."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    d = []
    for _ in range(n):
        rows = np.concatenate([idx[g] for g in rng.choice(uniq, size=len(uniq))])
        d.append(a_ok[rows].mean() - b_ok[rows].mean())
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def group_folds(groups, k=5, seed=0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


# ------------------------------------------------------------------ data
class Data:
    def __init__(self, feats, ri, device):
        feats = Path(feats)
        self.plan = json.loads((feats / "plan.json").read_text(encoding="utf-8"))
        self.meta = [json.loads(l) for l in open(feats / "meta.jsonl", encoding="utf-8")]
        done = np.load(feats / "done.npy")
        if not done.all():
            sys.exit(f"features incomplete: {int(done.sum())}/{len(done)} rows (rerun exp029_extract.py work)")
        self.device = device
        self.labels = sorted(ri.EMOTIONS)
        self.fields = list(ri.PROFILE_LABELS)
        self.tags = [(f, t) for f in self.fields for t in sorted(ri.PROFILE_LABELS[f])]
        self.fidx = {f: [j for j, (ff, _) in enumerate(self.tags) if ff == f] for f in self.fields}
        split = np.array([m["split"] for m in self.meta])
        self.idx = {s: np.flatnonzero(split == s) for s in ("train", "dev", "val")}
        for s, ix in self.idx.items():
            if len(ix) == 0:
                sys.exit(f"no {s} points in the features")
        self.y = np.array([self.labels.index(m["emotion"]) for m in self.meta])
        Y = np.zeros((len(self.meta), len(self.tags)), dtype=np.float32)
        for i, m in enumerate(self.meta):
            for j, (f, t) in enumerate(self.tags):
                Y[i, j] = t in (m["profile"].get(f) or [])
        self.Y = Y
        self.groups = np.array([m["conversation_id"] for m in self.meta])
        self.ids = [m["id"] for m in self.meta]
        log(f"loading features to {device} ...")
        self.last = torch.from_numpy(np.load(feats / "last.npy")).to(device)  # (N, S, D) fp16
        self.mean = torch.from_numpy(np.load(feats / "mean.npy")).to(device)  # (N, M, D) fp16
        tr = torch.as_tensor(self.idx["train"], device=device)
        self.stats = {}
        for name, X in (("last", self.last), ("mean", self.mean)):
            mu = torch.zeros(X.shape[1:], device=device)
            sq = torch.zeros(X.shape[1:], device=device)
            for chunk in torch.split(tr, 512):
                x = X[chunk].float()
                mu += x.sum(0)
                sq += (x * x).sum(0)
            mu /= len(tr)
            sd = (sq / len(tr) - mu * mu).clamp_min(1e-8).sqrt()
            self.stats[name] = (mu, sd)
        log(f"features: last {tuple(self.last.shape)}, mean {tuple(self.mean.shape)}; "
            f"train {len(self.idx['train'])}, dev {len(self.idx['dev'])}, val {len(self.idx['val'])}")

    def slot(self, kind, s, rows):
        X = self.last if kind == "last" else self.mean
        mu, sd = self.stats[kind]
        r = torch.as_tensor(rows, device=self.device)
        return (X[r, s].float() - mu[s]) / sd[s]

    def stack(self, rows, slots):
        """(values, keys) for the mixes: values = z-scored states, keys = RMSNorm of the RAW states
        (as in Attention Residuals: the raw residual stream still carries each depth's own direction)."""
        mu, sd = self.stats["last"]
        r = torch.as_tensor(rows, device=self.device)
        raw = self.last[r][:, slots].float()
        keys = raw * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + 1e-6)  # RMSNorm without gain
        return (raw - mu[slots]) / sd[slots], keys

    def class_weights(self, cw):
        k = len(self.labels)
        counts = np.bincount(self.y[self.idx["train"]], minlength=k).astype(float)
        if cw == "none":
            w = np.ones(k)
        else:
            w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** 0.5, 0.0)
            w = w / (w[self.y[self.idx["train"]]].mean())
        return torch.as_tensor(w, dtype=torch.float32, device=self.device)


# ------------------------------------------------------------------ models
def fit_linear(X, y, l2, weight=None, multilabel=False, max_iter=100):
    """Full-batch L-BFGS logistic regression. X (n, d) float32, y long (n,) or float (n, k)."""
    d = X.shape[1]
    k = y.shape[1] if multilabel else int(weight.shape[0])
    W = torch.zeros(d, k, device=X.device, requires_grad=True)
    b = torch.zeros(k, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ W + b
        if multilabel:
            loss = F.binary_cross_entropy_with_logits(z, y)
        else:
            loss = F.cross_entropy(z, y, weight=weight)
        loss = loss + l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


class Mix(torch.nn.Module):
    """scalar_mix: static softmax weights over depth; depth_attn: per-point alpha from a zero-init query."""

    def __init__(self, n_slots, d, k, kind):
        super().__init__()
        self.kind = kind
        self.s = torch.nn.Parameter(torch.zeros(n_slots))  # scalar_mix logits
        self.w = torch.nn.Parameter(torch.zeros(d))  # depth_attn pseudo-query (zero init = uniform mean)
        self.head = torch.nn.Linear(d, k)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def alpha(self, Z, K):  # Z values (n, S, d), K keys (n, S, d)
        if self.kind == "scalar_mix":
            return torch.softmax(self.s, 0).expand(Z.shape[0], -1)
        return torch.softmax(K @ self.w, dim=1)

    def forward(self, Z, K):
        a = self.alpha(Z, K)
        return self.head((a.unsqueeze(-1) * Z).sum(1)), a


def fit_mix(data, kind, slots, l2, cw, seed, epochs, lr=1e-3, mix_lr=1e-2, batch=256):
    """Adam; the depth weights (s / w) get a 10x larger step than the head so the mix actually moves.
    Epoch kept = lowest dev NLL (checked every 2 epochs); hyper-parameters are then chosen by dev accuracy."""
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    tr, dv = data.idx["train"], data.idx["dev"]
    k = len(data.labels)
    model = Mix(len(slots), data.plan["hidden_size"], k, kind).to(data.device)
    opt = torch.optim.Adam([{"params": model.head.parameters(), "lr": lr},
                            {"params": [model.s, model.w], "lr": mix_lr}])
    y_dv = torch.as_tensor(data.y[dv])
    weight = data.class_weights(cw)
    y_all = torch.as_tensor(data.y, device=data.device)
    best = (-1, None, None)
    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        for chunk in np.array_split(g.permutation(tr), max(1, len(tr) // batch)):
            logits, _ = model(*data.stack(chunk, slots))
            loss = F.cross_entropy(logits, y_all[torch.as_tensor(chunk, device=data.device)], weight=weight)
            loss = loss + l2 * (model.head.weight ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 2 == 0 or ep == epochs:
            lg = predict_mix(model, data, dv, slots)[0]
            nll = F.cross_entropy(torch.as_tensor(lg), y_dv).item()
            curve.append((ep, nll))
            if best[0] < 0 or nll < best[0]:
                best = (nll, ep, {n: p.detach().clone() for n, p in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model, best[1], curve


@torch.no_grad()
def predict_mix(model, data, rows, slots):
    model.eval()
    out, al = [], []
    for chunk in np.array_split(rows, max(1, len(rows) // 512)):
        lg, a = model(*data.stack(chunk, slots))
        out.append(lg.cpu().numpy())
        al.append(a.cpu().numpy())
    return np.concatenate(out), np.concatenate(al)


# ------------------------------------------------------------------ gen (the LLM's own label)
def llm_answers(path, ri, data):
    """Generated label + 16-way distribution from the label's first-token top logprobs, for val rows."""
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r
    k = len(data.labels)
    pred, logp = [], []
    for i in data.idx["val"]:
        r = rows.get(data.ids[i])
        if r is None:
            sys.exit(f"{path}: no answer for {data.ids[i]}")
        p, _ = ri.parse_prediction(data.ids[i], r["content"])
        label = p["emotion_label"]
        m = LABEL_RE.search(r["content"])  # = #4's lenient parser: 2 FP8 answers with broken JSON keep their label
        if m and m.group(1) in data.labels:
            label = m.group(1)
        pred.append(data.labels.index(label))
        top = r.get("label_top")
        top = ast.literal_eval(top) if isinstance(top, str) else (top or [])
        scores = ri.label_scores(top)
        floor = (min(scores.values()) if scores else 0.0) - 1.0
        v = np.array([scores.get(lab, floor) for lab in data.labels])
        logp.append(log_softmax_np(v[None])[0])
    return np.array(pred), np.array(logp)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--llm-answers", default=str(REPO / "outputs/exp023/lp_fp8_t10.jsonl"))
    ap.add_argument("--run-inference", default=str(REPO / "participant/run_inference.py"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--profile-baseline", type=float, default=PROFILE_BASELINE)
    args = ap.parse_args()
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 so CPU and GPU runs agree
    ri = load_ri(args.run_inference)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data(args.feats, ri, args.device)
    tr, dv, va = data.idx["train"], data.idx["dev"], data.idx["val"]
    y_tr = torch.as_tensor(data.y[tr], device=args.device)
    k = len(data.labels)
    S, M = data.plan["n_slots"], len(data.plan["mean_slots"])
    names = data.plan["slot_names"]
    res = {"plan": data.plan, "pass_rules": {
        "emotion": f"primary val acc >= {PASS_ACC} and McNemar p < {PASS_P} vs gen",
        "depth_attn_margin_dev": DEPTH_MARGIN,
        "profile": f"val per-field exact >= {args.profile_baseline} + {PROFILE_MARGIN}"}}

    # ---- 1. emotion: one linear probe per slot (last token) and per user-mean slot
    log("emotion: per-slot linear probes ...")
    curve = []  # (kind, slot, name, l2, cw, dev_acc, train_acc)
    probes = {}
    for kind, n_s in (("last", S), ("mean", M)):
        for s in range(n_s):
            Xtr, Xdv = data.slot(kind, s, tr), data.slot(kind, s, dv)
            best = None
            for l2 in L2_GRID:
                for cw in CW_GRID:
                    W, b = fit_linear(Xtr, y_tr, l2, data.class_weights(cw), max_iter=args.max_iter)
                    acc = float(((Xdv @ W + b).argmax(1).cpu().numpy() == data.y[dv]).mean())
                    if best is None or acc > best[0]:
                        tra = float(((Xtr @ W + b).argmax(1) == y_tr).float().mean())
                        best = (acc, l2, cw, W, b, tra)
            name = names[s] if kind == "last" else names[data.plan["mean_slots"][s]]
            curve.append({"kind": kind, "slot": s, "name": name, "l2": best[1], "cw": best[2],
                          "dev_acc": best[0], "train_acc": best[5]})
            probes[(kind, s)] = best
            log(f"  {kind:<4} {name:<11} dev {best[0]:.4f} (train {best[5]:.3f}, l2 {best[1]:g}, cw {best[2]})")
    with open(out / "layer_curve.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(curve[0]))
        wr.writeheader()
        wr.writerows(curve)
    res["layer_curve"] = curve

    def linear_readout(kind, s):
        acc, l2, cw, W, b, _ = probes[(kind, s)]
        lg = {sp: (data.slot(kind, s, data.idx[sp]) @ W + b).cpu().numpy() for sp in ("dev", "val")}
        return {"slot": names[s] if kind == "last" else names[data.plan["mean_slots"][s]], "kind": kind,
                "l2": l2, "cw": cw, "dev_acc": acc, "logits": lg}

    readouts = {}
    readouts["final"] = linear_readout("last", S - 1)
    s_best = max(range(S), key=lambda s: probes[("last", s)][0])
    readouts["best_layer"] = linear_readout("last", s_best)
    m_best = max(range(M), key=lambda s: probes[("mean", s)][0])
    readouts["user_mean"] = linear_readout("mean", m_best)

    # overfit diagnostic: does the probe read 'memorised' train labels? grouped 5-fold CV inside train
    Xtr_b = data.slot("last", s_best, tr)
    folds = group_folds(data.groups[tr])
    cv_ok = np.zeros(len(tr), dtype=bool)
    _, l2b, cwb, *_ = probes[("last", s_best)]
    for fo in range(5):
        a, bmask = np.flatnonzero(folds != fo), np.flatnonzero(folds == fo)
        W, b = fit_linear(Xtr_b[a], y_tr[a], l2b, data.class_weights(cwb), max_iter=args.max_iter)
        cv_ok[bmask] = (Xtr_b[bmask] @ W + b).argmax(1).cpu().numpy() == data.y[tr][bmask]
    res["overfit_check"] = {"slot": names[s_best], "train_in_sample": probes[("last", s_best)][5],
                            "train_grouped_cv": float(cv_ok.mean()), "dev": probes[("last", s_best)][0]}
    log(f"overfit check at {names[s_best]}: train in-sample {res['overfit_check']['train_in_sample']:.3f}, "
        f"train CV {cv_ok.mean():.3f}, dev {probes[('last', s_best)][0]:.3f}")

    # ---- 2. emotion: scalar mix and depth attention over embed + all decoder layers (final_norm excluded)
    mix_slots = list(range(S - 1))
    alpha_rows = []
    mix_cw = probes[("last", s_best)][2]  # class weighting as chosen for the best single layer (halves the grid)
    for kind in ("scalar_mix", "depth_attn"):
        log(f"emotion: {kind} over {len(mix_slots)} depths (cw {mix_cw}) ...")
        best = None
        for l2 in L2_GRID:
            for cw in (mix_cw,):
                models, eps = [], []
                for seed in range(args.seeds):
                    m, ep, _ = fit_mix(data, kind, mix_slots, l2, cw, seed, args.epochs)
                    models.append(m)
                    eps.append(ep)
                dev_p = np.mean([softmax_np(predict_mix(m, data, dv, mix_slots)[0]) for m in models], 0)
                acc = float((dev_p.argmax(1) == data.y[dv]).mean())
                log(f"  {kind} l2 {l2:g} cw {cw}: dev {acc:.4f} (best epochs {eps})")
                if best is None or acc > best[0]:
                    best = (acc, l2, cw, models, eps)
        acc, l2, cw, models, eps = best
        lg, al = {}, {}
        for sp in ("dev", "val"):
            preds = [predict_mix(m, data, data.idx[sp], mix_slots) for m in models]
            lg[sp] = np.log(np.mean([softmax_np(p[0]) for p in preds], 0) + 1e-12)
            al[sp] = np.mean([p[1] for p in preds], 0)
        readouts[kind] = {"slot": "all", "kind": "last", "l2": l2, "cw": cw, "dev_acc": acc, "epochs": eps,
                          "logits": lg}
        a_val = al["val"]
        ent = float(np.mean(-(a_val * np.log(a_val + 1e-12)).sum(1)))
        readouts[kind]["alpha_entropy_val"] = ent
        readouts[kind]["alpha_entropy_uniform"] = float(np.log(len(mix_slots)))
        top3 = np.argsort(-a_val.mean(0))[:3]
        readouts[kind]["alpha_top3_val"] = {names[mix_slots[j]]: round(float(a_val[:, j].mean()), 3) for j in top3}
        for j, s in enumerate(mix_slots):
            per_class = {data.labels[c]: float(a_val[data.y[va] == c, j].mean()) for c in range(k)
                         if np.any(data.y[va] == c)}
            alpha_rows.append({"readout": kind, "slot": names[s], "alpha_mean_val": float(a_val[:, j].mean()),
                               "alpha_sd_val": float(a_val[:, j].std()), **{f"c_{c}": v for c, v in per_class.items()}})
    with open(out / "alpha.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({kk for r in alpha_rows for kk in r}, key=lambda x: (x.startswith("c_"), x))
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(alpha_rows)

    # ---- 3. pre-registered primary choice (dev only)
    pool = ["final", "best_layer", "user_mean", "scalar_mix"]
    primary = max(pool, key=lambda r: readouts[r]["dev_acc"])
    if readouts["depth_attn"]["dev_acc"] >= readouts[primary]["dev_acc"] + DEPTH_MARGIN:
        primary = "depth_attn"
    res["primary"] = primary
    log(f"primary read-out (chosen on dev): {primary}")

    # ---- 4. val, once
    gen_pred, gen_logp = llm_answers(args.llm_answers, ri, data)
    y_va = data.y[va]
    gen_ok = gen_pred == y_va
    groups_va = data.groups[va]
    table = {"gen": {"val_acc": float(gen_ok.mean()), "val_macro_f1": macro_f1(gen_pred, y_va, k),
                     "val_ece_raw": ece(np.exp(gen_logp), y_va)}}
    val_rows = {data.ids[i]: {"id": data.ids[i], "gold": data.labels[data.y[i]], "gen": data.labels[gen_pred[j]]}
                for j, i in enumerate(va)}
    for name, r in readouts.items():
        lv, ld = r["logits"]["val"], r["logits"]["dev"]
        pred = lv.argmax(1)
        ok = pred == y_va
        T = fit_temperature(ld, data.y[dv])
        table[name] = {
            "dev_acc": r["dev_acc"], "val_acc": float(ok.mean()), "val_macro_f1": macro_f1(pred, y_va, k),
            "slot": r["slot"], "l2": r["l2"], "cw": r["cw"],
            "mcnemar_vs_gen": mcnemar_exact(ok, gen_ok), "diff_vs_gen_ci95": boot_diff(ok, gen_ok, groups_va),
            "val_ece_raw": ece(softmax_np(lv), y_va), "temperature_dev": T, "val_ece_temp": ece(softmax_np(lv / T), y_va),
            "val_recall": {data.labels[c]: float(ok[y_va == c].mean()) for c in range(k) if np.any(y_va == c)},
        }
        for extra in ("epochs", "alpha_entropy_val", "alpha_entropy_uniform", "alpha_top3_val"):
            if extra in r:
                table[name][extra] = r[extra]
        for j, i in enumerate(va):
            val_rows[data.ids[i]][name] = data.labels[pred[j]]
    table["gen"]["val_recall"] = {data.labels[c]: float(gen_ok[y_va == c].mean()) for c in range(k) if np.any(y_va == c)}

    # blend (exploratory): primary probe + the LLM's own label distribution, b by 5-fold CV over val conversations
    lp_probe = log_softmax_np(readouts[primary]["logits"]["val"])
    folds = group_folds(groups_va, seed=1)
    bgrid = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
    bl_pred = np.zeros(len(va), dtype=int)
    chosen = []
    for fo in range(5):
        a, bmask = folds != fo, folds == fo
        def acc_b(bb, m):
            return ((1 - bb) * lp_probe[m] + bb * gen_logp[m]).argmax(1) == y_va[m]
        bb = max(bgrid, key=lambda x: acc_b(x, a).mean())
        chosen.append(bb)
        bl_pred[bmask] = ((1 - bb) * lp_probe[bmask] + bb * gen_logp[bmask]).argmax(1)
    bl_ok = bl_pred == y_va
    table["blend"] = {"base": primary, "b_per_fold": chosen, "val_acc_cv": float(bl_ok.mean()),
                      "mcnemar_vs_gen": mcnemar_exact(bl_ok, gen_ok), "diff_vs_gen_ci95": boot_diff(bl_ok, gen_ok, groups_va)}
    for j, i in enumerate(va):
        val_rows[data.ids[i]]["blend"] = data.labels[bl_pred[j]]
    res["emotion"] = table
    p = table[primary]
    res["emotion_pass"] = bool(p["val_acc"] >= PASS_ACC and p["mcnemar_vs_gen"]["p"] < PASS_P)
    log(f"val: gen {table['gen']['val_acc']:.4f} | " + " | ".join(
        f"{n} {table[n]['val_acc']:.4f}" for n in readouts) + f" | blend(CV) {bl_ok.mean():.4f}")

    # ---- 5. profile: one-vs-rest logistic per slot, per-field threshold for exact match on dev
    log("profile: per-slot one-vs-rest probes ...")
    Y = torch.as_tensor(data.Y, device=args.device)

    def field_exact(P, rows, th):
        out_ = {}
        for f in data.fields:
            J = data.fidx[f]
            out_[f] = float(((P[:, J] >= th[f]) == (data.Y[rows][:, J] > 0.5)).all(1).mean())
        return out_

    prof = []
    best_prof = None
    for kind, n_s in (("last", S), ("mean", M)):
        for s in range(n_s):
            Xtr, Xdv = data.slot(kind, s, tr), data.slot(kind, s, dv)
            for l2 in L2_GRID:
                W, b = fit_linear(Xtr, Y[torch.as_tensor(tr, device=args.device)], l2, multilabel=True,
                                  max_iter=args.max_iter)
                P = torch.sigmoid(Xdv @ W + b).cpu().numpy()
                th = {}
                for f in data.fields:
                    J = data.fidx[f]
                    gold = data.Y[dv][:, J] > 0.5
                    th[f] = float(max(TH_GRID, key=lambda t: ((P[:, J] >= t) == gold).all(1).mean()))
                fe = field_exact(P, dv, th)
                score = float(np.mean(list(fe.values())))
                name = names[s] if kind == "last" else names[data.plan["mean_slots"][s]]
                prof.append({"kind": kind, "slot": name, "l2": l2, "dev_field_exact": score, **{f"dev_{f}": v for f, v in fe.items()}})
                if best_prof is None or score > best_prof[0]:
                    best_prof = (score, kind, s, l2, W, b, th, name)
    score, kind, s, l2, W, b, th, name = best_prof
    Pv = torch.sigmoid(data.slot(kind, s, va) @ W + b).cpu().numpy()
    fe_val = field_exact(Pv, va, th)
    whole = float(np.all([((Pv[:, data.fidx[f]] >= th[f]) == (data.Y[va][:, data.fidx[f]] > 0.5)).all(1)
                          for f in data.fields], axis=0).mean())
    empty = {f: float((data.Y[va][:, data.fidx[f]].sum(1) == 0).mean()) for f in data.fields}
    pred_empty = {f: float(((Pv[:, data.fidx[f]] >= th[f]).sum(1) == 0).mean()) for f in data.fields}
    res["profile"] = {"best": {"kind": kind, "slot": name, "l2": l2, "thresholds": th, "dev_field_exact": score},
                      "val_field_exact": fe_val, "val_field_exact_mean": float(np.mean(list(fe_val.values()))),
                      "val_whole_profile_exact": whole, "val_all_empty_field_exact": empty,
                      "val_predicted_empty": pred_empty, "baseline_cpu_classifier": args.profile_baseline,
                      "per_slot_dev": prof}
    res["profile_pass"] = bool(res["profile"]["val_field_exact_mean"] >= args.profile_baseline + PROFILE_MARGIN)
    for j, i in enumerate(va):
        val_rows[data.ids[i]]["profile"] = {f: [data.tags[jj][1] for jj in data.fidx[f] if Pv[j, jj] >= th[f]]
                                            for f in data.fields}
    log(f"profile: best {kind}/{name} l2 {l2:g}; val per-field exact {res['profile']['val_field_exact_mean']:.4f} "
        f"(baseline {args.profile_baseline}), whole {whole:.4f}")

    with open(out / "val_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in val_rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res["seconds"] = round(time.time() - t_start, 1)
    res["device"] = args.device
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(res, out / "report.md")
    log(f"EXP-029 emotion {'PASS' if res['emotion_pass'] else 'FAIL'} | profile {'PASS' if res['profile_pass'] else 'FAIL'} "
        f"| {res['seconds']:.0f} s -> {out}")


def write_report(res, path):
    e = res["emotion"]
    L = ["# EXP-029 results (auto-generated by scripts/exp029_probe.py)", "",
         f"Primary read-out (chosen on dev before val): **{res['primary']}**  ",
         f"Emotion: **{'PASS' if res['emotion_pass'] else 'FAIL'}** ({res['pass_rules']['emotion']})  ",
         f"Profile: **{'PASS' if res['profile_pass'] else 'FAIL'}** ({res['pass_rules']['profile']})", "",
         "| read-out | slot | dev acc | val acc | val macro-F1 | McNemar vs gen (only / only / p) | diff CI95 | ECE raw → T |",
         "|---|---|---|---|---|---|---|---|"]
    g = e["gen"]
    L.append(f"| gen (FP8 label) | – | – | {g['val_acc']:.4f} | {g['val_macro_f1']:.3f} | – | – | {g['val_ece_raw']:.3f} |")
    for n in ("final", "best_layer", "user_mean", "scalar_mix", "depth_attn"):
        r = e[n]
        m = r["mcnemar_vs_gen"]
        L.append(f"| {n} | {r['slot']} | {r['dev_acc']:.4f} | {r['val_acc']:.4f} | {r['val_macro_f1']:.3f} | "
                 f"{m['a_only']} / {m['b_only']} / {m['p']:.3g} | {r['diff_vs_gen_ci95'][0]:+.3f}…{r['diff_vs_gen_ci95'][1]:+.3f} | "
                 f"{r['val_ece_raw']:.3f} → {r['val_ece_temp']:.3f} (T {r['temperature_dev']:.2f}) |")
    b = e["blend"]
    L += [f"| blend ({b['base']} + gen logprobs, CV on val) | – | – | {b['val_acc_cv']:.4f} | – | "
          f"{b['mcnemar_vs_gen']['a_only']} / {b['mcnemar_vs_gen']['b_only']} / {b['mcnemar_vs_gen']['p']:.3g} | "
          f"{b['diff_vs_gen_ci95'][0]:+.3f}…{b['diff_vs_gen_ci95'][1]:+.3f} | – |", ""]
    oc = res["overfit_check"]
    L += [f"Overfit check at {oc['slot']}: train in-sample {oc['train_in_sample']:.3f}, train grouped CV "
          f"{oc['train_grouped_cv']:.3f}, dev {oc['dev']:.3f}.", ""]
    for n in ("scalar_mix", "depth_attn"):
        L.append(f"{n}: alpha entropy on val {e[n]['alpha_entropy_val']:.3f} (uniform {e[n]['alpha_entropy_uniform']:.3f}), "
                 f"heaviest depths {e[n]['alpha_top3_val']}, best epochs {e[n]['epochs']}, l2 {e[n]['l2']:g}, cw {e[n]['cw']}.  ")
    L += ["", "Layer curve (dev accuracy of the linear probe per slot): see layer_curve.csv; alpha per depth: alpha.csv.", ""]
    p = res["profile"]
    L += [f"Profile: best {p['best']['kind']}/{p['best']['slot']} (l2 {p['best']['l2']:g}, thresholds {p['best']['thresholds']}); "
          f"val per-field exact {p['val_field_exact_mean']:.4f} {json.dumps({k: round(v, 3) for k, v in p['val_field_exact'].items()})}, "
          f"whole profile {p['val_whole_profile_exact']:.4f}; all-empty {json.dumps({k: round(v, 3) for k, v in p['val_all_empty_field_exact'].items()})}; "
          f"baseline CPU classifier {p['baseline_cpu_classifier']}."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
