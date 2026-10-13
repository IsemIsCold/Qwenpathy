"""EXP-024a: LLM (#2) vs classifier vs hybrids on all 600 val points; 5-fold CV by conversation for any val-tuned parameter; bootstrap CI."""
import json, sys, random, importlib.util
import numpy as np
import data
spec = importlib.util.spec_from_file_location("ev", "../../scripts/eval.py"); sys.path.insert(0, "../../scripts")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
F = data.F
meta = json.load(open("P_val.json")); P = np.load("P_val.npy"); ids = meta["ids"]; tags = [tuple(t) for t in meta["tags"]]; thr = meta["thr"]
tr, va, n100, t500 = data.load()
rd = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()}
llm = {**rd("../../outputs/exp022/bf16/tune500_filter_on_b0.jsonl"), **rd("../../outputs/exp022/bf16/n100_filter_on_b0.jsonl")}
gold = [va[i]["profile"] for i in ids]; conv = [va[i]["conv"] for i in ids]
L = np.array([[t in (llm[i]["user_profile"].get(f) or []) for f, t in tags] for i in ids], dtype=float)
G = np.array([[t in (g.get(f) or []) for f, t in tags] for g in gold], dtype=int)
fidx = {f: [j for j, (ff, _) in enumerate(tags) if ff == f] for f in F}
def field_f1(pred_bin, rows, f):
    J = fidx[f]; tp = (pred_bin[np.ix_(rows, J)] * G[np.ix_(rows, J)]).sum(); fp = (pred_bin[np.ix_(rows, J)] * (1 - G[np.ix_(rows, J)])).sum(); fn = ((1 - pred_bin[np.ix_(rows, J)]) * G[np.ix_(rows, J)]).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
def avg_f1(pred_bin, rows): return np.mean([field_f1(pred_bin, rows, f) for f in F])
rows = np.arange(len(ids))
C = np.zeros_like(L)
for f in F:
    for j in fidx[f]: C[:, j] = P[:, j] >= thr[f]
# check against eval.py on the full set
def ev_avg(pred_bin, rows):
    preds = [{f: [tags[j][1] for j in fidx[f] if pred_bin[i, j]] for f in F} for i in rows]
    pairs = {f: [(ev.to_label_set(p[f]), ev.to_label_set(gold[i].get(f))) for p, i in zip(preds, rows)] for f in F}
    return ev.score_profile(pairs)["micro_f1_avg"]
print("sanity (eval.py vs local):", round(ev_avg(L, rows), 4), round(avg_f1(L, rows), 4))
variants = {"LLM #2": L, "classifier": C}
H = L.copy(); H[:, fidx["style"]] = C[:, fidx["style"]]; variants["hybrid: style from classifier"] = H
H2 = L.copy()
for f in ("style", "personality_traits"): H2[:, fidx[f]] = C[:, fidx[f]]
variants["hybrid: style+personality from classifier"] = H2
# blend: keep tag if a*LLM + (1-a)*p >= t, (a, t) per field picked by 5-fold CV over conversations
convs = sorted(set(conv)); random.Random(1).shuffle(convs); fold = {c: k % 5 for k, c in enumerate(convs)}
fo = np.array([fold[c] for c in conv]); B = np.zeros_like(L); grid = [(a, t) for a in np.arange(0, 1.01, 0.1) for t in np.arange(0.05, 1.0, 0.05)]
for k in range(5):
    trn, tst = rows[fo != k], rows[fo == k]
    for f in F:
        J = fidx[f]
        def bin_(a, t, r): out = np.zeros_like(L); out[np.ix_(r, J)] = (a * L[np.ix_(r, J)] + (1 - a) * P[np.ix_(r, J)]) >= t; return out
        a, t = max(grid, key=lambda g: field_f1(bin_(g[0], g[1], trn), trn, f))
        B[np.ix_(tst, J)] = (a * L[np.ix_(tst, J)] + (1 - a) * P[np.ix_(tst, J)]) >= t
variants["blend LLM+classifier (5-fold CV)"] = B
rng = np.random.default_rng(0); cl = np.array(convs); idx_by_conv = {c: np.where(np.array(conv) == c)[0] for c in convs}
boots = [np.concatenate([idx_by_conv[c] for c in rng.choice(cl, len(cl))]) for _ in range(1000)]
base = avg_f1(L, rows)
print(f"\n{'variant':<44}{'tune500':>9}{'n100':>8}{'all 600':>9}  Δ vs #2 [95% CI]   x15 pts   per field (600)")
t5, n1 = np.arange(500), np.arange(500, 600)
for name, V in variants.items():
    d = np.array([avg_f1(V, b) - avg_f1(L, b) for b in boots])
    per = {f[:5]: round(field_f1(V, rows, f), 3) for f in F}
    print(f"{name:<44}{avg_f1(V, t5):>9.4f}{avg_f1(V, n1):>8.4f}{avg_f1(V, rows):>9.4f}  {avg_f1(V, rows)-base:+.4f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  {15*(avg_f1(V, rows)-base):+.2f}   {per}")

def ev_exact(pred_bin, rows):
    preds = [{f: [tags[j][1] for j in fidx[f] if pred_bin[i, j]] for f in F} for i in rows]
    pairs = {f: [(ev.to_label_set(p[f]), ev.to_label_set(gold[i].get(f))) for p, i in zip(preds, rows)] for f in F}
    return ev.score_profile(pairs)["exact_match_avg"]
print("\nexact match (all 600):", {k: round(ev_exact(V, rows), 4) for k, V in variants.items()})
for f in F:
    J = fidx[f]
    def bin_(a, t, r): out = np.zeros_like(L); out[np.ix_(r, J)] = (a * L[np.ix_(r, J)] + (1 - a) * P[np.ix_(r, J)]) >= t; return out
    picks = []
    for k in range(5):
        trn = rows[fo != k]; picks.append(max(grid, key=lambda g: field_f1(bin_(g[0], g[1], trn), trn, f)))
    print(f, "fold picks (a, t):", [(round(a, 1), round(t, 2)) for a, t in picks])
print("\ncross-check with eval.py score_profile (all 600):", {k: round(ev_avg(V, rows), 4) for k, V in variants.items()})
