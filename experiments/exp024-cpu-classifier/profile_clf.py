"""EXP-024a: CPU profile classifier (char TF-IDF + one-vs-rest LR), thresholds tuned on held-out TRAIN only."""
import json, sys, random, time, importlib.util
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import data
sys.path.insert(0, "../../scripts")
spec = importlib.util.spec_from_file_location("ev", "../../scripts/eval.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
F = data.F
tr, va, n100, t500 = data.load()
convs = sorted({p["conv"] for p in tr}); random.Random(0).shuffle(convs)
hold = set(convs[: len(convs) // 10])
fit = [p for p in tr if p["conv"] not in hold]; dev = [p for p in tr if p["conv"] in hold]
tags = [(f, t) for f in F for t in sorted(ev.PROFILE_VOCAB[f])]
Y = lambda pts: np.array([[t in (p["profile"].get(f) or []) for f, t in tags] for p in pts], dtype=int)
t0 = time.time()
vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3, max_features=300000, sublinear_tf=True)
Xfit = vec.fit_transform([p["users"] for p in fit])
def X(pts):
    base = vec.transform([p["users"] for p in pts])
    nu = csr_matrix(np.array([[min(p["n_user"], 20) / 20] for p in pts]))
    return hstack([base, nu]).tocsr()
Xfit = hstack([Xfit, csr_matrix(np.array([[min(p["n_user"], 20) / 20] for p in fit]))]).tocsr()
Yfit = Y(fit)
models = []
for j, (f, t) in enumerate(tags):
    if Yfit[:, j].sum() < 5:
        models.append(None); continue
    m = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear"); m.fit(Xfit, Yfit[:, j]); models.append(m)
print(f"fit {len(fit)} pts, {len(tags)} tags, {time.time()-t0:.0f}s", flush=True)
def proba(pts):
    Xp = X(pts)
    return np.column_stack([m.predict_proba(Xp)[:, 1] if m is not None else np.zeros(len(pts)) for m in models])
Pdev = proba(dev)
def to_pred(P, thr):
    return [{f: [t for j, (ff, t) in enumerate(tags) if ff == f and P[i, j] >= thr[f]] for f in F} for i in range(P.shape[0])]
def f1(preds, golds):
    pairs = {f: [(ev.to_label_set(pr[f]), ev.to_label_set(g.get(f))) for pr, g in zip(preds, golds)] for f in F}
    s = ev.score_profile(pairs); return s["micro_f1_avg"], s["exact_match_avg"], {f: s[f]["micro_f1"] for f in F}
# per-field threshold tuned on held-out train
thr = {}
for f in F:
    best = None
    for th in np.arange(0.05, 0.9, 0.025):
        tt = {g: (th if g == f else 2) for g in F}
        pairs = [(ev.to_label_set(p[f]), ev.to_label_set(d["profile"].get(f))) for p, d in zip(to_pred(Pdev, tt), dev)]
        s = ev.score_profile({g: pairs if g == f else [(frozenset(), frozenset())] * len(pairs) for g in F})[f]["micro_f1"]
        if best is None or s > best[0]: best = (s, th)
    thr[f] = round(float(best[1]), 3)
print("thresholds (held-out train):", thr, "dev F1", round(f1(to_pred(Pdev, thr), [d["profile"] for d in dev])[0], 4))
# validation: tune500 + n100, never used above
rd = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()}
llm = {**rd("../../outputs/exp022/bf16/tune500_filter_on_b0.jsonl"), **rd("../../outputs/exp022/bf16/n100_filter_on_b0.jsonl")}
out = {}
for name, ids in [("tune500", t500), ("n100", n100)]:
    pts = [va[i] for i in ids]; P = proba(pts); golds = [p["profile"] for p in pts]
    clf = to_pred(P, thr)
    l = [llm[i]["user_profile"] for i in ids]
    print(f"{name:<8} LLM #2  F1 {f1(l, golds)[0]:.4f} exact {f1(l, golds)[1]:.4f} {({k: round(v,3) for k,v in f1(l, golds)[2].items()})}")
    print(f"{name:<8} classif F1 {f1(clf, golds)[0]:.4f} exact {f1(clf, golds)[1]:.4f} {({k: round(v,3) for k,v in f1(clf, golds)[2].items()})}")
    out[name] = (ids, P)
np.save("P_val.npy", np.vstack([out["tune500"][1], out["n100"][1]])); json.dump({"ids": t500 + n100, "tags": tags, "thr": thr}, open("P_val.json", "w"))
import pickle; pickle.dump({"vec": vec, "models": models, "tags": tags, "thr": thr}, open("profile_clf.pkl", "wb"))
