"""EXP-025: final blend parameters (all 600 val points) + export of the classifier for numpy-only inference."""
import json, pickle, sys, importlib.util
import numpy as np
import data
spec = importlib.util.spec_from_file_location("ev", "../../scripts/eval.py"); sys.path.insert(0, "../../scripts")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
F = data.F
meta = json.load(open("P_val.json")); P = np.load("P_val.npy"); ids = meta["ids"]; tags = [tuple(t) for t in meta["tags"]]
tr, va, n100, t500 = data.load()
rd = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()}
llm = {**rd("../../outputs/exp022/bf16/tune500_filter_on_b0.jsonl"), **rd("../../outputs/exp022/bf16/n100_filter_on_b0.jsonl")}
L = np.array([[t in (llm[i]["user_profile"].get(f) or []) for f, t in tags] for i in ids], dtype=float)
G = np.array([[t in (va[i]["profile"].get(f) or []) for f, t in tags] for i in ids], dtype=int)
fidx = {f: [j for j, (ff, _) in enumerate(tags) if ff == f] for f in F}
def f1(B, f):
    J = fidx[f]; tp = (B[:, J] * G[:, J]).sum(); fp = (B[:, J] * (1 - G[:, J])).sum(); fn = ((1 - B[:, J]) * G[:, J]).sum(); return 2 * tp / (2 * tp + fp + fn)
grid = [(round(a, 2), round(t, 2)) for a in np.arange(0, 1.01, 0.1) for t in np.arange(0.05, 1.0, 0.05)]
blend = {}
for f in F:
    J = fidx[f]
    scored = sorted(((f1((a * L + (1 - a) * P >= t).astype(float), f), a, t) for a, t in grid), reverse=True)
    blend[f] = [scored[0][1], scored[0][2]]
    print(f, "best (a, t) on all 600:", blend[f], "F1", round(scored[0][0], 4), "| next:", [(round(s, 4), a, t) for s, a, t in scored[1:4]])
B = np.zeros_like(L)
for f in F:
    a, t = blend[f]; J = fidx[f]; B[:, J] = (a * L[:, J] + (1 - a) * P[:, J]) >= t
print("in-sample avg F1 with final params:", round(np.mean([f1(B, f) for f in F]), 4), "(CV estimate was 0.4960; LLM 0.4609)")
# export
m = pickle.load(open("profile_clf.pkl", "rb")); vec = m["vec"]
vocab = [None] * len(vec.vocabulary_)
for g, j in vec.vocabulary_.items(): vocab[j] = g
n_feat = len(vocab) + 1  # + n_user/20
coef = np.zeros((len(tags), n_feat), dtype=np.float32); bias = np.full(len(tags), -30.0, dtype=np.float64)
for j, mdl in enumerate(m["models"]):
    if mdl is not None: coef[j] = mdl.coef_[0]; bias[j] = mdl.intercept_[0]
half = coef.shape[0] // 2  # two files, each < 30 MB
np.savez_compressed("../../participant/profile_clf/profile_clf_1.npz", idf=vec.idf_.astype(np.float64), bias=bias, coef=coef[:half])
np.savez_compressed("../../participant/profile_clf/profile_clf_2.npz", coef=coef[half:])
json.dump({"version": "exp025-v1", "vocab": vocab, "tags": [list(t) for t in tags], "ngram_range": [1, 3], "max_chars": 3000,
           "n_user_cap": 20, "trained_on": "train_public.jsonl, 16,750 points (90% of conversations, seed 0); thresholds/blend from val 600",
           "blend": blend, "weight_files": ["profile_clf_1.npz", "profile_clf_2.npz"]}, open("../../participant/profile_clf/profile_clf.json", "w", encoding="utf-8"), ensure_ascii=False)
import os; print({f: round(os.path.getsize(f) / 1e6, 1) for f in ("../../participant/profile_clf/profile_clf_1.npz", "../../participant/profile_clf/profile_clf_2.npz", "../../participant/profile_clf/profile_clf.json")}, "MB")
