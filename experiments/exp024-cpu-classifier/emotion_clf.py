"""EXP-024b: CPU emotion classifier on the last user message(s); alone and blended with the LLM's label logprobs (5-fold CV on val 600)."""
import json, sys, random, time, importlib.util, math
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import data
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
tuner = load("tuner", "../../scripts/tune_emotion_bias.py")
tr, va, n100, t500 = data.load()
ids = t500 + n100; pts = [va[i] for i in ids]
t0 = time.time()
v1 = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, max_features=200000, sublinear_tf=True)
v2 = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
Xtr = hstack([v1.fit_transform([p["last"] for p in tr]), 0.5 * v2.fit_transform([p["prev"] for p in tr])]).tocsr()
labels = sorted({p["emotion"] for p in tr})
ytr = np.array([labels.index(p["emotion"]) for p in tr])
clf = LogisticRegression(C=4.0, max_iter=3000); clf.fit(Xtr, ytr)
Xv = hstack([v1.transform([p["last"] for p in pts]), 0.5 * v2.transform([p["prev"] for p in pts])]).tocsr()
Pc = clf.predict_proba(Xv); print(f"fit {len(tr)} pts in {time.time()-t0:.0f}s; labels {len(labels)}")
gold = np.array([labels.index(p["emotion"]) for p in pts])
lp = {json.loads(l)["id"]: json.loads(l) for l in open("../../outputs/exp022/lp_bf16.jsonl", encoding="utf-8")}
llm_lab = {json.loads(l)["id"]: json.loads(l)["emotion_label"] for f in ["tune500_filter_on_b0", "n100_filter_on_b0"] for l in open(f"../../outputs/exp022/bf16/{f}.jsonl", encoding="utf-8")}
yl = np.array([labels.index(llm_lab[i]) for i in ids])
def llm_logp(k):
    M = np.full((len(ids), len(labels)), -20.0)
    for r, i in enumerate(ids):
        for lab, v in tuner.label_scores(lp[i].get("label_top"), k).items(): M[r, labels.index(lab)] = v
        if not tuner.label_scores(lp[i].get("label_top"), k): M[r, yl[r]] = 0.0  # no logprobs: trust the parsed label
    return M
acc = lambda pred, r: float((pred[r] == gold[r]).mean())
rows = np.arange(len(ids)); t5, n1 = rows[:500], rows[500:]
print(f"LLM #2 (parsed)      tune500 {acc(yl, t5):.4f}  n100 {acc(yl, n1):.4f}  all {acc(yl, rows):.4f}")
cp = Pc.argmax(1); print(f"classifier alone     tune500 {acc(cp, t5):.4f}  n100 {acc(cp, n1):.4f}  all {acc(cp, rows):.4f}")
conv = [p["conv"] for p in pts]; convs = sorted(set(conv)); random.Random(1).shuffle(convs); fold = {c: k % 5 for k, c in enumerate(convs)}; fo = np.array([fold[c] for c in conv])
for k_top in (5, 20):
    M = llm_logp(k_top); logc = np.log(Pc + 1e-9)
    ws = np.arange(0, 1.51, 0.05); pred = np.zeros(len(ids), dtype=int); picks = []
    for k in range(5):
        trn, tst = rows[fo != k], rows[fo == k]
        w = max(ws, key=lambda w: acc((M + w * logc).argmax(1), trn)); picks.append(round(float(w), 2))
        pred[tst] = (M + w * logc).argmax(1)[tst]
    rng = np.random.default_rng(0); idx = {c: np.where(np.array(conv) == c)[0] for c in convs}
    boots = [np.concatenate([idx[c] for c in rng.choice(convs, len(convs))]) for _ in range(1000)]
    d = np.array([acc(pred, b) - acc(yl, b) for b in boots])
    print(f"blend top-{k_top:<2} (CV)    tune500 {acc(pred, t5):.4f}  n100 {acc(pred, n1):.4f}  all {acc(pred, rows):.4f}  Δ {acc(pred, rows)-acc(yl, rows):+.4f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  w per fold {picks}  changed {(pred != yl).sum()}")
