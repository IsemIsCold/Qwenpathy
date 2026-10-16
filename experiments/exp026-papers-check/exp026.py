#!/usr/bin/env python3
"""
EXP-026 (offline, CPU only): checks of the ideas from the "Research proposal" papers and Zakaria's P2 package,
on data we already hold. Nothing here needs a GPU.

    python experiments/exp026-papers-check/exp026.py [--data data] [--outputs outputs]

Needs: data/train_public.jsonl, data/val_public.jsonl, data/val_gold_{tune500,n100}.jsonl,
outputs/exp022/lp_bf16.jsonl (#2 raw answers + label logprobs on the 600 points),
outputs/exp023/lp_fp8_t10.jsonl (FP8 raw answers, = #3's model), participant/profile_clf/.

1. Profile = trait? Gold profile consistency inside a conversation.
2. Emotion = state? Label persistence, and a "state inertia" prior (oracle previous label) on the LLM's label logprobs.
3. Profile pooling across the points of one conversation (classifier evidence): causal / all / last.
4. Zakaria A1: tag filter off, now with the EXP-025 blend.
5. FP8 anxiety bias, honest 5-fold CV estimate (EXP-023 only had in-sample).
6. Big Five structure: do opposite tags co-occur in gold; does our output ever contradict itself?
7. Zakaria B1: ceiling of rare-class oversampling on the 600 points.
Every val-tuned parameter is picked by 5-fold CV over val conversations (same folds as EXP-024); 95% CIs from
1,000 bootstrap resamples of conversations.
"""

import argparse
import collections
import importlib.util
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(REPO / "data"))
ap.add_argument("--outputs", default=str(REPO / "outputs"))
args = ap.parse_args()
DATA, OUT = Path(args.data), Path(args.outputs)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ri = load("ri", REPO / "participant/run_inference.py")
sys.path.insert(0, str(REPO / "scripts"))
ev = load("ev", REPO / "scripts/eval.py")
F = list(ri.PROFILE_LABELS)
clf = ri.ProfileClassifier(REPO / "participant/profile_clf")
tags = clf.tags
fidx = {f: [j for j, (ff, _) in enumerate(tags) if ff == f] for f in F}
RARE = {"surprise", "care", "disgust", "joy", "pride"}
LLMS = (("#2 bf16", OUT / "exp022/lp_bf16.jsonl"), ("#3 FP8", OUT / "exp023/lp_fp8_t10.jsonl"))
summary = {}


def read_conversations(path):
    pts = {}
    for line in open(path, encoding="utf-8-sig"):
        if not line.strip():
            continue
        c = json.loads(line)
        turns = sorted(c["turns"], key=lambda t: t["turn_id"])
        prev = None
        for p in sorted(c["prediction_points"], key=lambda p: p["target_user_turn_id"]):
            k = p["target_user_turn_id"]
            pid = f'{c["conversation_id"]}_t{k}'
            pts[pid] = {"id": pid, "conv": c["conversation_id"], "turn": k,
                        "history": [t for t in turns if t["turn_id"] <= k],
                        "profile": p["target"]["user_profile"], "emotion": p["target"]["emotion_label"], "prev_emotion": prev}
            prev = p["target"]["emotion_label"]
    return pts


train, val = read_conversations(DATA / "train_public.jsonl"), read_conversations(DATA / "val_public.jsonl")
by_conv = collections.defaultdict(list)
for i in val:
    by_conv[val[i]["conv"]].append(i)
for c in by_conv:
    by_conv[c].sort(key=lambda i: val[i]["turn"])
rd = lambda p: [json.loads(l)["id"] for l in open(p, encoding="utf-8-sig") if l.strip()]
ids = rd(DATA / "val_gold_tune500.jsonl") + rd(DATA / "val_gold_n100.jsonl")  # = order of EXP-024's reference file
rows, t500, n100 = np.arange(len(ids)), np.arange(500), np.arange(500, 600)
conv = [val[i]["conv"] for i in ids]
convs = sorted(set(conv))
random.Random(1).shuffle(convs)
fold = np.array([{c: k % 5 for k, c in enumerate(convs)}[c] for c in conv])
rng = np.random.default_rng(0)
ibc = {c: np.where(np.array(conv) == c)[0] for c in convs}
boots = [np.concatenate([ibc[c] for c in rng.choice(np.array(convs), len(convs))]) for _ in range(1000)]


def ci(delta_fn):
    d = np.array([delta_fn(b) for b in boots])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def raw_answers(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        pred, _ = ri.parse_prediction(r["id"], r["content"])
        alts = r["label_top"] if isinstance(r["label_top"], list) else []
        out[r["id"]] = {"pred": pred, "scores": ri.label_scores(alts[:5]), "scores20": ri.label_scores(alts)}
    return out


# ---------------------------------------------------------------- 1. profile = trait?
print("\n1. Profile = trait? (gold, consecutive points of one conversation)")
for name, pts in (("train", train), ("val", val)):
    c = collections.Counter()
    groups = collections.defaultdict(list)
    for i, p in pts.items():
        groups[p["conv"]].append(p)
    for g in groups.values():
        g.sort(key=lambda p: p["turn"])
        for a, b in zip(g, g[1:]):
            pa, pb = [set(a["profile"].get(f) or []) for f in F], [set(b["profile"].get(f) or []) for f in F]
            c["same" if pa == pb else "grows" if all(x <= y for x, y in zip(pa, pb)) else "shrinks" if all(x >= y for x, y in zip(pa, pb)) else "other"] += 1
    n = sum(c.values())
    print(f"   {name}: {n} pairs  " + "  ".join(f"{k} {v / n:.1%}" for k, v in c.most_common()))
    summary[f"profile_pairs_{name}"] = {k: v / n for k, v in c.items()}
G = np.array([[t in (val[i]["profile"].get(f) or []) for f, t in tags] for i in ids], dtype=int)


def field_f1(B, r, f):
    J = fidx[f]
    g, b = G[np.ix_(r, J)], B[np.ix_(r, J)]
    tp, fp, fn = (b * g).sum(), (b * (1 - g)).sum(), ((1 - b) * g).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


avg = lambda B, r: float(np.mean([field_f1(B, r, f) for f in F]))


def official(B, r):  # cross-check with eval.py's scorer
    preds = [{f: [tags[j][1] for j in fidx[f] if B[i, j]] for f in F} for i in r]
    pairs = {f: [(ev.to_label_set(p[f]), ev.to_label_set(val[ids[i]]["profile"].get(f))) for p, i in zip(preds, r)] for f in F}
    s = ev.score_profile(pairs)
    return s["micro_f1_avg"], s["exact_match_avg"]


Glast = np.array([[t in (val[by_conv[val[i]["conv"]][-1]]["profile"].get(f) or []) for f, t in tags] for i in ids], dtype=float)
f1, ex = official(Glast, rows)
print(f"   gold of the conversation's LAST point used for every point (600): micro-F1 {f1:.4f}, exact {ex:.4f}")
summary["gold_last_as_prediction"] = {"micro_f1": f1, "exact": ex}

# ---------------------------------------------------------------- 2. emotion = state?
print("\n2. Emotion = state?")
same = np.mean([p["emotion"] == p["prev_emotion"] for p in val.values() if p["prev_emotion"]])
marg = collections.Counter(p["emotion"] for p in train.values())
chance = sum((v / sum(marg.values())) ** 2 for v in marg.values())
print(f"   val: label = previous point's label {same:.1%} (chance with train frequencies {chance:.1%})")
trans = collections.Counter((p["prev_emotion"], p["emotion"]) for p in train.values() if p["prev_emotion"])
prev_n = collections.Counter(p["prev_emotion"] for p in train.values() if p["prev_emotion"])
E = sorted(marg)
pmi = lambda e, p: math.log((trans[(p, e)] + 1) / (prev_n[p] + len(E))) - math.log((marg[e] + 1) / (sum(marg.values()) + len(E)))
gold_e = {i: val[i]["emotion"] for i in ids}
answers = {name: raw_answers(path) for name, path in LLMS}


def cv_pick(grid, predict):
    pred = {}
    picks = []
    for k in range(5):
        trn, tst = rows[fold != k], rows[fold == k]
        w = max(grid, key=lambda w: (np.mean([predict(ids[j], w) == gold_e[ids[j]] for j in trn]), -w))
        picks.append(w)
        for j in tst:
            pred[j] = predict(ids[j], w)
    return pred, picks


for name, ans in answers.items():
    def inertia(i, w, ans=ans):
        sc, label = ans[i]["scores20"], ans[i]["pred"]["emotion_label"]
        prev = val[i]["prev_emotion"]  # ORACLE: gold label of the previous point
        if w == 0 or not sc or prev is None:
            return label
        floor = min(sc.values()) - 1.0
        return max(E, key=lambda e: sc.get(e, floor) + w * pmi(e, prev))
    base = np.mean([inertia(i, 0) == gold_e[i] for i in ids])
    pred, picks = cv_pick([0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0], inertia)
    acc = np.mean([pred[j] == gold_e[ids[j]] for j in rows])
    print(f"   {name}: greedy {base:.4f}; + oracle previous-label prior (CV) {acc:.4f}  ({acc - base:+.4f}), w picks {picks}")
    summary[f"inertia_{name}"] = {"greedy": base, "cv": acc}

# ---------------------------------------------------------------- 3 + 4. profile pooling, filter off
print("\n3/4. Profile: blend variants (5-fold CV), Δ vs the #3 blend (own point only)")
Pall = {i: clf.probabilities(val[i]) for i in val}
P = np.vstack([Pall[i] for i in ids])
ref = REPO / "experiments/exp024-cpu-classifier/P_val_reference.npy"
if ref.exists():
    print(f"   classifier = EXP-024 reference: max |diff| {np.abs(P - np.load(ref)).max():.1e}")
inset = set(ids)
pool = {
    "causal": np.vstack([np.mean([Pall[s] for s in by_conv[val[i]["conv"]] if val[s]["turn"] <= val[i]["turn"]], axis=0) for i in ids]),
    "all": np.vstack([np.mean([Pall[s] for s in by_conv[val[i]["conv"]]], axis=0) for i in ids]),
    "last": np.vstack([Pall[by_conv[val[i]["conv"]][-1]] for i in ids]),
    "last_in_600": np.vstack([Pall[[s for s in by_conv[val[i]["conv"]] if s in inset][-1]] for i in ids]),
}
A, T, LAM = np.round(np.arange(0, 1.01, 0.1), 2), np.round(np.arange(0.05, 1.0, 0.05), 2), [0, 0.25, 0.5, 0.75, 1.0]


def cv_blend(L, poolP=None):
    B = np.zeros_like(L)
    for k in range(5):
        trn, tst = rows[fold != k], rows[fold == k]
        for f in F:
            J = fidx[f]
            best = None
            for lam in ([1.0] if poolP is None else LAM):
                Q = P[:, J] if poolP is None else lam * P[:, J] + (1 - lam) * poolP[:, J]
                for a in A:
                    S = a * L[:, J] + (1 - a) * Q
                    for t in T:
                        Bt, g = S[trn] >= t, G[np.ix_(trn, J)]
                        tp, fp, fn = (Bt * g).sum(), (Bt * (1 - g)).sum(), ((1 - Bt) * g).sum()
                        sc = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
                        if best is None or sc > best[0] + 1e-12:
                            best = (sc, lam, a, t)
            _, lam, a, t = best
            Q = P[np.ix_(tst, J)] if poolP is None else lam * P[np.ix_(tst, J)] + (1 - lam) * poolP[np.ix_(tst, J)]
            B[np.ix_(tst, J)] = (a * L[np.ix_(tst, J)] + (1 - a) * Q) >= t
    return B


saved_filter = ri.OVERPREDICTED_TAGS
for name, path in LLMS:
    ans = answers[name]
    L = np.array([[t in (ans[i]["pred"]["user_profile"].get(f) or []) for f, t in tags] for i in ids], dtype=float)
    ri.OVERPREDICTED_TAGS = {}
    off = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        off[r["id"]] = ri.parse_prediction(r["id"], r["content"])[0]["user_profile"]
    ri.OVERPREDICTED_TAGS = saved_filter
    Loff = np.array([[t in (off[i].get(f) or []) for f, t in tags] for i in ids], dtype=float)
    base = cv_blend(L)
    variants = {
        "LLM alone": L,
        "blend, own point (= #3)": base,
        "blend, filter off (Zakaria A1)": cv_blend(Loff),
        "blend + pool causal (past points)": cv_blend(L, pool["causal"]),
        "blend + pool all points": cv_blend(L, pool["all"]),
        "blend + pool last point [FUTURE]": cv_blend(L, pool["last"]),
        "  same, only points in the 600": cv_blend(L, pool["last_in_600"]),
    }
    print(f"   {name}  ({int((Loff != L).sum())} LLM tags change with the filter off)")
    print(f"   {'variant':<36}{'tune500':>8}{'n100':>8}{'all600':>8}{'exact':>8}   Δ all600 [95% CI]")
    summary[f"profile_{name}"] = {}
    for vname, V in variants.items():
        lo, hi = ci(lambda b: avg(V, b) - avg(base, b))
        f1, ex = official(V, rows)
        assert abs(f1 - avg(V, rows)) < 1e-4  # eval.py rounds to 4 decimals
        print(f"   {vname:<36}{avg(V, t500):>8.4f}{avg(V, n100):>8.4f}{f1:>8.4f}{ex:>8.4f}   {f1 - avg(base, rows):+.4f} [{lo:+.3f}, {hi:+.3f}]")
        summary[f"profile_{name}"][vname.strip()] = {"tune500": avg(V, t500), "n100": avg(V, n100), "all600": f1, "exact600": ex, "ci": [lo, hi]}

# ---------------------------------------------------------------- 5. FP8 anxiety bias, CV
print("\n5. Anxiety bias (EXP-022/023 rule, top-5 alternatives), 5-fold CV")
for name, ans in answers.items():
    bias = lambda i, b, ans=ans: ri.apply_emotion_bias(ans[i]["pred"]["emotion_label"], ans[i]["scores"], b)
    base = np.mean([bias(i, 0) == gold_e[i] for i in ids])
    pred, picks = cv_pick([0, 0.25, 0.5, 0.65, 0.75, 1.0, 1.25, 1.5], bias)
    acc = np.mean([pred[j] == gold_e[ids[j]] for j in rows])
    lo, hi = ci(lambda b: np.mean([pred[j] == gold_e[ids[j]] for j in b]) - np.mean([bias(ids[j], 0) == gold_e[ids[j]] for j in b]))
    print(f"   {name}: no bias {base:.4f}, bias CV {acc:.4f} ({acc - base:+.4f}, 95% CI {lo:+.3f}..{hi:+.3f}), b picks {picks}")
    summary[f"bias_{name}"] = {"none": base, "cv": acc, "ci": [lo, hi]}

# ---------------------------------------------------------------- 6. Big Five structure
print("\n6. Opposite tags in gold (train) and in our blend output (600)")
single, co = collections.Counter(), collections.Counter()
for p in train.values():
    for f in F:
        vals = sorted(p["profile"].get(f) or [])
        single.update(vals)
        co.update(itertools.combinations(vals, 2))
n = len(train)
for a, b in (("emotionally_stable", "sensitive"), ("introverted", "extroverted"), ("brief", "detailed"), ("formal", "colloquial"),
             ("high_emotional_expression", "low_emotional_expression"), ("direct", "indirect")):
    k = co[tuple(sorted((a, b)))]
    print(f"   {a} + {b}: together {k}, expected if independent {single[a] * single[b] / n:.0f}")
for name, ans in answers.items():
    both = sum(1 for i in ids if {"brief", "detailed"} <= set(clf.apply(ans[i]["pred"]["user_profile"], val[i])["style"]))
    print(f"   {name} blend outputs with brief + detailed together: {both}")

# ---------------------------------------------------------------- 7. rare classes
print("\n7. Rare classes (surprise, care, disgust, joy, pride)")
allv = collections.Counter(p["emotion"] for p in val.values())
print(f"   share of all {len(val)} val points: {sum(allv[e] for e in RARE) / len(val):.1%}")
for name, ans in answers.items():
    wrong = [i for i in ids if ans[i]["pred"]["emotion_label"] != gold_e[i]]
    rare_wrong = sum(gold_e[i] in RARE for i in wrong)
    anx = sum(ans[i]["pred"]["emotion_label"] == "anxiety" for i in wrong)
    print(f"   {name}: {len(wrong)} errors on 600; gold rare {rare_wrong} (fixing all = +{rare_wrong / 6:.1f} acc points = +{25 * rare_wrong / 600:.2f} contest points); predicted anxiety {anx}")
    summary[f"rare_{name}"] = {"errors": len(wrong), "rare_errors": rare_wrong, "anxiety_errors": anx}

Path(__file__).with_name("summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
print(f"\nwrote {Path(__file__).with_name('summary.json')}")
