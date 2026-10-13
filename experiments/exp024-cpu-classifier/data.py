"""EXP-024 data: points from train_public / val_public in the test-input shape."""
import json, random
U = "../.."
F = ["personality_traits", "interests", "style"]

def points(path):
    out = []
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        turns = sorted(c["turns"], key=lambda t: t["turn_id"])
        for p in c["prediction_points"]:
            hist = [t for t in turns if t["turn_id"] <= p["target_user_turn_id"]]
            users = [t["content"] for t in hist if t["role"] == "user"]
            out.append({"id": f'{c["conversation_id"]}_t{p["target_user_turn_id"]}', "conv": c["conversation_id"],
                        "last": users[-1] if users else "", "prev": users[-2] if len(users) > 1 else "",
                        "users": "\n".join(users)[-3000:], "n_user": len(users),
                        "emotion": p["target"]["emotion_label"], "profile": p["target"]["user_profile"]})
    return out

def load():
    tr = points(f"{U}/data/train_public.jsonl")
    va = {p["id"]: p for p in points(f"{U}/data/val_public.jsonl")}
    rd = lambda f: [json.loads(l)["id"] for l in open(f, encoding="utf-8-sig") if l.strip()]
    n100, t500 = rd(f"{U}/data/val_gold_n100.jsonl"), rd(f"{U}/data/val_gold_tune500.jsonl")
    return tr, va, n100, t500
