import json
ex = {json.loads(l)["id"] for f in ("data/val_input_n50.jsonl", "data/val_input_n50b.jsonl")
      for l in open(f, encoding="utf-8")}
rows = [json.loads(l) for l in open("data/val_input.jsonl", encoding="utf-8")]
rows = [r for r in rows if r["id"] not in ex]
rows.sort(key=lambda r: sum(len(t["content"]) for t in r["history"]), reverse=True)
with open("data/val_input_long5.jsonl", "w", encoding="utf-8") as f:
    for r in rows[:5]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")