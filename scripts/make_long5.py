"""T9 stress file: the 5 longest val points that are not in n50/n50b.

Run from the repo root:  python make_long5.py
Output: data/val_input_long5.jsonl  (upload to /mnt/storage/val/ on Migu)
"""
import json

EXCLUDE_FILES = ("data/val_input_n50.jsonl", "data/val_input_n50b.jsonl")
SOURCE = "data/val_input.jsonl"
OUT = "data/val_input_long5.jsonl"

exclude = set()
for path in EXCLUDE_FILES:
    with open(path, encoding="utf-8") as f:
        for line in f:
            exclude.add(json.loads(line)["id"])

with open(SOURCE, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
rows = [r for r in rows if r["id"] not in exclude]


def chars(r):
    return sum(len(t["content"]) for t in r["history"])


rows.sort(key=chars, reverse=True)
top = rows[:5]

with open(OUT, "w", encoding="utf-8") as f:
    for r in top:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

for r in top:
    print(r["id"], chars(r), "chars")
print("wrote", OUT)
