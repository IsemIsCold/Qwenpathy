#!/usr/bin/env python3
"""
tests/stub_vllm.py -- fake vLLM OpenAI server for EXP-023 tests (no GPU, stdlib only).

Answers each sample from gold (STUB_GOLD = comma-separated gold files), but says
"anxiety" on ~40% of non-anxiety points with the gold label 0.1-1.5 logprob behind,
so a bias b recovers exactly the points whose gap is < b. Char-level tokens, except the
emotion label: its first token is the real Qwen3 first token ("an", "f", "rel", ...).

Logprobs are returned only when the request asks for them (top_logprobs alternatives,
sorted). Every request's option keys are appended to STUB_LOG (JSON lines) if set.
STUB_REPLY_FRACTION < 1 cuts each reply (to make ROUGE-L drop).

    python tests/stub_vllm.py --port 8000
Also usable as a fake `vllm` binary: `vllm serve MODEL ... --port N ...` (args logged to FAKE_VLLM_ARGS_LOG).
"""

import hashlib
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIRST = {"joy": "joy", "gratitude": "gr", "relaxed": "rel", "care": "care", "pride": "pr",
         "neutral": "neutral", "surprise": "sur", "mixed": "mixed", "sadness": "sad",
         "loneliness": "lon", "anxiety": "an", "anger": "anger", "fear": "f", "disgust": "dis",
         "shame": "sh", "helplessness": "help"}
ID_RE = re.compile(r"样本 ID: (\S+)")


def h(text, mod):
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % mod


def load_gold():
    gold = {}
    for path in filter(None, os.environ.get("STUB_GOLD", "").split(",")):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    gold[g["id"]] = g
    return gold


GOLD = load_gold()
FRACTION = float(os.environ.get("STUB_REPLY_FRACTION", "1"))


def plan(sid):
    """(said label, {label: logprob} at the label's first token)."""
    g = GOLD[sid]["emotion_label"]
    filler = "disgust" if g != "disgust" else "pride"  # a far-behind third label
    if g != "anxiety" and h(sid, 10) < 4:
        gap = 0.1 + h(sid + "gap", 15) / 10  # 0.1 .. 1.5
        return "anxiety", {"anxiety": -0.3, g: -0.3 - gap, filler: -6.0}
    other = "fear" if g != "fear" else "sadness"
    return g, {g: -0.05, other: -3.0, filler: -7.0}


def answer(sid):
    g = GOLD[sid]
    said, alts = plan(sid)
    reply = g.get("response_text") or "好的"
    reply = reply[:max(1, int(len(reply) * FRACTION))]
    head = json.dumps({"response_text": reply}, ensure_ascii=False)[:-1] + ', "emotion_label": "'
    tail = '", "user_profile": ' + json.dumps(g.get("user_profile") or {}, ensure_ascii=False) + ', "memory_refs": []}'
    first = FIRST[said]
    tokens = [(c, None) for c in head] + [(first, alts), (said[len(first):], None)] + [(c, None) for c in tail]
    tokens = [(t, a) for t, a in tokens if t]
    return "".join(t for t, _ in tokens), tokens


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send(200, {}) if self.path == "/health" else self.send(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        if self.path != "/v1/chat/completions" or req.get("model") != "model":
            return self.send(404, {"error": f"path {self.path} / model {req.get('model')}"})
        sid = ID_RE.search(req["messages"][-1]["content"]).group(1)
        if os.environ.get("STUB_LOG"):
            with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": sid, "keys": sorted(k for k in req if k not in ("messages",)),
                                    "logprobs": req.get("logprobs"), "top_logprobs": req.get("top_logprobs")}) + "\n")
        content, tokens = answer(sid)
        choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        if req.get("logprobs"):
            k = int(req.get("top_logprobs") or 0)
            out = []
            for tok, alts in tokens:
                if alts is None:
                    top = [{"token": tok, "logprob": -0.01}]
                    out.append({"token": tok, "logprob": -0.01, "top_logprobs": top[:k]})
                else:
                    ranked = sorted(((FIRST[l], lp) for l, lp in alts.items()), key=lambda x: -x[1])
                    out.append({"token": tok, "logprob": dict(ranked)[tok],
                                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in ranked[:k]]})
            choice["logprobs"] = {"content": out}
        self.send(200, {"id": "x", "object": "chat.completion", "choices": [choice]})


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "serve":  # fake `vllm serve ...`
        if os.environ.get("FAKE_VLLM_ARGS_LOG"):
            with open(os.environ["FAKE_VLLM_ARGS_LOG"], "a", encoding="utf-8") as f:
                f.write(json.dumps(argv) + "\n")
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8000
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
