#!/usr/bin/env python3
"""Run local Transformers or OpenAI-compatible inference and timing in one pass.

Public CLI contract: python run_inference.py TEST_FILE RESULT_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "inference_config.json"
EMOTIONS = {
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
}
PROFILE_LABELS = {
    "personality_traits": {"extroverted", "introverted", "open", "conservative", "high_conscientiousness", "casual", "agreeable", "assertive", "emotionally_stable", "sensitive"},
    "interests": {"study_exam", "programming_technology", "reading_writing", "film_animation", "music", "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design"},
    "style": {"brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous", "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user"},
}
MEMORY_RE = re.compile(r"^mem_[0-9]{6}$")
SYSTEM_PROMPT = """你正在参加数字情感陪伴比赛。根据截止到最新用户消息的对话历史，完成下一轮 Assistant 回复预测。

只输出一个 JSON object，不要输出 Markdown 或解释。object 必须且只能包含：
- response_text: 非空中文回复，应自然、贴合上下文和用户偏好；
- emotion_label: 最新用户消息的情绪，只能从 {emotions} 中选择；
- user_profile: object，包含 personality_traits、interests、style 三个字符串数组；
- memory_refs: 固定输出 []，因为测试集没有公开 memory bank。

画像标签只能使用以下枚举；没有充分证据时宁可输出空数组：
personality_traits: {personality}
interests: {interests}
style: {styles}

User profile rules:
- personality_traits: inferred stable personality traits of the user. Leave [] if insufficient information.
- interests: user hobbies, topics they like. Leave [] if insufficient information.
- style: communication style of the user. Leave [] if insufficient information.
Do NOT invent facts. Only extract information present in conversation. If nothing known, set each to empty list [].
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地模型单轮推理并计时")
    parser.add_argument("test_file", type=Path, help="具体测试 JSONL 文件")
    parser.add_argument("result_dir", type=Path, help="结果输出目录")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: JSON 解析失败: {exc.msg}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("history"), list):
                raise ValueError(f"{path}:{number}: 必须包含字符串 id 和数组 history")
            rows.append(row)
    if not rows:
        raise ValueError(f"测试文件为空: {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("测试文件包含重复 id")
    return rows


def validate_test_file(test_file: Path) -> Path:
    if not test_file.is_file():
        raise ValueError(f"测试文件不存在或不是文件: {test_file}")
    if test_file.suffix.lower() != ".jsonl":
        raise ValueError(f"测试文件必须使用 .jsonl 扩展名: {test_file}")
    return test_file


def prepare_result_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"结果路径已存在但不是目录: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    environment_overrides = {
        "EVALUATION_BACKEND": "backend",
        "EVALUATION_BASE_URL": "base_url",
        "EVALUATION_API_MODEL": "api_model",
        "EVALUATION_API_KEY": "api_key",
    }
    for environment_name, config_name in environment_overrides.items():
        if environment_name in os.environ and (
            config_name == "api_key" or os.environ[environment_name]
        ):
            config[config_name] = os.environ[environment_name]
    if os.environ.get("EVALUATION_MODEL_PATH"):
        config["model_path"] = os.environ["EVALUATION_MODEL_PATH"]
    if os.environ.get("EVALUATION_HARDWARE_LABEL"):
        config["hardware_label"] = os.environ["EVALUATION_HARDWARE_LABEL"]
    config.setdefault("backend", "transformers")
    required = {"backend", "generation", "performance_rounds", "hardware_label"}
    if not required <= set(config):
        raise ValueError(f"推理配置缺少字段: {sorted(required - set(config))}")
    if config["backend"] not in {"transformers", "openai_compatible"}:
        raise ValueError("backend 必须为 transformers 或 openai_compatible")
    if config["backend"] == "transformers" and not config.get("model_path"):
        raise ValueError("transformers 模式必须配置 model_path")
    if config["backend"] == "openai_compatible":
        missing = [name for name in ("base_url", "api_model") if not config.get(name)]
        if missing:
            raise ValueError(f"openai_compatible 模式缺少字段: {missing}")
        api_endpoint(config["base_url"])
        timeout = config.get("request_timeout_seconds", 600)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("request_timeout_seconds 必须为正数")
        if not isinstance(config.get("api_options", {}), dict):
            raise ValueError("api_options 必须为 JSON object")
    if not isinstance(config["performance_rounds"], int) or config["performance_rounds"] <= 0:
        raise ValueError("performance_rounds 必须为正整数")
    max_user_tokens = config.get("max_user_tokens")
    if max_user_tokens is not None and (
        not isinstance(max_user_tokens, int) or isinstance(max_user_tokens, bool) or max_user_tokens <= 0
    ):
        raise ValueError("max_user_tokens 必须为正整数或省略")
    bias = config.get("emotion_bias", 0)
    if not isinstance(bias, (int, float)) or isinstance(bias, bool) or bias < 0:
        raise ValueError("emotion_bias 必须为非负数或省略")
    if bias > 0 and config["backend"] != "openai_compatible":
        raise ValueError("emotion_bias 需要 openai_compatible 后端 (logprobs)")
    top = config.get("emotion_bias_top_logprobs", 5)
    if not isinstance(top, int) or isinstance(top, bool) or not 1 <= top <= 20:
        raise ValueError("emotion_bias_top_logprobs 必须为 1 到 20 之间的整数")
    classifier_dir = config.get("profile_classifier")
    if classifier_dir is not None and (not isinstance(classifier_dir, str) or not classifier_dir):
        raise ValueError("profile_classifier 必须为目录路径字符串或省略")
    return config


def model_path(config: dict[str, Any]) -> Path:
    path = Path(config["model_path"])
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"本地模型目录不存在: {resolved}")
    return resolved.resolve()


def api_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str):
        raise ValueError("base_url 必须为字符串")
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"base_url 必须是有效的 HTTP(S) URL: {base_url}")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def api_generation_options(generation: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "temperature", "top_p", "stop", "seed", "frequency_penalty",
        "presence_penalty", "min_p", "top_k",
    }
    options = {name: value for name, value in generation.items() if name in allowed}
    if "max_new_tokens" in generation:
        options["max_tokens"] = generation["max_new_tokens"]
    elif "max_tokens" in generation:
        options["max_tokens"] = generation["max_tokens"]
    if generation.get("do_sample") is False:
        options["temperature"] = 0
    return options


def response_content(payload: Any) -> str:
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("API 响应缺少 choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("API 响应的 message.content 必须为字符串或文本片段数组")


def request_chat_completion(
    config: dict[str, Any],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> str:
    return response_content(request_chat_completion_raw(config, messages, generation))


def request_chat_completion_raw(
    config: dict[str, Any],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
    extra_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST one chat completion and return the parsed JSON response."""
    reserved = {"model", "messages"}
    api_options = dict(config.get("api_options", {}))
    if reserved & set(api_options):
        raise ValueError(f"api_options 不得覆盖字段: {sorted(reserved & set(api_options))}")
    payload = {
        "model": config["api_model"],
        "messages": messages,
        **api_generation_options(generation),
        **api_options,
        **(extra_options or {}),
    }
    headers = {"Content-Type": "application/json"}
    api_key = config.get("api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        api_endpoint(config["base_url"]),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=float(config.get("request_timeout_seconds", 600)),
        ) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API 请求失败: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接本地 Base URL: {exc.reason}") from exc
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("API 响应不是有效 JSON") from exc


def conversation_text(sample: dict[str, Any]) -> str:
    rendered = []
    for turn in sample["history"]:
        if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"} or not isinstance(turn.get("content"), str):
            raise ValueError(f"样本 {sample['id']} 的 history 格式无效")
        rendered.append(f"{turn['role']}: {turn['content']}")
    return "样本 ID: " + sample["id"] + "\n对话历史:\n" + "\n".join(rendered)


def fit_history(sample: dict[str, Any], count_tokens, max_tokens: int) -> tuple[dict[str, Any], int]:
    """Drop the oldest turns until conversation_text() fits in max_tokens (Z1).

    Same rule as train/build_sft.py fit_history(), so inference sees the same input
    shape as training: never start on an assistant turn, always keep the last turn.
    Returns (new sample, number of turns dropped). The input sample is not modified.
    """
    history = sample["history"]
    dropped = 0
    while True:
        candidate = {**sample, "history": history}
        if len(history) == 1 or count_tokens(conversation_text(candidate)) <= max_tokens:
            return candidate, dropped
        cut = 1
        while cut < len(history) - 1 and history[cut]["role"] == "assistant":
            cut += 1  # never start the window on an assistant turn
        history = history[cut:]
        dropped += cut


def history_token_counter(config: dict[str, Any], tokenizer: Any = None):
    """Token counter for fit_history(): len(encode(text, add_special_tokens=False)),
    exactly like TokenCounter.count() in train/build_sft.py. Loaded once at start-up."""
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("max_user_tokens 需要 transformers 来加载分词器") from exc
        source = os.environ.get("VLLM_MODEL_DIR") or config.get("model_path")
        if not source:
            raise ValueError("max_user_tokens 需要模型目录: 设置 VLLM_MODEL_DIR 或配置 model_path")
        path = Path(source)
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"分词器目录不存在: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def extract_object(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidate = fenced.group(1) if fenced else text
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            value, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("模型输出中没有可解析的 JSON object")


def recover_response_text(text: str) -> str:
    """Preserve useful model text when its surrounding JSON is malformed."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    match = re.search(
        r'"response_text"\s*:\s*"((?:\\.|[^"\\])*)',
        cleaned,
        flags=re.S,
    )
    if match:
        try:
            recovered = json.loads(f'"{match.group(1)}"')
            if isinstance(recovered, str):
                return recovered.strip()
        except json.JSONDecodeError:
            pass
    return cleaned


# Tags the base model predicts on almost every sample but gold rarely uses
# (V0 on val n50: emotionally_stable 47 vs 2, study_exam 36 vs 0, colloquial 38 vs 3).
# Dropping them raised profile micro-F1 0.275 -> 0.360 offline (EXP-013).
OVERPREDICTED_TAGS = {
    "personality_traits": {"emotionally_stable"},
    "interests": {"study_exam"},
    "style": {"colloquial"},
}


# The model sometimes invents an emotion outside the 16 labels ("frustration",
# "annoyance", "hopefulness" on val n50b: 3/50). A blank label always scores 0,
# so map common inventions to the closest official label, else fall back to
# "anxiety" (the most frequent class, ~1/3 of val).
EMOTION_ALIASES = {
    "frustration": "anger", "frustrated": "anger", "annoyance": "anger", "annoyed": "anger",
    "irritation": "anger", "irritated": "anger", "rage": "anger", "resentment": "anger",
    "worry": "anxiety", "worried": "anxiety", "nervous": "anxiety", "nervousness": "anxiety",
    "stress": "anxiety", "stressed": "anxiety", "anxious": "anxiety", "unease": "anxiety",
    "scared": "fear", "afraid": "fear", "terror": "fear", "panic": "fear",
    "sad": "sadness", "grief": "sadness", "disappointment": "sadness", "disappointed": "sadness",
    "depression": "sadness", "sorrow": "sadness", "hurt": "sadness",
    "lonely": "loneliness", "isolation": "loneliness",
    "happy": "joy", "happiness": "joy", "excitement": "joy", "excited": "joy",
    "thankful": "gratitude", "grateful": "gratitude", "thanks": "gratitude",
    "calm": "relaxed", "content": "relaxed", "contentment": "relaxed", "hopefulness": "relaxed",
    "hope": "relaxed", "hopeful": "relaxed", "relief": "relaxed", "peaceful": "relaxed",
    "embarrassment": "shame", "embarrassed": "shame", "guilt": "shame", "guilty": "shame",
    "helpless": "helplessness", "powerless": "helplessness", "hopelessness": "helplessness",
    "confusion": "mixed", "confused": "mixed", "ambivalence": "mixed",
    "proud": "pride", "surprised": "surprise", "shock": "surprise", "disgusted": "disgust",
    "concern": "care", "caring": "care", "empathy": "care",
}
EMOTION_FALLBACK = "anxiety"


def clean_emotion_label(emotion: Any) -> str:
    if not isinstance(emotion, str):
        return EMOTION_FALLBACK
    label = emotion.strip().lower()
    if label in EMOTIONS:
        return label
    return EMOTION_ALIASES.get(label, EMOTION_FALLBACK)


# EXP-022/023: anxiety-prior correction. The model says "anxiety" more often than
# gold does. With "emotion_bias": b > 0 in the config, the request also asks for the
# top logprobs, and when the parsed label is anxiety it is switched to the best other
# label if logprob(other) > logprob(anxiety) - b. Same functions and rule as
# scripts/emotion_logprobs.py + scripts/tune_emotion_bias.py (b is tuned there).
LABEL_KEY = re.compile(r'"emotion_label"\s*:\s*"')
# First token of each label right after ' "' (Qwen3 tokenizer). Exact match only:
# "an" is anxiety, "anger" is its own token.
LABEL_FIRST_TOKEN = {
    "joy": "joy", "gr": "gratitude", "rel": "relaxed", "care": "care", "pr": "pride",
    "neutral": "neutral", "sur": "surprise", "mixed": "mixed", "sad": "sadness",
    "lon": "loneliness", "an": "anxiety", "anger": "anger", "f": "fear", "dis": "disgust",
    "sh": "shame", "help": "helplessness",
}


def label_position(tokens: list[dict[str, Any]], content: str) -> int | None:
    """Index of the token where the emotion label value starts, or None."""
    match = LABEL_KEY.search(content)
    if not match:
        return None
    start = match.end()
    offset = 0
    for index, token in enumerate(tokens):
        if offset == start:
            return index
        offset += len(token["token"])
        if offset > start:  # label start falls inside a token: not a clean boundary
            return index
    return None


def label_alternatives(tokens: list[dict[str, Any]], content: str) -> list[dict[str, Any]] | None:
    """Top alternatives ({"token", "logprob"}) at the label's first token, or None."""
    position = label_position(tokens, content)
    if position is None:
        return None
    token = tokens[position]
    alternatives = [{"token": a["token"], "logprob": a["logprob"]} for a in token.get("top_logprobs") or []]
    if not any(a["token"] == token["token"] for a in alternatives):
        alternatives.append({"token": token["token"], "logprob": token["logprob"]})
    return alternatives


def label_scores(alternatives: list[dict[str, Any]] | None) -> dict[str, float]:
    """{label: logprob}; several tokens for one label are summed in probability space."""
    probs: dict[str, float] = {}
    for alternative in alternatives or []:
        label = LABEL_FIRST_TOKEN.get(alternative["token"].strip())
        if label is not None:
            probs[label] = probs.get(label, 0.0) + math.exp(alternative["logprob"])
    return {label: math.log(p) for label, p in probs.items()}


def apply_emotion_bias(label: str, scores: dict[str, float], bias: float) -> str:
    if bias <= 0 or label != "anxiety" or "anxiety" not in scores:
        return label
    others = {name: value for name, value in scores.items() if name != "anxiety"}
    if not others:
        return label
    best = max(others, key=others.get)
    return best if others[best] > scores["anxiety"] - bias else label


class ProfileClassifier:
    """EXP-024/025: CPU profile-tag classifier blended with the LLM's own tags.

    Char 1-3-gram TF-IDF over the user's messages (+ number of user turns), one logistic
    regression per tag, trained offline on train_public only (experiments/exp024-cpu-classifier).
    This is a numpy re-implementation of sklearn's TfidfVectorizer(analyzer="char",
    sublinear_tf=True) + LogisticRegression.predict_proba, checked equal in
    tests/test_profile_blend.py. Per field, a tag is kept if
        a * [the LLM output has it] + (1 - a) * p_classifier >= t
    with (a, t) from the model file ("blend").
    """

    WHITESPACE = re.compile(r"\s\s+")

    def __init__(self, directory: Path) -> None:
        import numpy  # only needed when the classifier is enabled

        self.np = numpy
        meta = json.loads((directory / "profile_clf.json").read_text(encoding="utf-8"))
        blocks = []
        for index, name in enumerate(meta["weight_files"]):  # weights split in files < 30 MB
            with numpy.load(directory / name) as arrays:
                if index == 0:
                    self.idf = arrays["idf"]
                    self.bias = arrays["bias"]
                blocks.append(arrays["coef"])
        self.coef = numpy.vstack(blocks)
        self.version = meta["version"]
        self.index = {gram: i for i, gram in enumerate(meta["vocab"])}
        self.tags = [tuple(tag) for tag in meta["tags"]]
        self.min_n, self.max_n = meta["ngram_range"]
        self.max_chars = meta["max_chars"]
        self.n_user_cap = meta["n_user_cap"]
        self.blend = {field: (float(a), float(t)) for field, (a, t) in meta["blend"].items()}
        if set(self.blend) != set(PROFILE_LABELS) or self.coef.shape != (len(self.tags), len(self.index) + 1):
            raise ValueError("profile_classifier 文件不一致")

    def probabilities(self, sample: dict[str, Any]) -> Any:
        users = [turn["content"] for turn in sample["history"] if turn.get("role") == "user"]
        text = self.WHITESPACE.sub(" ", "\n".join(users)[-self.max_chars:].lower())
        counts: dict[int, int] = {}
        for n in range(self.min_n, self.max_n + 1):
            for start in range(len(text) - n + 1):
                column = self.index.get(text[start:start + n])
                if column is not None:
                    counts[column] = counts.get(column, 0) + 1
        np = self.np
        columns = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        values = (np.log(np.fromiter(counts.values(), dtype=np.float64, count=len(counts))) + 1.0) * self.idf[columns]
        norm = np.sqrt((values * values).sum())
        if norm > 0:
            values = values / norm
        logits = self.coef[:, columns].astype(np.float64) @ values + self.bias
        logits += self.coef[:, -1].astype(np.float64) * (min(len(users), self.n_user_cap) / self.n_user_cap)
        return 1.0 / (1.0 + np.exp(-logits))

    def apply(self, profile: dict[str, list[str]], sample: dict[str, Any]) -> dict[str, list[str]]:
        probs = self.probabilities(sample)
        blended: dict[str, list[str]] = {}
        for field in PROFILE_LABELS:
            a, t = self.blend[field]
            llm_tags = profile.get(field) or []
            keep = {
                tag for j, (tag_field, tag) in enumerate(self.tags)
                if tag_field == field and a * (tag in llm_tags) + (1 - a) * probs[j] >= t
            }
            blended[field] = [tag for tag in llm_tags if tag in keep] + sorted(keep - set(llm_tags))
        return blended


def response_logprob_tokens(payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return (payload["choices"][0].get("logprobs") or {}).get("content") or []
    except (KeyError, IndexError, TypeError, AttributeError):
        return []


def normalize_prediction(sample_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    response = raw.get("response_text")
    emotion = raw.get("emotion_label")
    profile = raw.get("user_profile")
    memories = raw.get("memory_refs")
    clean_response = response.strip() if isinstance(response, str) else ""
    clean_emotion = clean_emotion_label(emotion)
    profile = profile if isinstance(profile, dict) else {}
    clean_profile: dict[str, list[str]] = {}
    for name, allowed in PROFILE_LABELS.items():
        values = profile.get(name)
        valid = (
            isinstance(values, list)
            and all(isinstance(value, str) for value in values)
            and len(values) == len(set(values))
            and set(values) <= allowed
        )
        clean_profile[name] = values if valid else []
        clean_profile[name] = [v for v in clean_profile[name] if v not in OVERPREDICTED_TAGS.get(name, ())]
    valid_memories = (
        isinstance(memories, list)
        and all(isinstance(value, str) and MEMORY_RE.fullmatch(value) for value in memories)
        and len(memories) == len(set(memories))
    )
    return {
        "id": sample_id,
        "response_text": clean_response,
        "emotion_label": clean_emotion,
        "user_profile": clean_profile,
        "memory_refs": memories if valid_memories else [],
    }


def parse_prediction(sample_id: str, decoded: str) -> tuple[dict[str, Any], bool]:
    try:
        raw_prediction = extract_object(decoded)
        parse_failed = False
    except ValueError:
        raw_prediction = {
            "response_text": recover_response_text(decoded),
            "emotion_label": "",
            "user_profile": {},
            "memory_refs": [],
        }
        parse_failed = True
    return normalize_prediction(sample_id, raw_prediction), parse_failed


def synchronize(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    test_file = validate_test_file(args.test_file.resolve())
    result_dir = args.result_dir.resolve()
    prepare_result_dir(result_dir)
    config = load_config()
    backend = config["backend"]
    backend_report: dict[str, Any]

    if backend == "transformers":
        local_model = model_path(config)
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
            )
        except ImportError as exc:
            raise RuntimeError("请先安装 participant/requirements.txt") from exc

        trust_remote_code = bool(config.get("trust_remote_code", False))
        model_config = AutoConfig.from_pretrained(
            local_model,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        model_loader = (
            AutoModelForImageTextToText
            if model_config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
            else AutoModelForCausalLM
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model, trust_remote_code=trust_remote_code, local_files_only=True)
        model = model_loader.from_pretrained(
            local_model,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            device_map=config.get("device_map", "auto"),
            torch_dtype=config.get("torch_dtype", "auto"),
        )
        model.eval()
        backend_report = {
            "version": "transformers_inference_v2",
            "timing_scope": "tokenize+generate+decode+parse+validate; model loading excluded",
            "model_path_name": local_model.name,
            "model_type": model_config.model_type,
            "model_loader": model_loader.__name__,
        }
    else:
        backend_report = {
            "version": "openai_compatible_inference_v2",
            "timing_scope": "HTTP request+server inference+response decode+parse+validate; server startup excluded",
            "api_model": config["api_model"],
            "base_url": config["base_url"],
        }

    # Z1: optional history truncation (config key max_user_tokens; absent = unchanged behaviour)
    max_user_tokens = config.get("max_user_tokens")
    count_tokens = None
    truncated_samples = dropped_turns = 0
    if max_user_tokens:
        count_tokens = history_token_counter(config, tokenizer if backend == "transformers" else None)

    samples = read_jsonl(test_file)
    submission_path = result_dir / "submission.jsonl"
    submission_path.write_text("", encoding="utf-8")
    latencies: list[float] = []
    predictions: list[dict[str, Any]] = []
    parse_failure_ids: list[str] = []
    generation = dict(config["generation"])

    system_prompt = SYSTEM_PROMPT.format(
        emotions=", ".join(sorted(EMOTIONS)),
        personality=", ".join(sorted(PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(PROFILE_LABELS["style"])),
    )
    emotion_bias = float(config.get("emotion_bias", 0))
    logprob_options = (
        {"logprobs": True, "top_logprobs": config.get("emotion_bias_top_logprobs", 5)}
        if emotion_bias > 0 else None
    )
    bias_switched = bias_no_scores = 0
    profile_classifier = None
    if config.get("profile_classifier"):
        classifier_path = Path(config["profile_classifier"])
        profile_classifier = ProfileClassifier(classifier_path if classifier_path.is_absolute() else ROOT / classifier_path)
    profile_added = profile_removed = 0

    for index, sample in enumerate(samples, 1):
        truncation_ms = 0.0
        model_input = sample
        if count_tokens is not None:
            truncation_started = time.perf_counter()
            model_input, dropped = fit_history(sample, count_tokens, max_user_tokens)
            truncation_ms = (time.perf_counter() - truncation_started) * 1000
            if dropped:
                truncated_samples += 1
                dropped_turns += dropped
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": conversation_text(model_input)}]
        if backend == "transformers":
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            synchronize(torch)
            started = time.perf_counter()
            inputs = tokenizer(prompt, return_tensors="pt")
            input_device = next(model.parameters()).device
            inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation)
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
        else:
            started = time.perf_counter()
            if logprob_options is None:
                decoded = request_chat_completion(config, messages, generation)
            else:
                response = request_chat_completion_raw(config, messages, generation, logprob_options)
                decoded = response_content(response)
        prediction, parse_failed = parse_prediction(sample["id"], decoded)
        if parse_failed:
            parse_failure_ids.append(sample["id"])
        if logprob_options is not None:
            scores = label_scores(label_alternatives(response_logprob_tokens(response), decoded))
            bias_no_scores += not scores
            label = apply_emotion_bias(prediction["emotion_label"], scores, emotion_bias)
            bias_switched += label != prediction["emotion_label"]
            prediction["emotion_label"] = label
        if profile_classifier is not None:
            blended = profile_classifier.apply(prediction["user_profile"], sample)
            for field in PROFILE_LABELS:
                before, after = set(prediction["user_profile"][field]), set(blended[field])
                profile_added += len(after - before)
                profile_removed += len(before - after)
            prediction["user_profile"] = blended
        if backend == "transformers":
            synchronize(torch)
        elapsed_ms = (time.perf_counter() - started) * 1000 + truncation_ms
        if index <= config["performance_rounds"]:
            latencies.append(elapsed_ms)
        predictions.append(prediction)
        with submission_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"[{index}/{len(samples)}] {sample['id']}: {elapsed_ms:.3f} ms", flush=True)

    report = {
        **backend_report,
        "backend": backend,
        "complete": len(latencies) == config["performance_rounds"],
        "required_rounds": config["performance_rounds"],
        "rounds": len(latencies),
        "average_latency_ms": round(statistics.fmean(latencies), 6),
        "median_latency_ms": round(statistics.median(latencies), 6),
        "p95_latency_ms": round(percentile95(latencies), 6),
        "min_latency_ms": round(min(latencies), 6),
        "max_latency_ms": round(max(latencies), 6),
        "latencies_ms": [round(value, 6) for value in latencies],
        "hardware_label": config["hardware_label"],
        "test_file_sha256": sha256(test_file),
        "submission_sha256": sha256(submission_path),
        "samples": len(predictions),
        "parse_failure_count": len(parse_failure_ids),
        "parse_failure_ids": parse_failure_ids,
    }
    if count_tokens is not None:
        report["truncation"] = {
            "max_user_tokens": max_user_tokens,
            "truncated_samples": truncated_samples,
            "dropped_turns": dropped_turns,
        }
    (result_dir / "performance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if logprob_options is not None:
        print(f"emotion bias {emotion_bias:g} (top_logprobs {logprob_options['top_logprobs']}): "
              f"switched {bias_switched}/{len(predictions)}, no label scores {bias_no_scores}", flush=True)
    if profile_classifier is not None:
        print(f"profile classifier {profile_classifier.version}: tags added {profile_added}, "
              f"removed {profile_removed} over {len(predictions)} samples", flush=True)
    print(f"PASS: {submission_path}")
    print(f"PASS: {result_dir / 'performance_report.json'}")


if __name__ == "__main__":
    main()
