"""v3 judge prompts, extraction, transport and response caching.

This module owns no CLI or benchmark aggregation. Client construction is explicit.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import socket
import time
from collections import Counter
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import FAIL_MSG, OPTION_LABELS, RULES_VERSION
from .rules import NUMBER_RE, WORD_RE, decimal_number, ground_truth_labels, letter_sequence, word_number


_LETTER_RE = re.compile(rf"(?:\*\*)?\b([{OPTION_LABELS}])\s*\.\s")


_STRICT_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _build_option_str(choices: Dict[str, str]) -> str:
    """构建选项文本，与 mcq_utils.build_option_str 一致。"""
    s = "There are several options: \n"
    for c, content in choices.items():
        s += f"{c}. {content}\n"
    return s


def _build_mca_judge_prompt(question: str, options: str, prediction: str,
                            multi_letter: bool = False) -> str:
    """MCA 裁判 prompt（英文，参考 VLMEvalKit multiple_choice.py build_prompt）。

    Args:
        question: 题干
        options: 选项文本（"There are several options: \\nA. ...\\nB. ..."）
        prediction: 模型回答
        multi_letter: 是否多字母序列题（route_plan），要求输出所有字母按顺序
    """
    if multi_letter:
        tmpl = (
            "You are an AI assistant who will help me to match "
            "an answer with several options of a multi-step question. "
            "You are provided with a question, several options, and an answer. "
            "The answer may contain one or more steps, each corresponding to an option letter. "
            "Your task is to extract the sequence of option letters that the answer indicates, "
            "in the same order as they appear in the answer. "
            "If the meaning of all options are significantly different from the answer, output Z. "
            "Your should output only the uppercase letters from A, B, C, D, E, F "
            "(if they are valid options), retaining order and repeats "
            "(e.g. BF, A, FF) with no separator, "
            "or Z if none match. \n"
            "Example 1: \n"
            "Question: Fill in the steps. 1. [fill] 2. go straight.\n"
            "Options: A. turn left B. turn right C. turn 180 degrees D. go upstairs\n"
            "Answer: First turn right, then go upstairs.\nYour output: BD\n"
            "Example 2: \n"
            "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
        )
    else:
        tmpl = (
            "You are an AI assistant who will help me to match "
            "an answer with several options of a single-choice question. "
            "You are provided with a question, several options, and an answer, "
            "and you need to find which option is most similar to the answer. "
            "If the meaning of all options are significantly different from the answer, output Z. "
            "Your should output a single uppercase character in A, B, C, D, E, F (if they are valid options), and Z. \n"
            "Example 1: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
            "Answer: a cute teddy bear\nYour output: A\n"
            "Example 2: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
            "Answer: Spider\nYour output: Z\n"
            "Example 3: \n"
            "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
        )
    return tmpl.format(question, options, prediction)


def _build_na_extract_prompt(question: str, prediction: str) -> str:
    """构造数值题提取 prompt。

    这里的裁判不是根据题目重新求解，也不会看到标准答案；它仅从模型回答中
    提取模型最终给出的一个数值。因此不会改变 MRA 的数值评分口径。
    """
    return (
        "You are a strict answer-format extractor for a numerical question. "
        "Do NOT solve the question and do NOT estimate a new value. "
        "Read the model answer and extract the single final numerical answer it gives. "
        "Remove surrounding words and units (for example, cm, meters, m²). "
        "If there is no unambiguous final numerical answer, output Z. "
        "Output only one decimal number with optional sign, no unit and no explanation.\n"
        f"Question: {question}\n"
        f"Model answer: {prediction}\n"
        "Your output:"
    )


def _parse_finite_number(text: Any) -> Optional[float]:
    """仅接受完整、有限的数值字符串，避免 ``float`` 宽松解析。"""
    normalized = str(text).strip() if text is not None else ""
    if not _STRICT_NUMBER_RE.fullmatch(normalized):
        return None
    try:
        value = float(normalized)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _numbers_in_text(text: Any) -> List[float]:
    """Validate provenance, including number words and unit-attached numbers."""
    raw = re.sub(r"\bm\s*(?:²|\^\s*2|2\b)", " square meters ", str(text or ""), flags=re.I)
    raw = re.sub(r"(?<=\d)(?=cm\b|mm\b|m\b)", " ", raw, flags=re.I)
    values = []
    for token in NUMBER_RE.findall(raw):
        number = decimal_number(token)
        if number is not None:
            value = float(number)
            if math.isfinite(value):
                values.append(value)
    for match in WORD_RE.finditer(raw):
        number = word_number(match.group())
        if number is not None:
            values.append(float(number))
    return values


def _is_source_number(value: float, source_numbers: List[float]) -> bool:
    """验证裁判数值确实来自原回答，允许浮点序列化带来的极小误差。"""
    return any(
        abs(value - candidate) <= 1e-12 + 1e-9 * abs(candidate)
        for candidate in source_numbers
    )


def _extract_letter(text: str) -> Optional[str]:
    """从裁判返回中提取单个选项字母（取第一个合法字母）。"""
    if not text:
        return None
    m = _LETTER_RE.findall(text)
    if m:
        return m[0].upper()
    m = re.findall(rf"\b([{OPTION_LABELS}])\b", text)
    return m[0].upper() if m else None


def _extract_letter_sequence(text: str) -> List[str]:
    """Parse the saved judge sequence locally, retaining F, order and repeats."""
    sequence = letter_sequence(text, set(OPTION_LABELS))
    if sequence is not None:
        return sequence
    return [letter.upper() for letter in _LETTER_RE.findall(str(text or ""))]


def _extract_gt_letters(gt: str, multi_letter: bool) -> List[str]:
    """Local ground-truth parser; F and repeated route actions are significant."""
    try:
        labels = ground_truth_labels(gt, multi_letter)
        return labels if all(label in OPTION_LABELS for label in labels) else []
    except ValueError:
        return []


class JudgeConfigurationError(ValueError):
    """Stop on invalid credentials/model/endpoint instead of failing 5K calls."""


@dataclass
class JudgeReply:
    text: str = ""
    failed_reason: str | None = None
    error: str = ""
    attempts: int = 0
    cache_hit: bool = False


class _NoJudgeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer credential to a redirected host.
        return None


class JudgeClient:
    """Small OpenAI-compatible text client; never imports platform/SDK modules.

    Construct only when a dedicated judge key was supplied. Raw keys and server
    error bodies are never written to reports/caches or exception messages.
    """

    def __init__(self, api_key, model="gpt-4o", base_url="https://api.openai.com/v1",
                 timeout=60., max_attempts=3, qps=2., max_tokens=512, cache_path=None):
        self._api_key = str(api_key or "").strip()
        if not self._api_key or not self._api_key.isascii() or any(c.isspace() for c in self._api_key):
            raise JudgeConfigurationError("Judge API key must be a nonempty ASCII token without whitespace")
        parts = urlsplit(str(base_url))
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None or parts.query or parts.fragment):
            raise JudgeConfigurationError("Judge base URL must be an HTTP(S) API base without credentials/query/fragment")
        if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise JudgeConfigurationError("Use HTTPS for a remote judge endpoint; HTTP is allowed only on loopback")
        if not str(model).strip():
            raise JudgeConfigurationError("Judge model must not be empty")
        if (not math.isfinite(float(timeout)) or timeout <= 0 or not math.isfinite(float(qps)) or qps <= 0
                or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 10
                or not isinstance(max_tokens, int) or max_tokens <= 0):
            raise JudgeConfigurationError("Judge timeout/QPS/token limit must be positive; attempts must be 1..10")
        self.model, self.base_url = str(model).strip(), str(base_url).rstrip("/")
        self.timeout, self.max_attempts, self.qps, self.max_tokens = timeout, max_attempts, qps, max_tokens
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.api_requests = self.cache_hits = self.judged_samples = 0
        self.usage = Counter()
        self._cache, self._next_request = {}, 0.
        self._opener = build_opener(_NoJudgeRedirect())
        if self.cache_path and self.cache_path.exists():
            for number, line in enumerate(self.cache_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                try:
                    entry = json.loads(line)
                    if (not isinstance(entry, dict) or entry.get("format") != "spatial4d-judge-cache-v1"
                            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("key", "")))
                            or not isinstance(entry.get("response"), str)):
                        raise ValueError
                except (ValueError, TypeError):
                    raise JudgeConfigurationError(f"Invalid judge cache entry at line {number}") from None
                self._cache[entry["key"]] = self._redact(entry["response"])

    def _redact(self, value):
        return str(value).replace(self._api_key, "[REDACTED]")

    def _cache_key(self, prompt):
        identity = [RULES_VERSION, self.base_url, self.model, self.max_tokens, self.max_attempts, prompt]
        return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()

    def request(self, prompt):
        self.judged_samples += 1
        key = self._cache_key(prompt)
        if key in self._cache:
            self.cache_hits += 1
            return JudgeReply(self._cache[key], cache_hit=True)
        last = JudgeReply()
        for index in range(self.max_attempts):
            delay = self._next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_request = time.monotonic() + 1 / self.qps
            payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": min(index * .5, 2.), "max_tokens": self.max_tokens, "store": False}
            request = Request(self.base_url + "/chat/completions",
                              data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              headers={"Authorization": "Bearer " + self._api_key,
                                       "Content-Type": "application/json"}, method="POST")
            self.api_requests += 1
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices") if isinstance(data, dict) else None
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise ValueError("Missing chat choices")
                message = choices[0].get("message")
                if not isinstance(message, dict):
                    raise ValueError("Missing chat message")
                content = message.get("content")
                if not isinstance(content, (str, type(None))):
                    last = JudgeReply(failed_reason="judge_invalid_response", error="Non-text judge content", attempts=index + 1)
                    continue
                text = self._redact(content or "").strip()
                for name, value in (data.get("usage") or {}).items():
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        self.usage[name] += value
                if text and FAIL_MSG not in text:
                    # Cache completed responses (including Z/invalid answers),
                    # but not transport failures or empty completions.
                    if self.cache_path:
                        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                        entry = {"format": "spatial4d-judge-cache-v1", "key": key, "response": text}
                        with self.cache_path.open("ab+") as stream:
                            stream.seek(0, os.SEEK_END)
                            if stream.tell():
                                stream.seek(-1, os.SEEK_END)
                                if stream.read(1) != b"\n":
                                    stream.write(b"\n")
                            stream.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
                    self._cache[key] = text
                    return JudgeReply(text, attempts=index + 1)
                last = JudgeReply(failed_reason="judge_empty_response", error="Empty/failed judge completion", attempts=index + 1)
            except HTTPError as exc:
                code = exc.code
                exc.close()
                if code in {400, 401, 403, 404, 405, 422} or 300 <= code < 400:
                    raise JudgeConfigurationError(
                        f"Judge HTTP {code}; check API key, model and base URL. No rule fallback was used.") from None
                last = JudgeReply(failed_reason="judge_http_error", error=f"HTTP {code}", attempts=index + 1)
            except (URLError, TimeoutError, socket.timeout, ConnectionError, HTTPException):
                last = JudgeReply(failed_reason="judge_transport_error", error="Judge connection failed/timed out", attempts=index + 1)
            except (ValueError, UnicodeError, KeyError, TypeError, AttributeError):
                last = JudgeReply(failed_reason="judge_invalid_response", error="Malformed judge response", attempts=index + 1)
        return last

    def summary(self):
        return {"enabled": True, "model": self._redact(self.model), "base_url": self._redact(self.base_url),
                "judged_samples": self.judged_samples, "api_requests": self.api_requests,
                "cache_hits": self.cache_hits, "usage": dict(self.usage),
                "max_attempts": self.max_attempts, "timeout_seconds": self.timeout,
                "qps": self.qps, "max_tokens": self.max_tokens}
