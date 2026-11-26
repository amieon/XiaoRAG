"""无需微调的 OpenAI 兼容标题／正文判别器。

模型只补充缺失的原生标题证据；原生 DOCX 标题始终优先。API 密钥只从环境变量读取。
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Sequence
from urllib import error, request

from .heading_classifier import BODY_LABEL, HEADING_LABEL, HeadingExample
from .native_classifier import NativeHeadingClassifier


SYSTEM_PROMPT = """你是文档结构标注器。判断每个当前块是章节标题还是正文，并给出标题层级。
只依据输入证据，不改写文本，不补充输入中不存在的块。
规则：
1. 标题通常简短，概括后续内容；完整陈述句、列表项、表格和公式通常是正文。
2. 明确的章节编号可用于判断层级，例如“第一章”为1级，“1.2”为2级，“1.2.3”为3级。
3. 无编号标题需结合前后文判断层级；无法确定更深关系时选择1级，不要编造编号。
4. label只能是heading或body；heading_level仅heading需要，为1到6的整数，body必须为null。
5. 输入的每个块必须恰好输出一次，保持block_id不变。
只输出JSON对象：{"predictions":[{"block_id":"...","label":"heading","heading_level":1}]}。
"""


def _endpoint(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("zero-shot mode requires --base-url or HEADING_MODEL_BASE_URL")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def _content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Model response has no choices[0].message.content") from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") for item in content
            if isinstance(item, dict) and item.get("type") in {None, "text"}
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model response content is empty")
    return content.strip()


def _json_content(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().casefold() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc.msg}") from exc


def _validated_prediction(value: Any, expected_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Each model prediction must be an object")
    block_id = value.get("block_id")
    label = value.get("label")
    level = value.get("heading_level")
    if not isinstance(block_id, str) or block_id not in expected_ids:
        raise ValueError(f"Model returned an unknown block_id: {block_id!r}")
    if label not in {BODY_LABEL, HEADING_LABEL}:
        raise ValueError(f"Invalid heading label for {block_id}: {label!r}")
    if label == HEADING_LABEL:
        if type(level) is not int or not 1 <= level <= 6:
            raise ValueError(f"Heading {block_id} requires heading_level in 1..6")
    elif level is not None:
        raise ValueError(f"Body {block_id} must have heading_level=null")
    return {"block_id": block_id, "label": label, "heading_level": level}


class OpenAICompatibleHeadingClassifier:
    """批量零样本判别；transport 仅供无网络单元测试替换。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("zero-shot mode requires --model or HEADING_MODEL_NAME")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.url = _endpoint(base_url)
        self.model = model.strip()
        self.api_key = api_key if api_key is not None else os.environ.get("HEADING_MODEL_API_KEY")
        self.timeout = timeout
        self.transport = transport

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.transport is not None:
            return self.transport(payload)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            self.url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Heading model HTTP {exc.code}: {detail}") from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Heading model request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise ValueError("Heading model response must be a JSON object")
        return result

    def _predict_batch(self, examples: Sequence[HeadingExample]) -> list[dict[str, Any]]:
        items = [{
            "block_id": item.block_id,
            "text": item.text,
            "previous": (item.context.get("previous") or {}).get("text"),
            "next": (item.context.get("next") or {}).get("text"),
            "raw_label": item.raw_label,
            "reading_order": item.reading_order,
        } for item in examples]
        expected = [item.block_id for item in examples]
        expected_ids = set(expected)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"blocks": items}, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        response = self._post(payload)
        decoded = _json_content(_content(response))
        rows = decoded.get("predictions") if isinstance(decoded, dict) else decoded
        if not isinstance(rows, list):
            raise ValueError("Model JSON must contain a predictions array")
        validated = [_validated_prediction(row, expected_ids) for row in rows]
        ids = [row["block_id"] for row in validated]
        if len(ids) != len(set(ids)):
            raise ValueError("Model returned duplicate block_id values")
        if set(ids) != expected_ids or len(ids) != len(expected):
            missing = sorted(expected_ids - set(ids))
            raise ValueError(f"Model must return every input block exactly once; missing={missing[:3]}")
        by_id = {row["block_id"]: row for row in validated}
        return [by_id[block_id] for block_id in expected]

    def predict(
        self, examples: Sequence[HeadingExample], *, batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        native = NativeHeadingClassifier().predict(examples, batch_size=batch_size)
        unresolved = [index for index, row in enumerate(native) if not row["native_heading_count"]]
        results: list[dict[str, Any] | None] = list(native)
        for start in range(0, len(unresolved), batch_size):
            indices = unresolved[start:start + batch_size]
            predictions = self._predict_batch([examples[index] for index in indices])
            for index, prediction in zip(indices, predictions):
                item = examples[index]
                results[index] = {
                    "block_id": item.block_id,
                    "page_id": item.page_id,
                    "document_id": item.document_id,
                    "reading_order": item.reading_order,
                    **prediction,
                    "confidence": None,
                    "scores": None,
                    "native_heading_count": 0,
                    "native_evidence": [],
                    "decision_source": "zero_shot_llm",
                    "raw_label": item.raw_label,
                    "source_page_result": item.source_page_result,
                    "task": "binary_heading_body",
                    "mode": "zero_shot_llm",
                    "model": self.model,
                }
        return [row for row in results if row is not None]
