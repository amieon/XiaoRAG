"""完全信任显式 DOCX 原生结构的 heading/body 判定，不加载模型。"""
from __future__ import annotations

from typing import Any, Sequence

from .heading_classifier import BODY_LABEL, HEADING_LABEL, HeadingExample


DEFAULT_TRUSTED_SOURCES = frozenset({
    "docx_style", "docx_outline", "docx_style_outline",
})


class NativeHeadingClassifier:
    """原生证据存在即为标题，否则为正文；置信度字段不伪造模型概率。"""

    def __init__(self, trusted_sources: set[str] | frozenset[str] = DEFAULT_TRUSTED_SOURCES):
        self.trusted_sources = frozenset(trusted_sources)

    def predict(
        self, examples: Sequence[HeadingExample], *, batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results = []
        for example in examples:
            evidence = [
                item for item in example.native_structures
                if item.get("source") in self.trusted_sources
                and item.get("role") in {"heading", "document_title"}
            ]
            levels = [
                item.get("heading_level") for item in evidence
                if type(item.get("heading_level")) is int
            ]
            results.append({
                "block_id": example.block_id,
                "page_id": example.page_id,
                "document_id": example.document_id,
                "reading_order": example.reading_order,
                "label": HEADING_LABEL if evidence else BODY_LABEL,
                "confidence": None,
                "scores": None,
                "heading_level": levels[0] if len(evidence) == 1 and len(levels) == 1 else None,
                "native_heading_count": len(evidence),
                "native_evidence": evidence,
                "decision_source": "docx_native_trust" if evidence else "native_heading_absent",
                "raw_label": example.raw_label,
                "source_page_result": example.source_page_result,
                "task": "binary_heading_body",
                "mode": "native",
            })
        return results
