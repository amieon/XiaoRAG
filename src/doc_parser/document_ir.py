"""Canonical document intermediate representation.

页面证据与文档级关系分开保存，知识单元构建只依赖这里的稳定字段，
不依赖任何 OCR 引擎的私有 JSON。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1"


class DocumentIRValidationError(ValueError):
    """Raised when an IR object violates the canonical schema."""


def _number_list(value: Any, *, name: str, length: int = 4) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise DocumentIRValidationError(f"{name} must contain exactly {length} numbers")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise DocumentIRValidationError(f"{name} must contain only numbers") from exc
    return result


def _optional_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DocumentIRValidationError("confidence must be a number") from exc
    if not 0 <= result <= 1:
        raise DocumentIRValidationError("confidence must be between 0 and 1")
    return result


@dataclass(slots=True)
class SourceRef:
    """Pointer from a canonical object back to an engine result."""

    engine: str
    engine_version: str | None = None
    raw_page: int | None = None
    raw_block_id: str | None = None
    raw_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.engine:
            raise DocumentIRValidationError("source.engine is required")


@dataclass(slots=True)
class Block:
    """A page-level content block in canonical coordinates."""

    block_id: str
    page_id: str
    block_type: str
    bbox: list[float]
    polygon: list[list[float]] | None = None
    text: str = ""
    reading_order: int | None = None
    confidence: float | None = None
    parent_id: str | None = None
    source: SourceRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.block_id or not self.page_id or not self.block_type:
            raise DocumentIRValidationError("block_id, page_id, and block_type are required")
        self.bbox = _number_list(self.bbox, name=f"{self.block_id}.bbox")
        if self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]:
            raise DocumentIRValidationError(f"{self.block_id}.bbox has negative dimensions")
        if self.polygon is not None:
            if not isinstance(self.polygon, list) or len(self.polygon) < 3:
                raise DocumentIRValidationError(f"{self.block_id}.polygon must contain at least 3 points")
            self.polygon = [
                _number_list(point, name=f"{self.block_id}.polygon point", length=2)
                for point in self.polygon
            ]
        self.confidence = _optional_confidence(self.confidence)
        if self.reading_order is not None and self.reading_order < 0:
            raise DocumentIRValidationError(f"{self.block_id}.reading_order must be non-negative")
        if self.source:
            self.source.validate()


@dataclass(slots=True)
class Page:
    """A normalized page containing page-local blocks."""

    page_id: str
    page_no: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)
    rotation: int = 0
    source: SourceRef | None = None
    image_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.page_id:
            raise DocumentIRValidationError("page_id is required")
        if self.page_no < 1:
            raise DocumentIRValidationError(f"{self.page_id}.page_no must start at 1")
        if self.width <= 0 or self.height <= 0:
            raise DocumentIRValidationError(f"{self.page_id} must have positive dimensions")
        if self.rotation % 90:
            raise DocumentIRValidationError(f"{self.page_id}.rotation must be a multiple of 90")
        if self.source:
            self.source.validate()
        seen: set[str] = set()
        for block in self.blocks:
            block.validate()
            if block.page_id != self.page_id:
                raise DocumentIRValidationError(f"{block.block_id} points to another page")
            if block.block_id in seen:
                raise DocumentIRValidationError(f"duplicate block id: {block.block_id}")
            seen.add(block.block_id)


@dataclass(slots=True)
class Relation:
    """A typed, auditable relation between IR objects."""

    relation_id: str
    relation_type: str
    source_id: str
    target_id: str
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.relation_id or not self.relation_type:
            raise DocumentIRValidationError("relation_id and relation_type are required")
        if not self.source_id or not self.target_id:
            raise DocumentIRValidationError(f"{self.relation_id} needs source_id and target_id")
        self.confidence = _optional_confidence(self.confidence)


@dataclass(slots=True)
class DocumentIR:
    """Canonical document representation shared by parser and memory layers."""

    document_id: str
    pages: list[Page]
    schema_version: str = SCHEMA_VERSION
    source_path: str | None = None
    content_hash: str | None = None
    mime_type: str | None = None
    title: str | None = None
    relations: list[Relation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.document_id:
            raise DocumentIRValidationError("document_id is required")
        if not self.schema_version:
            raise DocumentIRValidationError("schema_version is required")
        seen_pages: set[str] = set()
        seen_objects: set[str] = set()
        for page in self.pages:
            page.validate()
            if page.page_id in seen_pages:
                raise DocumentIRValidationError(f"duplicate page id: {page.page_id}")
            seen_pages.add(page.page_id)
            seen_objects.add(page.page_id)
            seen_objects.update(block.block_id for block in page.blocks)
        for relation in self.relations:
            relation.validate()
            if relation.source_id not in seen_objects or relation.target_id not in seen_objects:
                raise DocumentIRValidationError(
                    f"{relation.relation_id} references an object outside this document"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocumentIR":
        def source(data: Mapping[str, Any] | None) -> SourceRef | None:
            if data is None:
                return None
            return SourceRef(**dict(data))

        pages: list[Page] = []
        for page_data in value.get("pages", []):
            blocks = [
                Block(
                    **{
                        **dict(block_data),
                        "source": source(block_data.get("source")),
                    }
                )
                for block_data in page_data.get("blocks", [])
            ]
            pages.append(
                Page(
                    **{
                        **dict(page_data),
                        "blocks": blocks,
                        "source": source(page_data.get("source")),
                    }
                )
            )
        relations = [Relation(**dict(item)) for item in value.get("relations", [])]
        result = cls(
            document_id=value["document_id"],
            pages=pages,
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            source_path=value.get("source_path"),
            content_hash=value.get("content_hash"),
            mime_type=value.get("mime_type"),
            title=value.get("title"),
            relations=relations,
            metadata=dict(value.get("metadata", {})),
        )
        result.validate()
        return result


def iter_blocks(document: DocumentIR) -> Iterable[Block]:
    """Yield blocks in page and reading order."""

    for page in sorted(document.pages, key=lambda item: item.page_no):
        yield from sorted(
            page.blocks,
            key=lambda item: item.reading_order if item.reading_order is not None else 10**9,
        )
