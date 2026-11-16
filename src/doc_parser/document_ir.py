"""Canonical document intermediate representation.

The IR deliberately keeps page-level evidence and document-level relationships
separate.  Parser adapters can normalize MinerU, PaddleOCR, or other engine
outputs into this module without making downstream code depend on an engine's
private JSON layout.
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


def text_from_mineru_block(block: Mapping[str, Any]) -> str:
    """Extract visible text from common MinerU line/span structures."""

    if isinstance(block.get("text"), str):
        return block["text"].strip()
    parts: list[str] = []
    for line in block.get("lines", []) or []:
        for span in line.get("spans", []) or []:
            content = span.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    if parts:
        return "".join(parts).strip()
    for key in ("caption", "table_caption", "img_caption"):
        value = block.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def _mineru_block_type(value: Any) -> str:
    raw = str(value or "unknown").lower()
    aliases = {
        "title": "section_heading",
        "doc_title": "document_title",
        "paragraph_title": "section_heading",
        "text": "paragraph",
        "abstract": "paragraph",
        "ref_text": "paragraph",
        "list": "list",
        "index": "list",
        "table": "table",
        "image": "figure",
        "figure": "figure",
        "chart": "figure",
        "seal": "figure",
        "equation": "formula",
        "interline_equation": "formula",
    }
    return aliases.get(raw, raw)


def _mineru_bbox(block: Mapping[str, Any]) -> list[float]:
    bbox = block.get("bbox")
    if bbox is not None:
        return _number_list(bbox, name="MinerU block bbox")
    boxes: list[list[float]] = []
    for line in block.get("lines", []) or []:
        if line.get("bbox") is not None:
            boxes.append(_number_list(line["bbox"], name="MinerU line bbox"))
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    ]


def from_mineru_middle_json(
    middle_json: Mapping[str, Any],
    *,
    document_id: str,
    source_path: str | None = None,
    engine_version: str | None = None,
) -> DocumentIR:
    """Normalize a MinerU ``*_middle.json`` object into :class:`DocumentIR`."""

    raw_pages = middle_json.get("pdf_info")
    if not isinstance(raw_pages, list):
        raise DocumentIRValidationError("MinerU middle JSON must contain a pdf_info list")
    pages: list[Page] = []
    for page_position, raw_page in enumerate(raw_pages):
        page_no = int(raw_page.get("page_idx", page_position)) + 1
        page_size = raw_page.get("page_size") or raw_page.get("page_size_points")
        if not isinstance(page_size, (list, tuple)) or len(page_size) < 2:
            raise DocumentIRValidationError(f"MinerU page {page_no} has no valid page_size")
        width, height = float(page_size[0]), float(page_size[1])
        page_id = f"{document_id}_p{page_no:04d}"
        source = SourceRef(
            engine="mineru",
            engine_version=engine_version,
            raw_page=page_no,
            raw_path=source_path,
        )
        blocks: list[Block] = []
        raw_blocks = raw_page.get("para_blocks") or []
        for block_position, raw_block in enumerate(raw_blocks):
            block_id = f"{page_id}_b{block_position:04d}"
            raw_confidence = raw_block.get("confidence", raw_block.get("score"))
            block_source = SourceRef(
                engine="mineru",
                engine_version=engine_version,
                raw_page=page_no,
                raw_block_id=str(raw_block.get("id", block_position)),
                raw_path=source_path,
            )
            blocks.append(
                Block(
                    block_id=block_id,
                    page_id=page_id,
                    block_type=_mineru_block_type(raw_block.get("type")),
                    bbox=_mineru_bbox(raw_block),
                    text=text_from_mineru_block(raw_block),
                    reading_order=block_position,
                    confidence=raw_confidence,
                    source=block_source,
                    metadata={
                        "raw_type": raw_block.get("type"),
                        "raw_keys": sorted(str(key) for key in raw_block.keys()),
                    },
                )
            )
        pages.append(
            Page(
                page_id=page_id,
                page_no=page_no,
                width=width,
                height=height,
                blocks=blocks,
                source=source,
            )
        )
    result = DocumentIR(
        document_id=document_id,
        pages=pages,
        source_path=source_path,
        metadata={"source_backend": middle_json.get("_backend", "unknown")},
    )
    result.validate()
    return result


def from_teleocr_middle_json(
    middle_json: Mapping[str, Any],
    *,
    document_id: str,
    source_path: str | None = None,
    engine_version: str | None = None,
) -> DocumentIR:
    """Normalize TeleOCR's MinerU-compatible ``*_middle.json`` output.

    TeleOCR uses the same ``pdf_info``/``para_blocks`` shape for its dumped
    middle result.  Keeping a named adapter makes the provenance explicit and
    leaves room for TeleOCR-specific fields later.
    """

    result = from_mineru_middle_json(
        middle_json,
        document_id=document_id,
        source_path=source_path,
        engine_version=engine_version,
    )
    for page in result.pages:
        if page.source:
            page.source.engine = "teleocr"
        for block in page.blocks:
            if block.source:
                block.source.engine = "teleocr"
    result.metadata["source_backend"] = middle_json.get("_backend", "teleocr")
    result.validate()
    return result


def from_paddleocr_json(
    payload: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    document_id: str,
    source_path: str | None = None,
    engine_version: str | None = None,
) -> DocumentIR:
    """Normalize PaddleOCR-VL/PP-Structure JSON output.

    ``save_to_json`` emits one object per page with ``width``, ``height`` and
    ``parsing_res_list``.  A list of those objects is accepted for callers
    which have already collected a multi-page result.
    """

    raw_pages = payload if isinstance(payload, list) else [payload]
    pages: list[Page] = []
    for position, raw_page in enumerate(raw_pages):
        raw_page_index = raw_page.get("page_index")
        page_no = int(raw_page_index) + 1 if raw_page_index is not None else position + 1
        blocks_data = raw_page.get("parsing_res_list") or raw_page.get("blocks") or []
        if not isinstance(blocks_data, list):
            raise DocumentIRValidationError("PaddleOCR parsing_res_list must be a list")
        width = float(raw_page.get("width", 0))
        height = float(raw_page.get("height", 0))
        if width <= 0 or height <= 0:
            # Some exported fixtures omit image dimensions. Derive a safe page
            # extent from block boxes while keeping the result valid.
            boxes = [
                _number_list(item.get("block_bbox", item.get("bbox")), name="PaddleOCR bbox")
                for item in blocks_data
                if item.get("block_bbox", item.get("bbox")) is not None
            ]
            width = max((box[2] for box in boxes), default=1.0)
            height = max((box[3] for box in boxes), default=1.0)
        page_id = f"{document_id}_p{page_no:04d}"
        pagesource = SourceRef(
            engine="paddleocr",
            engine_version=engine_version,
            raw_page=page_no,
            raw_path=source_path,
        )
        blocks: list[Block] = []
        for position_in_page, raw_block in enumerate(blocks_data):
            bbox_value = raw_block.get("block_bbox", raw_block.get("bbox"))
            bbox = _number_list(bbox_value, name="PaddleOCR block bbox") if bbox_value is not None else [0.0] * 4
            label = str(raw_block.get("block_label", raw_block.get("label", "unknown")))
            content = raw_block.get("block_content", raw_block.get("text", ""))
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            order = raw_block.get("block_order")
            reading_order = int(order) - 1 if isinstance(order, (int, float)) and order > 0 else position_in_page
            block_source = SourceRef(
                engine="paddleocr",
                engine_version=engine_version,
                raw_page=page_no,
                raw_block_id=str(raw_block.get("block_id", position_in_page)),
                raw_path=source_path,
            )
            metadata: dict[str, Any] = {"raw_label": label}
            polygon = raw_block.get("block_polygon_points")
            for key in ("group_id", "global_block_id", "global_group_id"):
                if key in raw_block:
                    metadata[key] = raw_block[key]
            blocks.append(
                Block(
                    block_id=f"{page_id}_b{position_in_page:04d}",
                    page_id=page_id,
                    block_type=_mineru_block_type(label),
                    bbox=bbox,
                    polygon=polygon,
                    text=content.strip(),
                    reading_order=reading_order,
                    confidence=raw_block.get("confidence", raw_block.get("score")),
                    source=block_source,
                    metadata=metadata,
                )
            )
        pages.append(
            Page(
                page_id=page_id,
                page_no=page_no,
                width=width,
                height=height,
                blocks=blocks,
                source=pagesource,
                image_ref=raw_page.get("input_path"),
            )
        )
    result = DocumentIR(
        document_id=document_id,
        pages=pages,
        source_path=source_path,
        metadata={"source_backend": "paddleocr"},
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
