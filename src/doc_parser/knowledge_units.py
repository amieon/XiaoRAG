"""知识单元构建：由 DocumentIR 构建元素位置表、语义树与知识单元。

全过程不调用模型。所有 ID、顺序和正文均由输入与配置确定性生成。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .document_ir import Block, DocumentIR, Page


KNOWLEDGE_UNIT_BUILDER_VERSION = "knowledge-unit-v1"
TOKENIZER_NAME = "unicode-word-v1"
_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\s]",
    re.UNICODE,
)
_NATURAL_BREAKS = set("。！？；.!?;")
_IGNORED_LABELS = ("number", "header", "footer", "header_image", "footer_image")


@dataclass(frozen=True)
class KnowledgeUnitConfig:
    """知识单元构建切分参数；token 特指 TOKENIZER_NAME，而非某个向量模型词表。"""

    target_tokens: int = 600
    min_tokens: int = 120
    max_tokens: int = 1000
    prefix_max_chars: int = 240
    review_confidence: float = 0.6
    ignored_raw_labels: tuple[str, ...] = _IGNORED_LABELS

    def __post_init__(self) -> None:
        values = (self.min_tokens, self.target_tokens, self.max_tokens)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
            raise ValueError("Knowledge unit token limits must be positive integers")
        if not self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("KnowledgeUnitConfig requires min_tokens <= target_tokens <= max_tokens")
        if not isinstance(self.prefix_max_chars, int) or self.prefix_max_chars < 32:
            raise ValueError("prefix_max_chars must be an integer >= 32")
        if not 0 <= self.review_confidence <= 1:
            raise ValueError("review_confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeUnitConfig":
        data = dict(value)
        if "ignored_raw_labels" in data:
            data["ignored_raw_labels"] = tuple(data["ignored_raw_labels"])
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeUnitConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class KnowledgeUnit:
    unit_id: str
    document_id: str
    source_content_hash: str | None
    parent_unit_id: str | None
    child_unit_ids: list[str]
    unit_type: str
    heading_path: list[str]
    text: str
    context_prefix: str
    token_count: int
    tokenizer_name: str
    block_ids: list[str]
    page_start: int
    page_end: int
    reading_order_start: int
    reading_order_end: int
    asset_paths: list[str]
    content_hash: str
    parser_pipeline: str | None
    builder_version: str = KNOWLEDGE_UNIT_BUILDER_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ElementLocation:
    location_id: str
    document_id: str
    unit_id: str
    block_id: str
    page_id: str
    page_no: int
    bbox: list[float]
    polygon: list[list[float]] | None
    page_width: float
    page_height: float
    reading_order: int | None
    text_start: int
    text_end: int
    source: dict[str, Any] | None
    asset_path: str | None
    evidence_hash: str


@dataclass(slots=True)
class KnowledgeUnitBuildResult:
    units: list[KnowledgeUnit]
    locations: list[ElementLocation]
    tree: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(slots=True)
class _Piece:
    page: Page
    block: Block
    text: str
    text_start: int
    text_end: int
    fallback_order: int

    @property
    def order(self) -> int:
        value = self.block.reading_order
        return value if value is not None else self.fallback_order


@dataclass(slots=True)
class _Atom:
    pieces: list[_Piece]
    unit_type: str
    parent_unit_id: str | None
    heading_path: list[str]
    split_reason: str = "atomic_block"
    force_single: bool = False
    consumed: bool = False
    caption: str | None = None

    @property
    def page_no(self) -> int:
        return self.pieces[0].page.page_no

    @property
    def token_count(self) -> int:
        return len(tokenize("\n\n".join(piece.text.strip() for piece in self.pieces if piece.text.strip())))


@dataclass(slots=True)
class _Section:
    unit_id: str
    level: int
    heading_path: list[str]


def tokenize(text: str) -> list[str]:
    """本地确定性长度单位：CJK 单字、拉丁词、数字和其他非空白符号。"""
    return [match.group(0) for match in _TOKEN_PATTERN.finditer(text)]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _label(block: Block) -> str:
    return str(block.metadata.get("raw_label", block.block_type)).lower()


def _kind(block: Block) -> str:
    label = _label(block)
    if "caption" in label or label in {"figure_title", "table_title"}:
        return "caption"
    if block.block_type == "table":
        return "table"
    if block.block_type == "figure":
        return "figure"
    if block.block_type == "formula":
        return "formula"
    if block.block_type == "list":
        return "list"
    return "paragraph"


def _is_heading(block: Block) -> bool:
    return block.block_type in {"document_title", "section_heading"}


def _heading_level(block: Block, diagnostics: list[dict[str, Any]]) -> int:
    if block.block_type == "document_title":
        return 0
    value = block.metadata.get("heading_level", block.metadata.get("level"))
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    if value is not None:
        diagnostics.append({"type": "invalid_heading_level", "block_id": block.block_id, "value": value})
    else:
        diagnostics.append({"type": "heading_level_defaulted", "block_id": block.block_id, "value": 1})
    return 1


def _context_prefix(document: DocumentIR, heading_path: list[str], config: KnowledgeUnitConfig) -> str:
    parts: list[str] = []
    if document.title:
        parts.append(document.title)
    for title in heading_path:
        if title and (not parts or title != parts[-1]):
            parts.append(title)
    full = " > ".join(parts)
    if len(full) <= config.prefix_max_chars:
        return full
    if not parts:
        return full[:config.prefix_max_chars]
    # 截断时优先保留文档标题和最近两级标题。
    head, tail = parts[0], parts[-2:]
    candidate = " > … > ".join([head, " > ".join(tail)]) if tail and tail != [head] else head
    if len(candidate) > config.prefix_max_chars:
        candidate = candidate[:config.prefix_max_chars - 1] + "…"
    return candidate


def _unit_id(document_id: str, atom: _Atom, *, container: bool) -> str:
    refs = [
        f"{piece.block.block_id}:{piece.text_start}:{piece.text_end}"
        for piece in atom.pieces
    ]
    canonical = json.dumps({
        "document_id": document_id,
        "parent": atom.parent_unit_id,
        "heading_path": atom.heading_path,
        "type": atom.unit_type,
        "container": container,
        "refs": refs,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "ku_" + _sha256(canonical)[:24]


def _split_piece(piece: _Piece, config: KnowledgeUnitConfig) -> tuple[list[_Piece], bool]:
    """返回块内互不重叠的字符区间，以及是否发生过无自然边界的硬切。"""
    matches = list(_TOKEN_PATTERN.finditer(piece.text))
    if len(matches) <= config.max_tokens:
        return [piece], False
    pieces: list[_Piece] = []
    token_start = 0
    char_start = 0
    used_fallback = False
    while len(matches) - token_start > config.max_tokens:
        target_end = token_start + config.target_tokens
        limit_end = token_start + config.max_tokens
        chosen = None
        for end in range(target_end, limit_end + 1):
            if matches[end - 1].group(0) in _NATURAL_BREAKS:
                chosen = end
        if chosen is None:
            # 换行也是自然边界；取目标与上限之间最后一个换行。
            raw_start = matches[target_end - 1].start()
            raw_end = matches[limit_end - 1].end()
            newline = piece.text.rfind("\n", raw_start, raw_end)
            if newline >= 0:
                end_char = newline + 1
                chosen = limit_end
                while chosen > target_end and matches[chosen - 1].start() >= end_char:
                    chosen -= 1
        if chosen is None:
            chosen = limit_end
            used_fallback = True
        end_char = matches[chosen - 1].end()
        pieces.append(_Piece(
            page=piece.page, block=piece.block, text=piece.text[char_start:end_char],
            text_start=char_start, text_end=end_char, fallback_order=piece.fallback_order,
        ))
        token_start = chosen
        char_start = end_char
    if char_start < len(piece.text):
        pieces.append(_Piece(
            page=piece.page, block=piece.block, text=piece.text[char_start:],
            text_start=char_start, text_end=len(piece.text), fallback_order=piece.fallback_order,
        ))
    return pieces, used_fallback


def _expand_long_atom(atom: _Atom, config: KnowledgeUnitConfig) -> list[_Atom]:
    if atom.unit_type not in {"paragraph", "list"}:
        return [atom]
    if not any(len(tokenize(piece.text)) > config.max_tokens for piece in atom.pieces):
        return [atom]
    expanded: list[_Atom] = []
    used_fallback = False
    for piece in atom.pieces:
        pieces, fallback = _split_piece(piece, config)
        used_fallback = used_fallback or fallback
        for part in pieces:
            expanded.append(_Atom(
                pieces=[part], unit_type=atom.unit_type,
                parent_unit_id=atom.parent_unit_id, heading_path=list(atom.heading_path),
                split_reason="length_fallback" if fallback else "length_boundary",
                force_single=True, caption=atom.caption,
            ))
    return expanded


def _same_caption_context(left: _Atom, right: _Atom) -> bool:
    return left.parent_unit_id == right.parent_unit_id and left.page_no == right.page_no


def _merge_captions(atoms: list[_Atom]) -> int:
    """题注并入相邻表格/图片；找不到对象时退化为普通段落。"""
    merged = 0
    for index, atom in enumerate(atoms):
        if atom.unit_type != "caption":
            continue
        target = None
        if index + 1 < len(atoms):
            candidate = atoms[index + 1]
            if (not candidate.consumed and candidate.unit_type in {"table", "figure"}
                    and _same_caption_context(atom, candidate)):
                target = candidate
        if target is None and index > 0:
            candidate = atoms[index - 1]
            if (not candidate.consumed and candidate.unit_type in {"table", "figure"}
                    and _same_caption_context(atom, candidate)):
                target = candidate
        if target is None:
            atom.unit_type = "paragraph"
            continue
        target.pieces.extend(atom.pieces)
        target.pieces.sort(key=lambda piece: (piece.page.page_no, piece.order, piece.block.block_id))
        target.caption = "\n".join(piece.text.strip() for piece in atom.pieces if piece.text.strip())
        atom.consumed = True
        merged += 1
    return merged


class _Builder:
    def __init__(self, document: DocumentIR, config: KnowledgeUnitConfig) -> None:
        self.document = document
        self.config = config
        self.units: list[KnowledgeUnit] = []
        self.unit_by_id: dict[str, KnowledgeUnit] = {}
        self.pieces_by_unit: dict[str, list[_Piece]] = {}

    def add_unit(self, atom: _Atom, *, container: bool = False, heading_level: int | None = None) -> KnowledgeUnit:
        atom.pieces.sort(key=lambda piece: (piece.page.page_no, piece.order, piece.block.block_id))
        text = "\n\n".join(piece.text.strip() for piece in atom.pieces if piece.text.strip())
        unit_id = _unit_id(self.document.document_id, atom, container=container)
        if unit_id in self.unit_by_id:
            raise ValueError(f"Deterministic memory unit ID collision: {unit_id}")
        pages = [piece.page.page_no for piece in atom.pieces]
        orders = [piece.order for piece in atom.pieces]
        assets = list(dict.fromkeys(
            str(piece.block.metadata["asset_path"])
            for piece in atom.pieces if piece.block.metadata.get("asset_path")
        ))
        confidences = [piece.block.confidence for piece in atom.pieces if piece.block.confidence is not None]
        minimum_confidence = min(confidences) if confidences else None
        metadata: dict[str, Any] = {
            "split_reason": atom.split_reason,
            "source_confidence_min": minimum_confidence,
            "needs_review": any(value < self.config.review_confidence for value in confidences),
            "is_container": container,
            "indexable": not container,
            "shared_block_split": False,
        }
        if atom.caption:
            metadata["caption"] = atom.caption
        if heading_level is not None:
            metadata["heading_level"] = heading_level
        unit = KnowledgeUnit(
            unit_id=unit_id,
            document_id=self.document.document_id,
            source_content_hash=self.document.content_hash,
            parent_unit_id=atom.parent_unit_id,
            child_unit_ids=[],
            unit_type=atom.unit_type,
            heading_path=list(atom.heading_path),
            text=text,
            context_prefix=_context_prefix(self.document, atom.heading_path, self.config),
            token_count=len(tokenize(text)),
            tokenizer_name=TOKENIZER_NAME,
            block_ids=[piece.block.block_id for piece in atom.pieces],
            page_start=min(pages),
            page_end=max(pages),
            reading_order_start=min(orders),
            reading_order_end=max(orders),
            asset_paths=assets,
            content_hash=_sha256(text),
            parser_pipeline=self.document.metadata.get("pipeline"),
            metadata=metadata,
        )
        self.units.append(unit)
        self.unit_by_id[unit_id] = unit
        self.pieces_by_unit[unit_id] = list(atom.pieces)
        if unit.parent_unit_id is not None:
            self.unit_by_id[unit.parent_unit_id].child_unit_ids.append(unit_id)
        return unit

    def build_locations(self) -> list[ElementLocation]:
        locations: list[ElementLocation] = []
        for unit in self.units:
            for piece in self.pieces_by_unit[unit.unit_id]:
                block = piece.block
                location_id = "loc_" + _sha256(
                    f"{unit.unit_id}|{block.block_id}|{piece.text_start}|{piece.text_end}"
                )[:24]
                locations.append(ElementLocation(
                    location_id=location_id,
                    document_id=self.document.document_id,
                    unit_id=unit.unit_id,
                    block_id=block.block_id,
                    page_id=piece.page.page_id,
                    page_no=piece.page.page_no,
                    bbox=list(block.bbox),
                    polygon=block.polygon,
                    page_width=piece.page.width,
                    page_height=piece.page.height,
                    reading_order=block.reading_order,
                    text_start=piece.text_start,
                    text_end=piece.text_end,
                    source=asdict(block.source) if block.source is not None else None,
                    asset_path=block.metadata.get("asset_path"),
                    evidence_hash=_sha256(piece.text),
                ))
        by_block: dict[str, list[ElementLocation]] = {}
        for location in locations:
            by_block.setdefault(location.block_id, []).append(location)
        for block_id, rows in by_block.items():
            unit_ids = {row.unit_id for row in rows}
            if len(unit_ids) > 1:
                for unit_id in unit_ids:
                    self.unit_by_id[unit_id].metadata["shared_block_split"] = True
            ordered = sorted(rows, key=lambda row: (row.text_start, row.text_end, row.unit_id))
            for previous, current in zip(ordered, ordered[1:]):
                if current.text_start < previous.text_end:
                    raise ValueError(f"Overlapping ownership of source block: {block_id}")
        return locations


def _tree(document: DocumentIR, units: list[KnowledgeUnit]) -> dict[str, Any]:
    nodes = {
        unit.unit_id: {
            "unit_id": unit.unit_id,
            "node_type": "container" if unit.metadata["is_container"] else "memory_unit",
            "unit_type": unit.unit_type,
            "heading_path": unit.heading_path,
            "indexable": unit.metadata["indexable"],
            "children": [],
        }
        for unit in units
    }
    roots = []
    for unit in units:
        node = nodes[unit.unit_id]
        if unit.parent_unit_id is None:
            roots.append(node)
        else:
            nodes[unit.parent_unit_id]["children"].append(node)
    reachable = 0
    def count(node):
        nonlocal reachable
        reachable += 1
        for child in node["children"]:
            count(child)
    for root in roots:
        count(root)
    if reachable != len(units):
        raise ValueError("Memory tree contains a cycle or an unreachable parent chain")
    return {
        "schema_version": "document-tree.v1",
        "document_id": document.document_id,
        "source_content_hash": document.content_hash,
        "children": roots,
    }


def build_knowledge_units(document: DocumentIR, config: KnowledgeUnitConfig | None = None) -> KnowledgeUnitBuildResult:
    """从一份已校验 DocumentIR 构建完整的知识单元结果。"""
    config = config or KnowledgeUnitConfig()
    document.validate()
    builder = _Builder(document, config)
    atoms: list[_Atom] = []
    omitted: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    stack: list[_Section] = []
    missing_reading_order = 0

    for page in sorted(document.pages, key=lambda item: item.page_no):
        ordered = sorted(
            enumerate(page.blocks),
            key=lambda item: (item[1].reading_order if item[1].reading_order is not None else 10**9, item[0]),
        )
        for fallback_order, block in ordered:
            missing_reading_order += int(block.reading_order is None)
            label = _label(block)
            kind = _kind(block)
            if label in config.ignored_raw_labels:
                omitted.append({"block_id": block.block_id, "reason": "ignored_raw_label"})
                continue
            if not block.text.strip() and not _is_heading(block) and kind not in {"table", "figure", "formula"}:
                omitted.append({"block_id": block.block_id, "reason": "empty_text"})
                continue
            piece = _Piece(page, block, block.text, 0, len(block.text), fallback_order)
            if _is_heading(block):
                if not block.text.strip():
                    omitted.append({"block_id": block.block_id, "reason": "empty_heading"})
                    continue
                level = _heading_level(block, diagnostics)
                while stack and stack[-1].level >= level:
                    stack.pop()
                parent = stack[-1] if stack else None
                previous_level = parent.level if parent else 0
                if level > previous_level + 1:
                    diagnostics.append({
                        "type": "heading_level_jump", "block_id": block.block_id,
                        "from": previous_level, "to": level,
                    })
                heading_path = list(parent.heading_path) if parent else []
                heading_path.append(block.text.strip())
                atom = _Atom(
                    pieces=[piece],
                    unit_type="document_title" if level == 0 else "section",
                    parent_unit_id=parent.unit_id if parent else None,
                    heading_path=heading_path,
                    split_reason="heading_boundary",
                )
                unit = builder.add_unit(atom, container=True, heading_level=level)
                stack.append(_Section(unit.unit_id, level, heading_path))
                continue
            parent = stack[-1] if stack else None
            atoms.append(_Atom(
                pieces=[piece], unit_type=kind,
                parent_unit_id=parent.unit_id if parent else None,
                heading_path=list(parent.heading_path) if parent else [],
            ))

    caption_merges = _merge_captions(atoms)
    expanded: list[_Atom] = []
    for atom in atoms:
        if not atom.consumed:
            expanded.extend(_expand_long_atom(atom, config))

    current: _Atom | None = None
    def flush() -> None:
        nonlocal current
        if current is not None:
            builder.add_unit(current)
            current = None

    for atom in expanded:
        if atom.unit_type not in {"paragraph", "list"} or atom.force_single:
            flush()
            builder.add_unit(atom)
            continue
        if current is None:
            current = atom
            continue
        compatible = (
            current.unit_type == atom.unit_type
            and current.parent_unit_id == atom.parent_unit_id
            and current.page_no == atom.page_no  # 首版不猜测跨页续段。
        )
        combined_tokens = current.token_count + atom.token_count
        should_merge = compatible and combined_tokens <= config.max_tokens and (
            current.token_count < config.target_tokens or atom.token_count < config.min_tokens
        )
        if not should_merge:
            flush()
            current = atom
            continue
        current.pieces.extend(atom.pieces)
        current.split_reason = "feature_group"
    flush()

    locations = builder.build_locations()
    tree = _tree(document, builder.units)
    source_block_ids = {block.block_id for page in document.pages for block in page.blocks}
    mapped_block_ids = {location.block_id for location in locations}
    omitted_block_ids = {item["block_id"] for item in omitted}
    missing = sorted(source_block_ids - mapped_block_ids - omitted_block_ids)
    unexpected = sorted((mapped_block_ids | omitted_block_ids) - source_block_ids)
    overlap = sorted(mapped_block_ids & omitted_block_ids)
    if missing or unexpected or overlap:
        raise ValueError(f"Knowledge unit coverage failure: missing={missing}, unexpected={unexpected}, overlap={overlap}")

    indexable = [unit for unit in builder.units if unit.metadata["indexable"]]
    lengths = [unit.token_count for unit in indexable]
    ir_canonical = json.dumps(
        document.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    manifest = {
        "schema_version": "knowledge-unit-manifest.v1",
        "status": "success",
        "builder_version": KNOWLEDGE_UNIT_BUILDER_VERSION,
        "document_id": document.document_id,
        "source_content_hash": document.content_hash,
        "document_ir_sha256": _sha256(ir_canonical),
        "config": config.to_dict(),
        "tokenizer_name": TOKENIZER_NAME,
        "counts": {
            "source_blocks": len(source_block_ids),
            "mapped_blocks": len(mapped_block_ids),
            "omitted_blocks": len(omitted),
            "units": len(builder.units),
            "indexable_units": len(indexable),
            "container_units": len(builder.units) - len(indexable),
            "locations": len(locations),
            "caption_merges": caption_merges,
            "length_split_units": sum(
                unit.metadata["split_reason"].startswith("length_") for unit in indexable
            ),
            "needs_review_units": sum(unit.metadata["needs_review"] for unit in indexable),
            "missing_reading_order": missing_reading_order,
        },
        "token_lengths": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        },
        "coverage_ratio": len(mapped_block_ids) / len(source_block_ids) if source_block_ids else 1.0,
        "omitted": omitted,
        "diagnostics": diagnostics,
    }
    result = KnowledgeUnitBuildResult(builder.units, locations, tree, manifest)
    validate_knowledge_units(document, result)
    return result


def validate_knowledge_units(document: DocumentIR, result: KnowledgeUnitBuildResult) -> None:
    """校验真实块/页引用、字符证据、树关系、坐标与块覆盖。"""
    document.validate()
    source_blocks: dict[str, tuple[Page, Block, int]] = {}
    for page in document.pages:
        for fallback_order, block in enumerate(page.blocks):
            if block.block_id in source_blocks:
                raise ValueError(f"Source block IDs must be document-wide unique: {block.block_id}")
            source_blocks[block.block_id] = (page, block, fallback_order)

    unit_ids = [unit.unit_id for unit in result.units]
    location_ids = [location.location_id for location in result.locations]
    if len(unit_ids) != len(set(unit_ids)) or len(location_ids) != len(set(location_ids)):
        raise ValueError("Knowledge unit unit/location IDs must be unique")
    units = {unit.unit_id: unit for unit in result.units}
    for unit in result.units:
        if unit.document_id != document.document_id or unit.source_content_hash != document.content_hash:
            raise ValueError(f"Unit document identity mismatch: {unit.unit_id}")
        if unit.content_hash != _sha256(unit.text) or unit.token_count != len(tokenize(unit.text)):
            raise ValueError(f"Unit content statistics mismatch: {unit.unit_id}")
        if len(unit.block_ids) != len(set(unit.block_ids)):
            raise ValueError(f"Duplicate block reference inside unit: {unit.unit_id}")
        if unit.parent_unit_id is not None:
            if unit.parent_unit_id not in units:
                raise ValueError(f"Dangling parent: {unit.unit_id}")
            parent = units[unit.parent_unit_id]
            if unit.unit_id not in parent.child_unit_ids or not parent.metadata.get("is_container"):
                raise ValueError(f"Inconsistent parent relation: {unit.unit_id}")
        for child in unit.child_unit_ids:
            if child not in units or units[child].parent_unit_id != unit.unit_id:
                raise ValueError(f"Inconsistent child relation: {unit.unit_id} -> {child}")

    locations_by_unit: dict[str, list[ElementLocation]] = {}
    locations_by_block: dict[str, list[ElementLocation]] = {}
    for location in result.locations:
        if location.unit_id not in units or location.block_id not in units[location.unit_id].block_ids:
            raise ValueError(f"Dangling element location: {location.location_id}")
        if location.block_id not in source_blocks:
            raise ValueError(f"Unknown source block: {location.block_id}")
        page, block, _ = source_blocks[location.block_id]
        if (location.document_id != document.document_id or location.page_id != page.page_id
                or location.page_no != page.page_no):
            raise ValueError(f"Location page identity mismatch: {location.location_id}")
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (location.text_start, location.text_end)):
            raise ValueError(f"Text ranges must be integers: {location.location_id}")
        if not 0 <= location.text_start <= location.text_end <= len(block.text):
            raise ValueError(f"Invalid text range: {location.location_id}")
        if location.evidence_hash != _sha256(block.text[location.text_start:location.text_end]):
            raise ValueError(f"Evidence hash mismatch: {location.location_id}")
        if (location.bbox != block.bbox or location.polygon != block.polygon
                or location.page_width != page.width or location.page_height != page.height):
            raise ValueError(f"Location geometry mismatch: {location.location_id}")
        if not all(math.isfinite(value) for value in (*location.bbox, page.width, page.height)):
            raise ValueError(f"Non-finite location geometry: {location.location_id}")
        x0, y0, x1, y1 = location.bbox
        if x0 < 0 or y0 < 0 or x1 > page.width or y1 > page.height:
            raise ValueError(f"Location bbox exceeds page: {location.location_id}")
        locations_by_unit.setdefault(location.unit_id, []).append(location)
        locations_by_block.setdefault(location.block_id, []).append(location)

    for unit in result.units:
        rows = locations_by_unit.get(unit.unit_id, [])
        if {row.block_id for row in rows} != set(unit.block_ids):
            raise ValueError(f"Unit/location block set mismatch: {unit.unit_id}")
        if not rows:
            raise ValueError(f"Memory unit has no source location: {unit.unit_id}")
        if [unit.page_start, unit.page_end] != [min(row.page_no for row in rows), max(row.page_no for row in rows)]:
            raise ValueError(f"Unit page range mismatch: {unit.unit_id}")
        orders = []
        for row in rows:
            _, block, fallback_order = source_blocks[row.block_id]
            orders.append(block.reading_order if block.reading_order is not None else fallback_order)
        if [unit.reading_order_start, unit.reading_order_end] != [min(orders), max(orders)]:
            raise ValueError(f"Unit reading-order range mismatch: {unit.unit_id}")

    for block_id, rows in locations_by_block.items():
        farthest_end = 0
        for row in sorted(rows, key=lambda item: (item.text_start, item.text_end, item.unit_id)):
            if row.text_start < farthest_end:
                raise ValueError(f"Overlapping ownership of source block: {block_id}")
            farthest_end = max(farthest_end, row.text_end)

    omitted_rows = result.manifest.get("omitted", [])
    omitted = {item["block_id"] for item in omitted_rows}
    if len(omitted) != len(omitted_rows) or any(not item.get("reason") for item in omitted_rows):
        raise ValueError("Every omitted block must have one explicit reason")
    covered = set(locations_by_block)
    if set(source_blocks) != covered | omitted or covered & omitted or not omitted <= set(source_blocks):
        raise ValueError("Every source block must be either mapped or explicitly omitted")
    if result.tree != _tree(document, result.units):
        raise ValueError("Document tree does not match unit parent/child relations")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_knowledge_units(result: KnowledgeUnitBuildResult, output: str | Path) -> None:
    """写入四个确定性文件；拒绝覆盖非空目录，避免混合两次构建。"""
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Knowledge unit output directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "knowledge-units.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for unit in result.units:
            stream.write(json.dumps(asdict(unit), ensure_ascii=False) + "\n")
    with (root / "element-locations.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for location in result.locations:
            stream.write(json.dumps(asdict(location), ensure_ascii=False) + "\n")
    _write_json(root / "document-tree.json", result.tree)
    _write_json(root / "manifest.json", result.manifest)


def build_knowledge_units_from_json(
    input_path: str | Path,
    output: str | Path,
    *,
    config: KnowledgeUnitConfig | None = None,
) -> KnowledgeUnitBuildResult:
    document = DocumentIR.from_dict(json.loads(Path(input_path).read_text(encoding="utf-8")))
    result = build_knowledge_units(document, config)
    write_knowledge_units(result, output)
    return result
