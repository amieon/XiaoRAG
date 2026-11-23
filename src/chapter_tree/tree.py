"""确定性章节树：显式原生／上游标题优先，编号补充，无证据区域平铺。"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any, Sequence

from doc_chunking.model import normalize_document


TRUSTED_NATIVE_SOURCES = frozenset({
    "docx_style", "docx_outline", "docx_style_outline",
})
HEADING_ROLES = {"heading", "document_title"}
TERMINAL_PUNCTUATION = ("。", "！", "？", "；", ".", "!", "?", ";")


@dataclass(frozen=True)
class ChapterTreeConfig:
    heading_mode: str = "hybrid"  # native / upstream / hybrid
    promote_numbered_headings: bool = True
    max_promoted_heading_chars: int = 80

    def __post_init__(self) -> None:
        if self.heading_mode not in {"native", "upstream", "hybrid"}:
            raise ValueError("heading_mode must be native, upstream or hybrid")
        if self.max_promoted_heading_chars < 1:
            raise ValueError("max_promoted_heading_chars must be positive")


@dataclass
class _Node:
    section_id: str
    parent_id: str | None
    title: str
    level: int
    kind: str
    order: int
    heading_block_id: str | None = None
    heading_source: str | None = None
    block_ids: list[str] | None = None
    content_block_ids: list[str] | None = None
    children: list[str] | None = None

    def __post_init__(self) -> None:
        self.block_ids = [] if self.block_ids is None else self.block_ids
        self.content_block_ids = [] if self.content_block_ids is None else self.content_block_ids
        self.children = [] if self.children is None else self.children


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _native_evidence(block: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = block.get("metadata") or {}
    values = metadata.get("native_structures", metadata.get("native_structure", []))
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        return []
    return [
        item for item in values if isinstance(item, dict)
        and item.get("source") in TRUSTED_NATIVE_SOURCES
        and item.get("role") in HEADING_ROLES
    ]


def _metadata_level(block: dict[str, Any]) -> int | None:
    metadata = block.get("metadata") or {}
    for key in ("heading_level", "title_level", "level"):
        value = metadata.get(key)
        if type(value) is int and value > 0:
            return value
    return None


def _numbering_level(text: str) -> int | None:
    """识别明确的章节编号；第 X 条和小数正文不在此提升为标题。"""
    stripped = text.strip()
    match = re.match(r"^第\s*[0-9一二三四五六七八九十百千万零〇两]+\s*([部篇章节])(?:\s|$)", stripped)
    if match:
        return {"部": 1, "篇": 1, "章": 1, "节": 2}[match.group(1)]
    match = re.match(r"^(\d+(?:\.\d+){0,5})(?:[.、])?\s+\S", stripped)
    return len(match.group(1).split(".")) if match else None


def _can_promote_numbering(text: str, level: int | None, config: ChapterTreeConfig) -> bool:
    return bool(
        config.promote_numbered_headings and level is not None
        and "\n" not in text.strip() and 0 < len(text.strip()) <= config.max_promoted_heading_chars
        and not text.strip().endswith(TERMINAL_PUNCTUATION)
    )


def _upstream_heading(
    block: dict[str, Any], *, predictions: dict[str, dict] | None,
    config: ChapterTreeConfig, warnings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    block_id = block["block_id"]
    native = _native_evidence(block)
    prediction = (predictions or {}).get(block_id)
    native_kind = str(block.get("native_kind") or block.get("kind") or "")

    if config.heading_mode == "native":
        if not native:
            return None
        levels = [item.get("heading_level") for item in native if type(item.get("heading_level")) is int]
        if len(native) != 1 or len(levels) != 1:
            warnings.append({"type": "ambiguous_native_heading_level", "block_id": block_id})
        return {
            "level": levels[0] if len(levels) == 1 else 1,
            "kind": "document_title" if native[0].get("role") == "document_title" else "section",
            "source": native[0].get("source"),
        }

    if native:
        levels = [item.get("heading_level") for item in native if type(item.get("heading_level")) is int]
        return {
            "level": levels[0] if len(levels) == 1 else 1,
            "kind": "document_title" if native[0].get("role") == "document_title" else "section",
            "source": native[0].get("source"),
        }

    if config.heading_mode == "hybrid" and prediction:
        if prediction.get("label") == "heading":
            level = prediction.get("heading_level")
            return {
                "level": level if type(level) is int and level > 0 else 1,
                "kind": "section", "source": prediction.get("mode", "prediction"),
            }
        # Transformer 的 body 可以否决上游标签；native_absent 只是缺证据，不能否决。
        if prediction.get("mode") == "transformer":
            return None

    numbered_level = _numbering_level(_first_line(block.get("text", "")))
    if block.get("kind") == "heading" or native_kind in {"title", "doc_title", "document_title", "paragraph_title", "section_heading"}:
        level = _metadata_level(block) or numbered_level
        if level is None:
            level = 1
            warnings.append({"type": "heading_level_defaulted", "block_id": block_id, "level": 1})
        return {
            "level": level,
            "kind": "document_title" if native_kind in {"doc_title", "document_title", "title"} else "section",
            "source": "upstream_layout_label",
        }
    metadata = block.get("metadata") or {}
    if block.get("kind") == "paragraph" and not metadata.get("list_structures") and _can_promote_numbering(
        block.get("text", ""), numbered_level, config,
    ):
        return {"level": numbered_level, "kind": "section", "source": "explicit_numbering"}
    return None


def build_chapter_tree(
    blocks: Sequence[dict[str, Any]], *, document_id: str,
    config: ChapterTreeConfig | None = None,
    predictions: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按输入块顺序建树；不改写文本，不用 DFS/BFS 替代物理阅读顺序。"""
    config = config or ChapterTreeConfig()
    block_ids = [block.get("block_id") for block in blocks]
    if any(not isinstance(value, str) or not value for value in block_ids) or len(set(block_ids)) != len(block_ids):
        raise ValueError("chapter_tree_requires_unique_block_ids")
    prediction_map = {}
    for item in predictions or []:
        block_id = item.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            raise ValueError("heading_prediction_requires_block_id")
        if block_id in prediction_map:
            raise ValueError(f"duplicate_heading_prediction:{block_id}")
        prediction_map[block_id] = item
    unknown_predictions = sorted(set(prediction_map) - set(block_ids))
    if unknown_predictions:
        raise ValueError(f"predictions_reference_unknown_blocks:{unknown_predictions[:3]}")

    root = _Node(f"{document_id}:sec00000", None, "", 0, "root", 0)
    nodes = [root]
    stack: list[_Node] = []
    decisions = []
    warnings: list[dict[str, Any]] = []
    for block in blocks:
        block_id = block["block_id"]
        heading = _upstream_heading(block, predictions=prediction_map, config=config, warnings=warnings)
        if heading is None:
            parent = stack[-1] if stack else root
            parent.block_ids.append(block_id)
            parent.content_block_ids.append(block_id)
            decisions.append({"block_id": block_id, "is_heading": False, "source": "body"})
            continue

        level = heading["level"]
        while stack and not (
            stack[-1].kind == "document_title" and heading["kind"] != "document_title"
        ) and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else root
        if level > parent.level + 1 and parent.kind != "document_title":
            warnings.append({
                "type": "heading_level_jump", "block_id": block_id,
                "parent_section_id": parent.section_id,
                "parent_level": parent.level, "heading_level": level,
            })
        node = _Node(
            section_id=f"{document_id}:sec{len(nodes):05d}",
            parent_id=parent.section_id,
            title=_first_line(block.get("text", "")),
            level=level,
            kind=heading["kind"],
            order=len(nodes),
            heading_block_id=block_id,
            heading_source=heading["source"],
            block_ids=[block_id],
        )
        parent.children.append(node.section_id)
        nodes.append(node)
        stack.append(node)
        decisions.append({
            "block_id": block_id, "is_heading": True, "heading_level": level,
            "section_id": node.section_id, "source": heading["source"],
        })

    by_id = {node.section_id: node for node in nodes}

    def path(node: _Node) -> list[str]:
        result, current = [], node
        while current.parent_id is not None:
            if current.title:
                result.append(current.title)
            current = by_id[current.parent_id]
        return list(reversed(result))

    serialized_nodes = []
    for node in nodes:
        serialized_nodes.append({
            "section_id": node.section_id,
            "parent_section_id": node.parent_id,
            "child_section_ids": node.children,
            "order": node.order,
            "depth": len(path(node)),
            "level": node.level,
            "kind": node.kind,
            "title": node.title,
            "title_path": path(node),
            "heading_block_id": node.heading_block_id,
            "heading_source": node.heading_source,
            "block_ids": node.block_ids,
            "content_block_ids": node.content_block_ids,
        })
    represented = [block_id for node in nodes for block_id in node.block_ids]
    counts = Counter(represented)
    missing = sorted(set(block_ids) - set(represented))
    duplicated = sorted(value for value, count in counts.items() if count > 1)
    if missing or duplicated or len(represented) != len(block_ids):
        raise ValueError(f"chapter_tree_source_coverage:missing={missing[:3]},duplicated={duplicated[:3]}")
    return {
        "schema_version": "chapter-tree.v1",
        "document_id": document_id,
        "root_section_id": root.section_id,
        "section_order": "preorder",
        "nodes": serialized_nodes,
        "block_decisions": decisions,
        "stats": {
            "input_block_count": len(blocks),
            "heading_block_count": sum(item["is_heading"] for item in decisions),
            "section_count": len(nodes) - 1,
            "empty_section_count": sum(not node.content_block_ids for node in nodes[1:]),
            "max_depth": max((item["depth"] for item in serialized_nodes), default=0),
            "flat_document": len(nodes) == 1,
        },
        "warnings": warnings,
        "config": asdict(config),
        "self_check": {
            "status": "passed",
            "represented_block_count": len(represented),
            "missing_block_count": len(missing),
            "duplicate_block_count": len(duplicated),
        },
    }


def build_chapter_tree_from_ir(
    ir: dict[str, Any], *, config: ChapterTreeConfig | None = None,
    predictions: Sequence[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blocks = normalize_document(ir)
    return build_chapter_tree(
        blocks, document_id=ir["document_id"], config=config, predictions=predictions,
    ), blocks
