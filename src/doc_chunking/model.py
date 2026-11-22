"""把本项目的 DocumentIR 适配为切块前处理使用的公共块序列。

这里不生成章节树、不选择切分策略，也不改变正文。页面与块按上游已经给出的
page_no/reading_order 排序；BFS/DFS 只适用于章节树遍历，不能替代物理阅读顺序。
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


KINDS = {
    "text": "paragraph",
    "body": "paragraph",
    "paragraph": "paragraph",
    "title": "heading",
    "doc_title": "heading",
    "document_title": "heading",
    "paragraph_title": "heading",
    "section_heading": "heading",
    "figure": "image",
    "image": "image",
    "figure_caption": "caption",
    "image_caption": "caption",
    "figure_title": "caption",
    "table_title": "caption",
    "display_formula": "formula",
    "equation": "formula",
    "interline_equation": "formula",
    "inline_formula": "formula",
    "inline_equation": "formula",
    "list": "list_item",
    "list_item": "list_item",
    "reference": "paragraph",
    "header": "header",
    "page_header": "header",
    "header_image": "header",
    "footer": "footer",
    "page_footer": "footer",
    "footer_image": "footer",
    "page_number": "page_number",
    "number": "page_number",
}


def _metadata(value: dict[str, Any]) -> dict[str, Any]:
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("document_ir_metadata_requires_object")
    return metadata


def _page_number(page: dict[str, Any]) -> int | None:
    value = page.get("page_no", page.get("number", page.get("page_number")))
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("document_ir_page_numbers_must_be_positive_integers")
    return value


def _unit_type(page: dict[str, Any]) -> str:
    metadata = _metadata(page)
    value = metadata.get(
        "unit_type",
        metadata.get("page_kind", "physical_page" if _page_number(page) is not None else "logical_document"),
    )
    return value if isinstance(value, str) and value else "logical_document"


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _box_on_page(block: dict[str, Any], page: dict[str, Any]) -> list[float] | None:
    box = block.get("bbox")
    width, height = page.get("width"), page.get("height")
    if not (_number(width) and _number(height) and width > 0 and height > 0):
        return None
    if not (isinstance(box, list) and len(box) == 4 and all(_number(value) for value in box)):
        return None
    x0, y0, x1, y1 = box
    if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
        return None
    return deepcopy(box)


def _source_type(raw: dict[str, Any], page: dict[str, Any]) -> str:
    metadata = _metadata(raw)
    page_metadata = _metadata(page)
    value = raw.get(
        "source_type",
        metadata.get("source_type", page.get("source_type", page_metadata.get("source_type"))),
    )
    value = {"native_office": "native", "ocr_text": "ocr"}.get(value, value)
    return value if isinstance(value, str) and value else "unknown"


def _native_kind(raw: dict[str, Any]) -> str:
    value = raw.get("block_type", raw.get("kind", raw.get("type")))
    if not isinstance(value, str) or not value:
        raise ValueError("document_ir_block_requires_kind")
    # 本项目部分布局标签保存在 metadata.raw_label，而 block_type 仍是 paragraph。
    raw_label = _metadata(raw).get("raw_label")
    return raw_label if isinstance(raw_label, str) and raw_label else value


def _positive_level(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def normalize_document(ir: dict[str, Any]) -> list[dict[str, Any]]:
    """拍平页面块，保留阅读顺序、结构父指针和可定位来源。

    支持本项目 canonical DocumentIR，也兼容学长 IR 的 id/kind/order 等字段别名。
    返回值是深拷贝，不修改输入；缺失 reading_order 时只使用页内原始位置兜底。
    """
    if not isinstance(ir, dict) or not isinstance(ir.get("pages"), list):
        raise ValueError("document_ir_requires_pages_list")
    document_id = ir.get("document_id")
    if not isinstance(document_id, str) or not document_id or len(document_id) > 256:
        raise ValueError("document_id_required")
    sections = {
        section.get("id"): section
        for section in ir.get("sections", []) or []
        if isinstance(section, dict)
    }

    indexed_pages = list(enumerate(ir["pages"]))
    for _, page in indexed_pages:
        if not isinstance(page, dict) or not isinstance(page.get("blocks"), list):
            raise ValueError("document_ir_page_requires_blocks_list")
        _page_number(page)
    # 页码是物理顺序；无页码的 PPT/逻辑单元保持输入顺序。
    indexed_pages.sort(key=lambda item: (_page_number(item[1]) is None, _page_number(item[1]) or item[0] + 1, item[0]))

    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for unit_index, (page_input_index, page) in enumerate(indexed_pages, 1):
        unit_id = page.get("page_id", page.get("id", f"unit_{unit_index}"))
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("invalid_unit_id")
        page_number = _page_number(page)
        unit_type = _unit_type(page)
        if unit_type == "physical_page" and page_number is None:
            raise ValueError("physical_page_requires_page_number")
        ordered_blocks = list(enumerate(page["blocks"]))
        for input_order, raw in ordered_blocks:
            if not isinstance(raw, dict):
                raise ValueError("document_ir_block_requires_object")
            block_id = raw.get("block_id", raw.get("id"))
            if not isinstance(block_id, str) or not block_id or block_id in seen_ids:
                raise ValueError("document_ir_block_ids_must_be_unique_strings")
            seen_ids.add(block_id)
            text = raw.get("text", "")
            if not isinstance(text, str):
                raise ValueError("document_ir_block_text_requires_string")
            metadata = deepcopy(_metadata(raw))
            native_kind = _native_kind(raw)
            kind = KINDS.get(native_kind, native_kind)
            source = raw.get("source") or {}
            if not isinstance(source, dict):
                raise ValueError("block_source_requires_object")
            confidence = raw.get("confidence")
            if not (_number(confidence) and 0 <= confidence <= 1):
                confidence = None
            section = sections.get(block_id, {})
            level = metadata.get(
                "heading_level",
                metadata.get("title_level", metadata.get("level", raw.get("level", section.get("level")))),
            )
            raw_order = raw.get("reading_order", raw.get("order"))
            order = raw_order if raw_order is not None else input_order
            if not _number(order):
                raise ValueError("document_ir_order_requires_number")
            if raw.get("page_number", page_number) != page_number:
                raise ValueError("document_ir_inconsistent_block_page")
            if raw.get("page_id") and page.get("page_id") and raw["page_id"] != page["page_id"]:
                raise ValueError("document_ir_inconsistent_block_page")
            ref = {
                "block_id": block_id,
                "unit_id": unit_id,
                "unit_index": unit_index,
                "unit_type": unit_type,
                "page_number": page_number if unit_type == "physical_page" else None,
                "bbox": _box_on_page(raw, page),
                "polygon": deepcopy(raw.get("polygon")),
                "source_type": _source_type(raw, page),
                "confidence": confidence,
                "native_id": raw.get("native_id", source.get("raw_block_id")),
                "source": deepcopy(source),
            }
            result.append({
                "block_id": block_id,
                "document_id": document_id,
                "kind": kind,
                "native_kind": native_kind,
                "text": text,
                "level": _positive_level(level),
                "parent_id": raw.get("parent_id"),
                "unit_index": unit_index,
                "unit_id": unit_id,
                "unit_type": unit_type,
                "page_number": page_number if unit_type == "physical_page" else None,
                "order": order,
                "input_order": input_order,
                "page_input_index": page_input_index,
                "metadata": metadata,
                "sources": [ref],
            })
    # 稳定排序保证相同/缺失 reading_order 不会改变原始相对顺序。
    return sorted(result, key=lambda block: (block["unit_index"], block["order"], block["input_order"]))


def unique_refs(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按来源块和字符区间去重，保持首次出现顺序。"""
    result, seen = [], set()
    for block in blocks:
        for ref in block.get("sources", []):
            key = (ref.get("unit_id"), ref.get("block_id"), ref.get("char_start"), ref.get("char_end"))
            if key not in seen:
                result.append(deepcopy(ref))
                seen.add(key)
    return result
