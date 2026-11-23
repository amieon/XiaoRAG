"""为标题结构模型构造可审计的页内 crop 上下文窗口。"""
from __future__ import annotations

from typing import Any, Sequence


def build_crop_context(
    regions: Sequence[Any],
    texts: Sequence[str],
    position: int,
    *,
    page_id: str,
    max_chars: int,
) -> dict[str, Any]:
    """保存相邻 crop 的识别原文窗口；前文取后缀，后文取前缀。"""
    if len(regions) != len(texts):
        raise ValueError("regions and texts must have the same length")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    def neighbor(neighbor_position: int, side: str) -> dict[str, Any] | None:
        if not 0 <= neighbor_position < len(regions):
            return None
        region = regions[neighbor_position]
        text = texts[neighbor_position]
        full_length = len(text)
        if side == "previous":
            start, end = max(0, full_length - max_chars), full_length
        else:
            start, end = 0, min(full_length, max_chars)
        return {
            "block_id": f"{page_id}_b{region.index:04d}",
            "region_index": region.index,
            "raw_label": region.label,
            "text": text[start:end],
            "char_start": start,
            "char_end": end,
            "full_length": full_length,
            "truncated": end - start < full_length,
        }

    return {
        "schema_version": "crop-context.v1",
        "scope": "page",
        "order_source": "layout_region_order",
        "max_chars_per_side": max_chars,
        "previous": neighbor(position - 1, "previous"),
        "next": neighbor(position + 1, "next"),
    }
