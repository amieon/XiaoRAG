"""上游区域顺序的保留与审计；这里不做坐标重排或 XY-Cut。"""
from __future__ import annotations

from typing import Any, Sequence


READING_ORDER_SOURCE = "layout_region_order"


def upstream_reading_order_metadata(
    regions: Sequence[Any], *, page_id: str,
) -> dict[str, Any]:
    """region.index 即模型顺序；只记录来源，不修改区域或正文块。"""
    return {
        "status": "model_provided",
        "source": READING_ORDER_SOURCE,
        "ordered_block_ids": [
            f"{page_id}_b{region.index:04d}" for region in regions
        ],
    }
