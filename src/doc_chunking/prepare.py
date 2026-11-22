"""可单独运行的切块前处理链，不生成知识单元或父子块。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .formats import format_groups
from .model import normalize_document
from .preprocess import preprocess_document


DEFAULT_CONFIG = {
    "preprocess": {
        "strip_furniture": True,
        "strip_toc": True,
        "merge_cross_page": True,
    },
    "formats": {
        "slide_merge_below_chars": 0,
        "slide_merge_max_chars": 800,
        "excel_rows_per_group": 10,
        "excel_header_mode": "first_candidate",
    },
}


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return deepcopy(DEFAULT_CONFIG)
    if not isinstance(config, dict):
        raise ValueError("chunk_preprocess_config_requires_object")
    result = deepcopy(config)
    for section in ("preprocess", "formats"):
        value = result.get(section)
        if value is None:
            result[section] = deepcopy(DEFAULT_CONFIG[section])
        elif not isinstance(value, dict):
            raise ValueError(f"chunk_preprocess_{section}_requires_object")
    return result


def prepare_document(
    ir: dict[str, Any],
    native: dict[str, Any] | None = None,
    source_manifest: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行 normalize → preprocess → format_groups，返回未切块的中间结果。

    profiles 不在这里消费。format_result 为 None 表示普通文档；为字典时仅表示
    已识别 PPT/Excel 自然单元，不代表已经生成父块或子块。
    """
    settings = _config(config)
    native = {} if native is None else native
    source_manifest = {} if source_manifest is None else source_manifest
    if not isinstance(native, dict) or not isinstance(source_manifest, dict):
        raise ValueError("native_and_manifest_require_objects")
    blocks = normalize_document(ir)
    cleaned = preprocess_document(blocks, ir, native, settings)
    # 与学长实现一致：格式通道读取原始清单，PPT 备注可从显式排除区按规则恢复；
    # 普通文档后续只消费 cleaned["blocks"]，不会把页眉页脚带回去。
    formatted = format_groups(
        blocks, ir, native, source_manifest, settings,
    )
    warnings = list(dict.fromkeys(
        cleaned["warnings"] + (formatted.get("warnings", []) if formatted else []),
    ))
    return {
        "schema_version": "chunk-preprocess.v1",
        "document_id": ir["document_id"],
        "blocks": cleaned["blocks"],
        "excluded": cleaned["excluded"],
        "merges": cleaned["merges"],
        "warnings": warnings,
        "format_result": formatted,
        "config": settings,
    }
