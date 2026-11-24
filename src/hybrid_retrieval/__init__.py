"""混合检索模块：统一知识单元输入；当前不包含向量化或权重训练。"""

from .adapter import (
    PARENT_SCHEMA,
    PRODUCERS,
    UNIT_SCHEMA,
    adapt_structural_chunks,
    validate_security,
)

__all__ = [
    "PARENT_SCHEMA", "PRODUCERS", "UNIT_SCHEMA",
    "adapt_structural_chunks", "validate_security",
]
