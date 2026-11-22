"""切块前的公共 IR 规整层；暂不包含策略选择和父子块生成。"""

from .model import normalize_document, unique_refs
from .preprocess import preprocess_document
from .formats import format_groups
from .prepare import prepare_document

__all__ = [
    "normalize_document",
    "unique_refs",
    "preprocess_document",
    "format_groups",
    "prepare_document",
]

VERSION = "0.1.0-preprocess"
