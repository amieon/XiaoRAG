"""公共 IR 规整层；父子块入口位于 doc_chunking.engine，避免章节树循环导入。"""

from .model import normalize_document, unique_refs
from .preprocess import preprocess_document
from .formats import format_groups
from .prepare import prepare_document

__all__ = [
    "normalize_document", "unique_refs", "preprocess_document", "format_groups", "prepare_document",
]

VERSION = "0.2.0-chapter-tree"
