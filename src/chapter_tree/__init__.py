"""crop 上下文、标题判断、确定性章节树与父子分块兼容 group。"""

from .context import build_crop_context
from .groups import chapter_groups_document, chapter_tree_to_groups
from .heading_classifier import (
    BODY_LABEL,
    HEADING_LABEL,
    LABELS,
    HeadingExample,
    TransformerHeadingClassifier,
    format_model_input,
    load_heading_examples,
    split_examples_by_document,
    train_transformer,
)
from .native_classifier import NativeHeadingClassifier
from .reading_order import READING_ORDER_SOURCE, upstream_reading_order_metadata
from .tree import ChapterTreeConfig, build_chapter_tree, build_chapter_tree_from_ir

__all__ = [
    "BODY_LABEL", "HEADING_LABEL", "LABELS", "HeadingExample",
    "TransformerHeadingClassifier", "NativeHeadingClassifier", "format_model_input", "load_heading_examples",
    "split_examples_by_document", "train_transformer", "build_crop_context",
    "READING_ORDER_SOURCE", "upstream_reading_order_metadata",
    "ChapterTreeConfig", "build_chapter_tree", "build_chapter_tree_from_ir",
    "chapter_tree_to_groups", "chapter_groups_document",
]
