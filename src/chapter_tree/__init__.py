"""章节树前置能力：模型顺序保留、crop 上下文与标题二分类。"""

from .context import build_crop_context
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

__all__ = [
    "BODY_LABEL", "HEADING_LABEL", "LABELS", "HeadingExample",
    "TransformerHeadingClassifier", "NativeHeadingClassifier", "format_model_input", "load_heading_examples",
    "split_examples_by_document", "train_transformer", "build_crop_context",
    "READING_ORDER_SOURCE", "upstream_reading_order_metadata",
]
