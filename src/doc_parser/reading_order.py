"""Deterministic page reading order: XY-Cut plus single-figure text wrapping.

The module orders existing page blocks only. It never changes text, coordinates,
source references, or block identities. Ambiguous geometry produces an explicit
risky result with a stable coordinate fallback instead of a confident guess.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable

from .document_ir import Block, Page


@dataclass(frozen=True)
class ReadingOrderConfig:
    """Auditable geometry parameters for the model-free first version."""

    enabled: bool = True
    direction: str = "ltr"
    min_horizontal_gap: float = 4.0
    min_vertical_gap: float = 8.0
    max_depth: int = 32
    overlap_ratio_threshold: float = 0.2
    auxiliary_labels: tuple[str, ...] = (
        "header", "footer", "number", "page_number",
        "header_image", "footer_image", "aside_text",
    )
    figure_types: tuple[str, ...] = ("figure",)
    figure_caption_types: tuple[str, ...] = (
        "figure_caption", "image_caption", "figure_title", "image_title",
    )

    def __post_init__(self) -> None:
        if self.direction not in {"ltr", "rtl"}:
            raise ValueError("reading_order.direction must be 'ltr' or 'rtl'")
        for name in ("min_horizontal_gap", "min_vertical_gap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"reading_order.{name} must be a non-negative finite number")
        if not isinstance(self.max_depth, int) or isinstance(self.max_depth, bool) or self.max_depth < 1:
            raise ValueError("reading_order.max_depth must be a positive integer")
        if not 0 <= self.overlap_ratio_threshold <= 1:
            raise ValueError("reading_order.overlap_ratio_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReadingOrderConfig":
        data = dict(value)
        for key in ("auxiliary_labels", "figure_types", "figure_caption_types"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass
class ReadingOrderResult:
    """A complete block permutation plus quality diagnostics."""

    status: str
    source: str
    ordered_block_ids: list[str]
    main_block_ids: list[str]
    auxiliary_block_ids: list[str]
    risks: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raw_label(block: Block) -> str:
    return str(block.metadata.get("raw_label", block.block_type)).lower()


def _is_auxiliary(block: Block, config: ReadingOrderConfig) -> bool:
    return _raw_label(block) in config.auxiliary_labels


def _is_figure(block: Block, config: ReadingOrderConfig) -> bool:
    return block.block_type in config.figure_types


def _is_figure_caption(block: Block, config: ReadingOrderConfig) -> bool:
    label = _raw_label(block)
    return block.block_type in config.figure_caption_types or label in config.figure_caption_types


def _valid_geometry(block: Block) -> bool:
    x0, y0, x1, y1 = block.bbox
    return (
        all(math.isfinite(value) for value in block.bbox)
        and x1 > x0
        and y1 > y0
    )


def _fallback(blocks: Iterable[Block], original: dict[str, int]) -> list[Block]:
    def key(block: Block) -> tuple[float, float, int]:
        if not _valid_geometry(block):
            return (math.inf, math.inf, original[block.block_id])
        return (block.bbox[1], block.bbox[0], original[block.block_id])

    return sorted(blocks, key=key)


def _split_axis(
    blocks: list[Block],
    *,
    axis: int,
    min_gap: float,
) -> list[list[Block]]:
    """Split only at real empty gutters; touching boxes do not form a cut."""
    start, end = axis, axis + 2
    ordered = sorted(blocks, key=lambda block: (block.bbox[start], block.bbox[end]))
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_end = -math.inf
    for block in ordered:
        if current and block.bbox[start] - current_end >= min_gap:
            groups.append(current)
            current = []
        current.append(block)
        current_end = max(current_end, block.bbox[end])
    if current:
        groups.append(current)
    return groups


def _horizontal_overlap(first: Block, second: Block) -> float:
    return max(0.0, min(first.bbox[2], second.bbox[2]) - max(first.bbox[0], second.bbox[0]))


def _nearby_captions(
    figure: Block,
    blocks: list[Block],
    config: ReadingOrderConfig,
) -> list[Block]:
    """Attach only explicit figure captions close to and aligned with the figure."""
    figure_height = figure.bbox[3] - figure.bbox[1]
    max_distance = max(4 * config.min_horizontal_gap, 0.5 * figure_height)
    captions = []
    for block in blocks:
        if block is figure or not _is_figure_caption(block, config):
            continue
        if _horizontal_overlap(figure, block) <= 0:
            continue
        if block.bbox[1] < figure.bbox[1]:
            continue
        distance = max(0.0, block.bbox[1] - figure.bbox[3])
        if distance <= max_distance:
            captions.append(block)
    return captions


def _order_blocks(
    blocks: list[Block],
    *,
    config: ReadingOrderConfig,
    original: dict[str, int],
    risks: list[str],
    strategies: set[str],
    depth: int,
) -> list[Block]:
    if len(blocks) <= 1:
        return list(blocks)
    figures = [block for block in blocks if _is_figure(block, config)]
    if len(figures) == 1 and all(
        block is figures[0] or _is_figure_caption(block, config) for block in blocks
    ):
        # A touching figure and its explicit captions are one resolved object
        # group, not an ambiguous two-block layout.
        strategies.add("figure_caption_group")
        return [figures[0], *_fallback(
            [block for block in blocks if block is not figures[0]], original,
        )]
    if depth > config.max_depth:
        risks.append("maximum_xy_cut_depth_exceeded")
        return _fallback(blocks, original)

    # Prefer a vertical gutter. If both columns happen to contain aligned
    # paragraph gaps, horizontal-first splitting would interleave the columns.
    columns = _split_axis(blocks, axis=0, min_gap=config.min_vertical_gap)
    if len(columns) > 1:
        if config.direction == "rtl":
            columns.reverse()
        strategies.add("vertical_cut")
        result: list[Block] = []
        for column in columns:
            result.extend(_order_blocks(
                column, config=config, original=original, risks=risks,
                strategies=strategies, depth=depth + 1,
            ))
        return result

    # A full-width heading or lower band blocks a vertical cut, so horizontal
    # bands are considered second and recursively ordered on their own.
    bands = _split_axis(blocks, axis=1, min_gap=config.min_horizontal_gap)
    if len(bands) > 1:
        strategies.add("horizontal_cut")
        result = []
        for band in bands:
            result.extend(_order_blocks(
                band, config=config, original=original, risks=risks,
                strategies=strategies, depth=depth + 1,
            ))
        return result

    wrapped = _order_single_figure_wrap(
        blocks, config=config, original=original, risks=risks,
        strategies=strategies, depth=depth,
    )
    if wrapped is not None:
        return wrapped

    risks.append("no_valid_geometric_cut")
    return _fallback(blocks, original)


def _order_single_figure_wrap(
    blocks: list[Block],
    *,
    config: ReadingOrderConfig,
    original: dict[str, int],
    risks: list[str],
    strategies: set[str],
    depth: int,
) -> list[Block] | None:
    """Split around one rectangular figure without requiring polygon crops."""
    figures = [block for block in blocks if _is_figure(block, config)]
    if len(figures) != 1:
        return None
    figure = figures[0]
    captions = _nearby_captions(figure, blocks, config)
    object_group = [figure, *_fallback(captions, original)]
    object_ids = {block.block_id for block in object_group}
    remaining = [block for block in blocks if block.block_id not in object_ids]

    tolerance = config.min_horizontal_gap
    before = [block for block in remaining if block.bbox[3] <= figure.bbox[1] + tolerance]
    after = [block for block in remaining if block.bbox[1] >= figure.bbox[3] - tolerance]
    assigned = {block.block_id for block in before + after}
    during = [block for block in remaining if block.block_id not in assigned]
    left = [block for block in during if block.bbox[2] <= figure.bbox[0] + tolerance]
    right = [block for block in during if block.bbox[0] >= figure.bbox[2] - tolerance]
    side_ids = {block.block_id for block in left + right}

    # A central text obstacle or multiple overlapping flows is outside v1.
    if len(side_ids) != len(during) or not after or not (left or right):
        return None

    strategies.add("single_figure_wrap")
    result = _order_blocks(
        before, config=config, original=original, risks=risks,
        strategies=strategies, depth=depth + 1,
    )
    for side in (left, right):
        result.extend(_order_blocks(
            side, config=config, original=original, risks=risks,
            strategies=strategies, depth=depth + 1,
        ))
    result.extend(object_group)
    result.extend(_order_blocks(
        after, config=config, original=original, risks=risks,
        strategies=strategies, depth=depth + 1,
    ))
    return result


def _overlap_risks(blocks: list[Block], config: ReadingOrderConfig) -> list[str]:
    """Report severe text-region overlap without rejecting intentional figures."""
    text_blocks = [block for block in blocks if not _is_figure(block, config)]
    risks = []
    for index, first in enumerate(text_blocks):
        first_area = (first.bbox[2] - first.bbox[0]) * (first.bbox[3] - first.bbox[1])
        if first_area <= 0:
            continue
        for second in text_blocks[index + 1:]:
            second_area = (second.bbox[2] - second.bbox[0]) * (second.bbox[3] - second.bbox[1])
            if second_area <= 0:
                continue
            width = _horizontal_overlap(first, second)
            height = max(0.0, min(first.bbox[3], second.bbox[3]) - max(first.bbox[1], second.bbox[1]))
            ratio = width * height / min(first_area, second_area)
            if ratio > config.overlap_ratio_threshold:
                risks.append(f"overlapping_text_blocks:{first.block_id},{second.block_id}")
    return risks


def plan_reading_order(
    page: Page,
    config: ReadingOrderConfig | None = None,
) -> ReadingOrderResult:
    """Return a complete deterministic ordering plan without mutating the page."""
    config = config or ReadingOrderConfig()
    original = {block.block_id: index for index, block in enumerate(page.blocks)}
    if len(original) != len(page.blocks):
        raise ValueError(f"duplicate block id on page {page.page_id}")
    if not config.enabled:
        ordered = sorted(page.blocks, key=lambda block: original[block.block_id])
        return ReadingOrderResult(
            status="disabled", source="original_layout_order",
            ordered_block_ids=[block.block_id for block in ordered],
            main_block_ids=[block.block_id for block in ordered if not _is_auxiliary(block, config)],
            auxiliary_block_ids=[block.block_id for block in ordered if _is_auxiliary(block, config)],
        )

    main = [block for block in page.blocks if not _is_auxiliary(block, config)]
    auxiliary = [block for block in page.blocks if _is_auxiliary(block, config)]
    risks: list[str] = []
    strategies: set[str] = set()
    invalid = [block.block_id for block in page.blocks if not _valid_geometry(block)]
    if invalid:
        risks.append("invalid_or_empty_block_geometry:" + ",".join(invalid))
    risks.extend(_overlap_risks([block for block in page.blocks if _valid_geometry(block)], config))

    if invalid:
        ordered_main = _fallback(main, original)
        ordered_auxiliary = _fallback(auxiliary, original)
        risks.append("coordinate_fallback_after_invalid_geometry")
    else:
        ordered_main = _order_blocks(
            main, config=config, original=original, risks=risks,
            strategies=strategies, depth=0,
        )
        ordered_auxiliary = _order_blocks(
            auxiliary, config=config, original=original, risks=risks,
            strategies=strategies, depth=0,
        )
    ordered = ordered_main + ordered_auxiliary
    return ReadingOrderResult(
        status="risky" if risks else "confirmed",
        source="deterministic_xy_cut_v1",
        ordered_block_ids=[block.block_id for block in ordered],
        main_block_ids=[block.block_id for block in ordered_main],
        auxiliary_block_ids=[block.block_id for block in ordered_auxiliary],
        risks=list(dict.fromkeys(risks)),
        strategies=sorted(strategies),
    )


def restore_page_reading_order(
    page: Page,
    config: ReadingOrderConfig | None = None,
) -> ReadingOrderResult:
    """Apply a complete ordering plan while preserving original region provenance."""
    result = plan_reading_order(page, config)
    by_id = {block.block_id: block for block in page.blocks}
    for order_index, block_id in enumerate(result.ordered_block_ids):
        block = by_id[block_id]
        block.metadata.setdefault("layout_reading_order", block.reading_order)
        block.reading_order = order_index
    page.metadata["reading_order"] = result.to_dict()
    return result
