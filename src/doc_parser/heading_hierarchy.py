"""Document-level heading hierarchy recovery for MinerU middle JSON.

Explicit numbering determines a heading's level. Numbered headings also serve as
style anchors for unnumbered headings. Visual style never overrides explicit
numbering; conflicts are retained as auditable warnings.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    merge_para_with_text,
    union_make,
)
from mineru.utils.enum_class import MakeMode

TITLE_TYPES = {"title", "doc_title", "paragraph_title"}
BODY_TYPES = {"text", "abstract", "list", "index", "ref_text"}
MAX_LEVEL = 6

_DECIMAL = re.compile(r"^\s*(\d+(?:[.]\d+)*)(?:[、．)]|[.](?!\d)|\s)+(?=\S)")
_CHAPTER = re.compile(r"^\s*第([一二三四五六七八九十百千万零〇两\d]+)(章|篇|部)\s*")
_SECTION = re.compile(r"^\s*第([一二三四五六七八九十百千万零〇两\d]+)(节)\s*")
_CHINESE = re.compile(r"^\s*([一二三四五六七八九十百千万零〇两]+)[、．.]\s*")
_PAREN_CHINESE = re.compile(r"^\s*[（(]([一二三四五六七八九十百千万零〇两]+)[）)]\s*")
_PAREN_NUMBER = re.compile(r"^\s*[（(](\d+)[）)]\s*")
_CIRCLED = re.compile(r"^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s*")


@dataclass
class Heading:
    heading_id: str
    page_idx: int
    order: int
    block: dict[str, Any]
    text: str
    original_level: int
    numbering: dict[str, Any] | None
    line_height: float | None
    body_line_height: float | None
    relative_line_height: float | None
    centered_score: float | None
    left_indent_ratio: float | None
    line_count: int
    font_height_source: str = "ocr_line_bbox"
    ocr_line_height: float | None = None
    layout_block_height: float | None = None
    resolved_level: int | None = None
    method: str = "unresolved"
    confidence: float = 0.0
    style_cluster_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    parent_heading_id: str | None = None
    resolved_role: str = "section_heading"
    role_method: str = "layout_title_default"
    role_confidence: float = 0.5
    role_scores: dict[str, float] = field(default_factory=dict)
    role_evidence: list[str] = field(default_factory=list)
    target_block_id: str | None = None


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 4
        and all(isinstance(x, (int, float)) for x in value[:4])
        and value[2] >= value[0]
        and value[3] >= value[1]
    )


def _line_heights(block: dict[str, Any]) -> list[float]:
    heights: list[float] = []
    for line in block.get("lines", []):
        bbox = line.get("bbox")
        if _valid_bbox(bbox) and bbox[3] > bbox[1]:
            heights.append(float(bbox[3] - bbox[1]))
    return heights


def _median_line_height(block: dict[str, Any]) -> float | None:
    heights = _line_heights(block)
    return statistics.median(heights) if heights else None


def _estimate_heading_height(block: dict[str, Any]) -> tuple[float | None, str, float | None, float | None]:
    """Estimate visual font size without trusting noisy OCR boxes first.

    For single-line headings the layout block height is a better style signal
    than an OCR box, which can absorb underlines or colored decorations. For
    wrapped headings, the per-line OCR median avoids treating total block height
    as font size.
    """
    ocr_height = _median_line_height(block)
    bbox = block.get("bbox")
    layout_height = float(bbox[3] - bbox[1]) if _valid_bbox(bbox) and bbox[3] > bbox[1] else None
    line_count = len(_line_heights(block))
    if line_count == 1 and layout_height:
        return layout_height, "layout_block_bbox", ocr_height, layout_height
    if ocr_height:
        return ocr_height, "ocr_line_bbox", ocr_height, layout_height
    if layout_height:
        return layout_height, "layout_block_bbox_fallback", ocr_height, layout_height
    return None, "unavailable", ocr_height, layout_height


def _page_body_baseline(page: dict[str, Any]) -> float | None:
    heights: list[float] = []
    for block in page.get("para_blocks", []):
        if block.get("type") in BODY_TYPES:
            heights.extend(_line_heights(block))
    return statistics.median(heights) if heights else None


def _document_body_baseline(pages: list[dict[str, Any]]) -> float | None:
    values = [value for page in pages if (value := _page_body_baseline(page))]
    return statistics.median(values) if values else None


def parse_numbering(text: str) -> dict[str, Any] | None:
    """Parse common section numbering from an already-detected heading."""
    if match := _CHAPTER.match(text):
        return {"scheme": "chapter", "raw": match.group(0).strip(), "depth": 1}
    if match := _SECTION.match(text):
        return {"scheme": "section", "raw": match.group(0).strip(), "depth": 2}
    if match := _CHINESE.match(text):
        return {"scheme": "chinese", "raw": match.group(0).strip(), "depth": 1}
    if match := _PAREN_CHINESE.match(text):
        return {"scheme": "parenthesized_chinese", "raw": match.group(0).strip(), "depth": 2}
    if match := _PAREN_NUMBER.match(text):
        return {"scheme": "parenthesized_number", "raw": match.group(0).strip(), "depth": 3}
    if match := _CIRCLED.match(text):
        return {"scheme": "circled_number", "raw": match.group(0).strip(), "depth": 4}
    if match := _DECIMAL.match(text):
        raw = match.group(1)
        return {"scheme": "decimal", "raw": raw, "depth": len(raw.split("."))}
    return None


def _safe_level(value: Any, default: int = 2) -> int:
    return value if isinstance(value, int) and 1 <= value <= MAX_LEVEL else default


def collect_headings(middle_json: dict[str, Any]) -> tuple[list[Heading], dict[int, float | None]]:
    pages = middle_json.get("pdf_info")
    if not isinstance(pages, list):
        raise ValueError("middle JSON must contain a pdf_info list")

    global_baseline = _document_body_baseline(pages)
    page_baselines: dict[int, float | None] = {}
    headings: list[Heading] = []
    sequence = 0
    for page_pos, page in enumerate(pages):
        page_idx = int(page.get("page_idx", page_pos))
        local_baseline = _page_body_baseline(page)
        baseline = local_baseline or global_baseline
        page_baselines[page_idx] = baseline
        page_size = page.get("page_size") or []
        page_width = float(page_size[0]) if page_size and page_size[0] else None
        for order, block in enumerate(page.get("para_blocks", [])):
            if block.get("type") not in TITLE_TYPES:
                continue
            sequence += 1
            text = merge_para_with_text(block).strip()
            bbox = block.get("bbox")
            line_height, height_source, ocr_height, layout_height = _estimate_heading_height(block)
            relative = line_height / baseline if line_height and baseline else None
            centered = None
            indent = None
            if page_width and _valid_bbox(bbox):
                block_center = (bbox[0] + bbox[2]) / 2
                centered = min(1.0, abs(block_center - page_width / 2) / (page_width / 2))
                indent = max(0.0, min(1.0, float(bbox[0]) / page_width))
            headings.append(
                Heading(
                    heading_id=f"p{page_idx}_h{sequence}",
                    page_idx=page_idx,
                    order=order,
                    block=block,
                    text=text,
                    original_level=_safe_level(block.get("level")),
                    numbering=parse_numbering(text),
                    line_height=line_height,
                    body_line_height=baseline,
                    relative_line_height=relative,
                    centered_score=centered,
                    left_indent_ratio=indent,
                    line_count=len(_line_heights(block)),
                    font_height_source=height_source,
                    ocr_line_height=ocr_height,
                    layout_block_height=layout_height,
                )
            )
    return headings, page_baselines


def _numbering_level(numbering: dict[str, Any], base_level: int) -> int:
    return min(MAX_LEVEL, max(1, base_level + int(numbering["depth"]) - 1))


def _horizontal_overlap_ratio(a: list[float], b: list[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    width = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return overlap / width


def resolve_heading_roles(headings: list[Heading], pages: list[dict[str, Any]]) -> None:
    """Classify object titles before hierarchy recovery using local structure."""
    pages_by_idx = {int(page.get("page_idx", pos)): page for pos, page in enumerate(pages)}
    visual_roles = {"table": "table_title", "image": "figure_title", "chart": "figure_title"}
    for heading in headings:
        if heading.numbering:
            heading.role_scores = {"section_heading": 0.99}
            heading.role_evidence = ["explicit_section_numbering"]
            heading.role_confidence = 0.99
            continue
        blocks = pages_by_idx[heading.page_idx].get("para_blocks", [])
        next_block = blocks[heading.order + 1] if heading.order + 1 < len(blocks) else None
        if not next_block or next_block.get("type") not in visual_roles:
            heading.role_scores = {"section_heading": 0.65}
            heading.role_evidence = ["no_adjacent_visual_object"]
            heading.role_confidence = 0.65
            continue
        title_bbox, object_bbox = heading.block.get("bbox"), next_block.get("bbox")
        normalized_gap = None
        overlap = None
        if _valid_bbox(title_bbox) and _valid_bbox(object_bbox):
            normalized_gap = max(0.0, object_bbox[1] - title_bbox[3]) / max(1.0, heading.body_line_height or 1.0)
            overlap = _horizontal_overlap_ratio(title_bbox, object_bbox)
        object_role = visual_roles[next_block["type"]]
        object_score = 0.45
        section_score = 0.40
        evidence = [f"next_block_is_{next_block['type']}", "no_intervening_block"]
        if normalized_gap is not None and normalized_gap <= 1.5:
            object_score += 0.25
            evidence.append(f"normalized_gap={normalized_gap:.3f}")
        if overlap is not None and overlap >= 0.75:
            object_score += 0.15
            evidence.append(f"horizontal_overlap={overlap:.3f}")
        # A heading followed only by one visual object before the next title has
        # object-title scope; section headings with explicit numbering were handled above.
        after_object = blocks[heading.order + 2] if heading.order + 2 < len(blocks) else None
        if after_object is None or after_object.get("type") in TITLE_TYPES:
            object_score += 0.15
            evidence.append("scope_contains_only_visual_object")
        scores = {"section_heading": min(1.0, section_score), object_role: min(1.0, object_score)}
        heading.role_scores = scores
        if object_score >= 0.75 and object_score - section_score >= 0.15:
            heading.resolved_role = object_role
            heading.role_method = "adjacent_visual_object"
            heading.role_confidence = round(object_score, 3)
            heading.role_evidence = evidence
            heading.target_block_id = f"p{heading.page_idx}_b{heading.order + 1}"
        else:
            heading.role_confidence = round(section_score, 3)
            heading.role_evidence = evidence + ["role_score_not_decisive"]


def _feature_distance(heading: Heading, profile: dict[str, Any]) -> float | None:
    distances: list[tuple[float, float]] = []
    for value, key, tolerance, weight in (
        (heading.relative_line_height, "relative_line_height_median", 0.18, 0.90),
        # Column position varies across a page, so geometry is only weak evidence.
        (heading.left_indent_ratio, "left_indent_ratio_median", 0.30, 0.05),
        (heading.centered_score, "centered_score_median", 0.30, 0.05),
    ):
        target = profile.get(key)
        if value is not None and target is not None:
            distances.append((abs(value - target) / tolerance, weight))
    if not distances:
        return None
    total_weight = sum(weight for _, weight in distances)
    return sum(distance * weight for distance, weight in distances) / total_weight


def _median(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(valid) if valid else None


def _build_style_registry(headings: list[Heading]) -> list[dict[str, Any]]:
    anchors: dict[int, list[Heading]] = defaultdict(list)
    for heading in headings:
        if heading.numbering and heading.resolved_level is not None:
            anchors[heading.resolved_level].append(heading)

    registry: list[dict[str, Any]] = []
    for level in sorted(anchors):
        items = anchors[level]
        profile = {
            "cluster_id": f"numbered_level_{level}",
            "resolved_level": level,
            "relative_line_height_median": _median([x.relative_line_height for x in items]),
            "line_height_median": _median([x.line_height for x in items]),
            "left_indent_ratio_median": _median([x.left_indent_ratio for x in items]),
            "centered_score_median": _median([x.centered_score for x in items]),
            "anchor_count": len(items),
            "sample_heading_ids": [x.heading_id for x in items[:5]],
            "confidence": round(min(0.99, 0.82 + 0.03 * len(items)), 3),
        }
        registry.append(profile)
    return registry


def _apply_unanchored_style_clustering(headings: list[Heading], base_level: int) -> list[dict[str, Any]]:
    """Infer two visual levels when an unnumbered page has a clear size gap."""
    profiles: list[dict[str, Any]] = []
    by_page: dict[int, list[Heading]] = defaultdict(list)
    for heading in headings:
        if heading.resolved_role == "section_heading" and heading.resolved_level is None and heading.relative_line_height:
            by_page[heading.page_idx].append(heading)
    for page_idx, items in by_page.items():
        values = sorted({round(float(x.relative_line_height), 4) for x in items}, reverse=True)
        if len(values) < 2:
            continue
        gaps = [(values[i] / values[i + 1], i) for i in range(len(values) - 1) if values[i + 1] > 0]
        ratio, split = max(gaps)
        if ratio < 1.20:
            continue
        threshold = (values[split] + values[split + 1]) / 2
        upper = [x for x in items if float(x.relative_line_height) > threshold]
        lower = [x for x in items if float(x.relative_line_height) <= threshold]
        # Avoid turning several progressively different font sizes into deep trees:
        # this fallback deliberately creates only two adjacent levels.
        if not upper or not lower:
            continue
        for heading in upper:
            heading.resolved_level = base_level
            heading.method = "page_style_clustering"
            heading.style_cluster_id = f"page_{page_idx}_prominent"
            heading.confidence = round(min(0.88, 0.58 + (ratio - 1.20) * 0.35), 3)
        for heading in lower:
            heading.resolved_level = min(MAX_LEVEL, base_level + 1)
            heading.method = "page_style_clustering"
            heading.style_cluster_id = f"page_{page_idx}_secondary"
            heading.confidence = round(min(0.82, 0.54 + (ratio - 1.20) * 0.30), 3)
        profiles.extend([
            {"cluster_id": f"page_{page_idx}_prominent", "scope": "page", "page_idx": page_idx, "resolved_level": base_level, "relative_line_height_median": _median([x.relative_line_height for x in upper]), "sample_heading_ids": [x.heading_id for x in upper], "confidence": _median([x.confidence for x in upper])},
            {"cluster_id": f"page_{page_idx}_secondary", "scope": "page", "page_idx": page_idx, "resolved_level": min(MAX_LEVEL, base_level + 1), "relative_line_height_median": _median([x.relative_line_height for x in lower]), "sample_heading_ids": [x.heading_id for x in lower], "confidence": _median([x.confidence for x in lower])},
        ])
    return profiles


def resolve_headings(headings: list[Heading], base_level: int = 2) -> list[dict[str, Any]]:
    """Resolve headings using numbering, anchored style, then page clustering."""
    for heading in headings:
        if heading.resolved_role == "section_heading" and heading.numbering:
            heading.resolved_level = _numbering_level(heading.numbering, base_level)
            heading.method = "numbering"
            heading.confidence = 0.99

    registry = _build_style_registry(headings)
    for heading in headings:
        if heading.resolved_role != "section_heading" or heading.resolved_level is not None:
            continue
        matches = []
        for profile in registry:
            distance = _feature_distance(heading, profile)
            if distance is not None:
                matches.append((distance, profile))
        matches.sort(key=lambda item: item[0])
        if matches and matches[0][0] <= 1.0:
            distance, profile = matches[0]
            ambiguous = len(matches) > 1 and matches[1][0] - distance < 0.20
            if not ambiguous:
                heading.resolved_level = int(profile["resolved_level"])
                heading.style_cluster_id = str(profile["cluster_id"])
                heading.method = "document_style_mapping"
                heading.confidence = round(max(0.55, 0.9 - 0.25 * distance), 3)
            else:
                heading.warnings.append("ambiguous_style_match")

    registry.extend(_apply_unanchored_style_clustering(headings, base_level))
    for heading in headings:
        if heading.resolved_role != "section_heading":
            heading.resolved_level = 0
            heading.method = "object_title_role"
            heading.confidence = heading.role_confidence
        elif heading.resolved_level is None:
            heading.resolved_level = heading.original_level
            heading.method = "mineru_default"
            heading.confidence = 0.35
            heading.warnings.append("insufficient_heading_evidence")

    # Style is diagnostic for numbered headings and never overrides numbering.
    for heading in headings:
        if not heading.numbering:
            continue
        own_profile = next(
            (profile for profile in registry if profile["resolved_level"] == heading.resolved_level),
            None,
        )
        if own_profile:
            heading.style_cluster_id = str(own_profile["cluster_id"])
            own_distance = _feature_distance(heading, own_profile)
            other_distances = [
                (distance, profile)
                for profile in registry
                if profile["resolved_level"] != heading.resolved_level
                and (distance := _feature_distance(heading, profile)) is not None
            ]
            other_distances.sort(key=lambda item: item[0])
            if (
                own_distance is not None
                and other_distances
                and other_distances[0][0] + 0.20 < own_distance
            ):
                heading.warnings.append("numbering_style_level_conflict")
                heading.confidence = 0.94

    return registry


def build_heading_tree(headings: list[Heading]) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    stack: list[tuple[int, Heading, dict[str, Any]]] = []
    for heading in headings:
        if heading.resolved_role != "section_heading":
            continue
        level = int(heading.resolved_level or heading.original_level)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack and level > stack[-1][0] + 1:
            heading.warnings.append("heading_level_jump")
        node = {
            "heading_id": heading.heading_id,
            "text": heading.text,
            "level": level,
            "page_idx": heading.page_idx,
            "children": [],
        }
        if stack:
            heading.parent_heading_id = stack[-1][1].heading_id
            stack[-1][2]["children"].append(node)
        else:
            roots.append(node)
        stack.append((level, heading, node))
    return roots


def _heading_record(heading: Heading) -> dict[str, Any]:
    return {
        "heading_id": heading.heading_id,
        "page_idx": heading.page_idx,
        "order": heading.order,
        "text": heading.text,
        "original_level": heading.original_level,
        "numbering": heading.numbering,
        "role_resolution": {
            "resolved_role": heading.resolved_role,
            "method": heading.role_method,
            "confidence": heading.role_confidence,
            "scores": heading.role_scores,
            "evidence": heading.role_evidence,
            "target_block_id": heading.target_block_id,
        },
        "visual": {
            "line_height": heading.line_height,
            "body_line_height": heading.body_line_height,
            "relative_line_height": heading.relative_line_height,
            "centered_score": heading.centered_score,
            "left_indent_ratio": heading.left_indent_ratio,
            "line_count": heading.line_count,
            "font_height_source": heading.font_height_source,
            "ocr_line_height": heading.ocr_line_height,
            "layout_block_height": heading.layout_block_height,
            "font_evidence_source": "visual_geometry",
        },
        "resolution": {
            "resolved_level": heading.resolved_level,
            "method": heading.method,
            "style_cluster_id": heading.style_cluster_id,
            "parent_heading_id": heading.parent_heading_id,
            "confidence": heading.confidence,
            "warnings": heading.warnings,
            "status": "final",
        },
    }


def enhance_middle_json(
    middle_json: dict[str, Any], *, base_level: int = 2
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return enhanced JSON, inventory, style registry, and heading tree."""
    enhanced = copy.deepcopy(middle_json)
    headings, page_baselines = collect_headings(enhanced)
    resolve_heading_roles(headings, enhanced["pdf_info"])
    registry = resolve_headings(headings, base_level=base_level)
    tree = build_heading_tree(headings)

    for heading in headings:
        heading.block["level"] = heading.resolved_level
        heading.block["heading_role"] = {
            "resolved_role": heading.resolved_role,
            "method": heading.role_method,
            "confidence": heading.role_confidence,
            "scores": heading.role_scores,
            "evidence": heading.role_evidence,
            "target_block_id": heading.target_block_id,
        }
        heading.block["heading_resolution"] = {
            "original_level": heading.original_level,
            "resolved_level": heading.resolved_level,
            "method": heading.method,
            "numbering": heading.numbering,
            "style_cluster_id": heading.style_cluster_id,
            "parent_heading_id": heading.parent_heading_id,
            "confidence": heading.confidence,
            "warnings": heading.warnings,
            "status": "final",
        }

    enhanced["_heading_hierarchy"] = {
        "version": "1.0",
        "strategy": "numbering_and_style",
        "base_level": base_level,
        "heading_count": len(headings),
        "numbered_heading_count": sum(bool(x.numbering) for x in headings),
    }
    inventory = {
        "strategy": "numbering_and_style",
        "base_level": base_level,
        "page_body_line_heights": page_baselines,
        "headings": [_heading_record(x) for x in headings],
    }
    return enhanced, inventory, registry, tree


def generate_outputs(middle_json: dict[str, Any], image_path: str = "images") -> tuple[Any, Any, str]:
    pdf_info = middle_json["pdf_info"]
    content = union_make(pdf_info, MakeMode.CONTENT_LIST, image_path)
    content_v2 = union_make(pdf_info, MakeMode.CONTENT_LIST_V2, image_path)
    markdown = union_make(pdf_info, MakeMode.MM_MD, image_path)
    return content, content_v2, markdown


def process_file(input_path: Path, output_dir: Path, base_level: int = 2) -> dict[str, Path]:
    middle = json.loads(input_path.read_text("utf-8"))
    enhanced, inventory, registry, tree = enhance_middle_json(middle, base_level=base_level)
    content, content_v2, markdown = generate_outputs(enhanced)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.name.removesuffix("_middle.json").removesuffix(".json")
    outputs = {
        "middle": output_dir / f"{stem}_middle_heading_enhanced.json",
        "inventory": output_dir / f"{stem}_heading_inventory.json",
        "registry": output_dir / f"{stem}_heading_style_registry.json",
        "tree": output_dir / f"{stem}_heading_tree.json",
        "content": output_dir / f"{stem}_content_list_heading_enhanced.json",
        "content_v2": output_dir / f"{stem}_content_list_v2_heading_enhanced.json",
        "markdown": output_dir / f"{stem}_heading_enhanced.md",
    }
    payloads = {
        "middle": enhanced,
        "inventory": inventory,
        "registry": registry,
        "tree": tree,
        "content": content,
        "content_v2": content_v2,
    }
    for key, payload in payloads.items():
        outputs[key].write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    outputs["markdown"].write_text(markdown + ("\n" if markdown else ""), "utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover document-level heading hierarchy from MinerU middle JSON.")
    parser.add_argument("input", type=Path, help="MinerU *_middle.json input")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--base-level", type=int, default=2, choices=range(1, MAX_LEVEL + 1))
    args = parser.parse_args()
    outputs = process_file(args.input, args.output, args.base_level)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
