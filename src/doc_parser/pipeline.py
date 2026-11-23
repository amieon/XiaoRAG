"""公开模型区域管线：版面检测、裁图、路由、识别与 DocumentIR 组装。

模型适配器只提供 detect_layout、recognize_text、recognize_vlm 三个操作。
纯路由和 IR 组装可以用小型假后端做无模型测试。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

from .document_ir import Block, DocumentIR, Page, SourceRef
from .routing import (
    GateConfig,
    VisualGate,
    attach_native_text,
    detector_fields,
    page_contexts,
    route_visual_groups,
)


PIPELINE_VERSION = "document-ir-v2"


class Backend(Protocol):
    """区域管线使用的全部模型接口。"""

    def detect_layout(self, image: Any) -> Mapping[str, Any]: ...

    def recognize_text(self, images: list[Any]) -> list[dict[str, Any]]: ...

    def recognize_vlm(self, groups: dict[tuple[int, int], dict]) -> None: ...


@dataclass(frozen=True)
class ParserConfig:
    """可审计的识别、路由与并行配置。"""

    gate: GateConfig = field(default_factory=GateConfig)
    crop_context_chars: int = 300
    min_layout_box_side: float = 6.0
    native_min_coverage: float = 0.9
    max_new_tokens: int = 4096
    save_region_images: bool = True
    drop_layout_labels: tuple[str, ...] = ("reference",)
    figure_labels: tuple[str, ...] = (
        "image", "header_image", "footer_image", "figure",
    )
    markdown_ignore_labels: tuple[str, ...] = (
        "number", "footnote", "header", "header_image", "footer",
        "footer_image", "aside_text",
    )
    ocr_pixel_limits: tuple[int, int] = (112896, 1003520)
    table_pixel_limits: tuple[int, int] = (112896, 1003520)
    formula_pixel_limits: tuple[int, int] = (112896, 1003520)
    chart_pixel_limits: tuple[int, int] = (112896, 1003520)
    seal_pixel_limits: tuple[int, int] = (112896, 1003520)
    # 0 表示自动：CPU 按任务数取小并发，GPU 逐个加载并按空闲显存准入。
    cpu_workers: int = 0
    gpu_workers: int = 0
    max_gpu_workers: int = 4
    gpu_free_margin_mb: int = 4096
    gpu_worker_floor_mb: int = 2048

    def __post_init__(self) -> None:
        if (not isinstance(self.crop_context_chars, int) or
                isinstance(self.crop_context_chars, bool) or self.crop_context_chars < 1):
            raise ValueError("crop_context_chars must be a positive integer")
        if not math.isfinite(self.min_layout_box_side) or self.min_layout_box_side < 1:
            raise ValueError("min_layout_box_side must be at least 1 pixel")
        if not 0 <= self.native_min_coverage <= 1:
            raise ValueError("native_min_coverage must be in [0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        for name in ("cpu_workers", "gpu_workers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer; 0 means auto")
        if not isinstance(self.max_gpu_workers, int) or self.max_gpu_workers < 1:
            raise ValueError("max_gpu_workers must be a positive integer")
        if not isinstance(self.gpu_free_margin_mb, int) or self.gpu_free_margin_mb < 0:
            raise ValueError("gpu_free_margin_mb must be a non-negative integer")
        if not isinstance(self.gpu_worker_floor_mb, int) or self.gpu_worker_floor_mb < 1:
            raise ValueError("gpu_worker_floor_mb must be a positive integer")
        for name in (
            "ocr_pixel_limits", "table_pixel_limits", "formula_pixel_limits",
            "chart_pixel_limits", "seal_pixel_limits",
        ):
            value = getattr(self, name)
            if len(value) != 2 or not 0 < value[0] <= value[1]:
                raise ValueError(f"{name} must be (positive_min, larger_max)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParserConfig":
        data = dict(value)
        if "gate" in data:
            data["gate"] = GateConfig(**dict(data["gate"]))
        # 旧配置中的 reading_order 对应已删除的 XY-Cut，不再静默恢复坐标排序。
        data.pop("reading_order", None)
        for key in (
            "drop_layout_labels", "figure_labels", "markdown_ignore_labels",
            "ocr_pixel_limits", "table_pixel_limits", "formula_pixel_limits",
            "chart_pixel_limits", "seal_pixel_limits",
        ):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "ParserConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class PageRegion:
    """One cropped layout region and its pre-recognition evidence."""

    index: int
    label: str
    bbox: list[float]
    image: Any
    polygon: list[list[float]] | None
    query: str | None
    pixel_key: tuple[int, int] | None
    metadata: dict[str, Any]


@dataclass
class PageParseResult:
    page: Page
    records: list[dict[str, Any]]
    markdown: str
    layout: dict[str, Any]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    return value


def public_result_dict(result: Any) -> dict[str, Any]:
    """Normalize public PaddleX result objects without importing their classes."""
    if isinstance(result, Mapping):
        data = dict(result)
    else:
        data = getattr(result, "json", None)
        if callable(data):
            data = data()
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, Mapping):
            try:
                data = dict(result)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unsupported public model result: {type(result).__name__}") from exc
        else:
            data = dict(data)
    while "res" in data and not {"boxes", "result"}.intersection(data):
        nested = data["res"]
        if not isinstance(nested, Mapping):
            break
        data = dict(nested)
    return data


def prompt_for_label(label: str) -> str | None:
    normalized = label.lower()
    if normalized == "table":
        return "Table Recognition:"
    if normalized == "chart":
        return "Chart Recognition:"
    if "formula" in normalized and normalized != "formula_number":
        return "Formula Recognition:"
    if normalized == "spotting":
        return "Spotting:"
    if normalized == "seal":
        return "Seal Recognition:"
    return "OCR:"


def _pixel_limits(label: str, config: ParserConfig) -> tuple[int, int]:
    normalized = label.lower()
    if normalized == "table":
        return config.table_pixel_limits
    if "formula" in normalized:
        return config.formula_pixel_limits
    if normalized == "chart":
        return config.chart_pixel_limits
    if normalized == "seal":
        return config.seal_pixel_limits
    return config.ocr_pixel_limits


def _block_type(label: str) -> str:
    normalized = label.lower()
    aliases = {
        "text": "paragraph", "title": "section_heading",
        "doc_title": "document_title", "paragraph_title": "section_heading",
        "table": "table", "image": "figure", "figure": "figure",
        "chart": "figure", "seal": "figure", "display_formula": "formula",
        "inline_formula": "formula", "equation": "formula",
    }
    return aliases.get(normalized, normalized or "unknown")


def _crop_image(image: Any, bbox: list[float], polygon: Any = None) -> Any:
    """Crop with NumPy/Pillow only; polygon masking does not require OpenCV."""
    import numpy as np
    from PIL import Image, ImageDraw

    array = np.asarray(image)
    height, width = array.shape[:2]
    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty crop for bbox {bbox}")
    crop = array[y0:y1, x0:x1].copy()
    if polygon:
        points = [(float(point[0]) - x0, float(point[1]) - y0) for point in polygon]
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        keep = np.asarray(mask, dtype=bool)
        if crop.ndim == 3:
            keep = keep[:, :, None]
        crop = np.where(keep, crop, 255).astype(crop.dtype)
    return crop


def build_regions(
    image: Any,
    layout: Mapping[str, Any],
    *,
    native_blocks: list[dict] | None = None,
    config: ParserConfig = ParserConfig(),
) -> list[PageRegion]:
    """把公开版面结果转换成有稳定次序的区域裁图对象。"""
    height, width = image.shape[:2]
    detectors: list[dict] = []
    blocks: list[dict] = []
    for raw in layout.get("boxes", []) or []:
        box = public_result_dict(raw)
        label = str(box.get("label", box.get("cls_id", "unknown"))).lower()
        if label in config.drop_layout_labels:
            continue
        coordinate = box.get("coordinate", box.get("box"))
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 4:
            continue
        try:
            bbox = [float(value) for value in coordinate]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in bbox):
            continue
        x0, y0, x1, y1 = bbox
        clipped = [
            max(0.0, min(float(width), x0)), max(0.0, min(float(height), y0)),
            max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
        ]
        if (clipped[2] - clipped[0] < config.min_layout_box_side or
                clipped[3] - clipped[1] < config.min_layout_box_side):
            continue
        polygon = _plain(box.get("polygon_points")) if box.get("polygon_points") is not None else None
        crop = _crop_image(image, clipped, polygon)
        detector = detector_fields({**box, "coordinate": clipped})
        detectors.append(detector)
        block = {
            "img": crop, "box": clipped, "label": label,
            "polygon_points": polygon,
        }
        for key in ("score", "angle", "vertical", "math_risk"):
            if key in detector:
                block[key] = detector[key]
        blocks.append(block)

    contexts = page_contexts(blocks, detectors)
    attach_native_text(contexts, native_blocks, config.native_min_coverage)
    regions: list[PageRegion] = []
    for index, (block, metadata) in enumerate(zip(blocks, contexts)):
        label = block["label"]
        query = None if label in config.figure_labels else prompt_for_label(label)
        regions.append(PageRegion(
            index=index,
            label=label,
            bbox=block["box"],
            image=block["img"],
            polygon=block["polygon_points"],
            query=query,
            pixel_key=_pixel_limits(label, config) if query else None,
            metadata=metadata,
        ))
    return regions


def group_regions(
    regions: list[PageRegion], *, page_index: int,
) -> dict[tuple[int, int], dict]:
    """按像素限制分组，同时保留每个区域的页号与区域号。"""
    groups: dict[tuple[int, int], dict] = {}
    for region in regions:
        if region.query is None or region.pixel_key is None:
            continue
        group = groups.setdefault(region.pixel_key, {
            "images": [], "queries": [], "figure_token_maps": [],
            "vlm_block_ids": [], "region_metadata": [], "curr_vlm_block_idx": 0,
        })
        group["images"].append(region.image)
        group["queries"].append(region.query)
        group["figure_token_maps"].append({})
        group["vlm_block_ids"].append((page_index, region.index))
        group["region_metadata"].append(region.metadata)
    return groups


def _confidence(
    output: Mapping[str, Any], engine: str, metadata: Mapping[str, Any] | None = None,
) -> float | None:
    if engine == "native_text":
        return 1.0
    if engine == "layout_only" and metadata and metadata.get("score") is not None:
        try:
            return float(metadata["score"])
        except (TypeError, ValueError):
            return None
    scores = output.get("line_scores")
    if isinstance(scores, list) and scores:
        try:
            values = [float(value) for value in scores]
            return min(values) if values else None
        except (TypeError, ValueError):
            return None
    return None


def _layout_only_record(region: PageRegion, page_index: int) -> dict[str, Any]:
    return {
        "page_id": page_index,
        "block_id": [page_index, region.index],
        "bbox": region.bbox,
        "query": None,
        "engine": "layout_only",
        "engine_name": "layout_detection",
        "route_reason": ["non_text_layout_label"],
        "risk_score": None,
        "shape": list(region.image.shape),
        "pixel_limits": None,
        "result": "",
        "line_scores": None,
        "gate": None,
        "gate_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "layout": region.metadata,
    }


def _crop_context(
    regions: list[PageRegion],
    texts: list[str],
    position: int,
    *,
    page_id: str,
    max_chars: int,
) -> dict[str, Any]:
    """保存相邻 crop 的识别原文窗口；前文取后缀，后文取前缀。"""
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


def render_markdown(page: Page, config: ParserConfig) -> str:
    """Deterministic human-facing export; DocumentIR remains the primary output."""
    chunks: list[str] = []
    for block in sorted(page.blocks, key=lambda item: item.reading_order or 0):
        raw_label = str(block.metadata.get("raw_label", block.block_type)).lower()
        if raw_label in config.markdown_ignore_labels:
            continue
        text = block.text.strip()
        if block.block_type == "figure":
            asset = block.metadata.get("asset_path")
            if asset:
                chunks.append(f"![{raw_label}]({asset})")
            continue
        if not text:
            continue
        if block.block_type == "document_title":
            chunks.append(f"# {text}")
        elif block.block_type == "section_heading":
            chunks.append(f"## {text}")
        elif block.block_type == "formula" and "$" not in text:
            chunks.append(f"$$\n{text}\n$$")
        elif block.block_type == "table" and "<table" not in text.lower():
            # Preserve OTSL or other raw structures instead of pretending they are HTML.
            import html
            chunks.append(f"<pre>{html.escape(text)}</pre>")
        else:
            chunks.append(text)
    return "\n\n".join(chunks) + ("\n" if chunks else "")


class RegionPipeline:
    """Layout -> crop -> route -> recognize -> Page IR, with no hidden hooks."""

    def __init__(
        self,
        backend: Backend,
        config: ParserConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or ParserConfig()
        self.gate = VisualGate(self.config.gate)

    def parse_image(
        self,
        image_path: str | Path,
        *,
        document_id: str,
        page_no: int,
        source_path: str | None = None,
        image_ref: str | None = None,
        native_blocks: list[dict] | None = None,
        output_dir: str | Path | None = None,
    ) -> PageParseResult:
        import numpy as np
        from PIL import Image

        source = Path(image_path)
        with Image.open(source) as image:
            array = np.asarray(image.convert("RGB"))
        height, width = array.shape[:2]
        layout = public_result_dict(self.backend.detect_layout(array))
        regions = build_regions(
            array, layout, native_blocks=native_blocks, config=self.config,
        )
        page_index = page_no - 1
        groups = group_regions(regions, page_index=page_index)
        records = route_visual_groups(
            groups,
            self.backend.recognize_text,
            self.backend.recognize_vlm,
            gate=self.gate,
        )
        outputs: dict[tuple[int, int], dict] = {}
        for group in groups.values():
            outputs.update({
                tuple(block_id): output or {}
                for block_id, output in zip(group["vlm_block_ids"], group["vlm_results"])
            })
        route_by_id = {
            tuple(record["block_id"]): record for record in records
        }

        output = Path(output_dir) if output_dir is not None else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
        page_id = f"{document_id}_p{page_no:04d}"
        region_texts = [
            str(outputs.get((page_index, region.index), {}).get("result") or "").strip()
            for region in regions
        ]
        blocks: list[Block] = []
        for position, region in enumerate(regions):
            key = (page_index, region.index)
            route = route_by_id.get(key)
            if route is None:
                route = _layout_only_record(region, page_index)
                records.append(route)
            result = outputs.get(key, {})
            context = _crop_context(
                regions, region_texts, position,
                page_id=page_id, max_chars=self.config.crop_context_chars,
            )
            route["context"] = context
            asset_path = None
            if output is not None and self.config.save_region_images:
                asset_name = f"{page_id}_b{region.index:04d}.png"
                (output / "assets").mkdir(parents=True, exist_ok=True)
                Image.fromarray(region.image).save(output / "assets" / asset_name)
                asset_path = f"assets/{asset_name}"
            engine = str(route.get("engine_name") or route.get("engine") or "unknown")
            metadata = {
                "raw_label": region.label,
                "query": region.query,
                "engine": route.get("engine"),
                "route_reason": route.get("route_reason", []),
                "risk_score": route.get("risk_score"),
                "gate": route.get("gate"),
                "line_scores": result.get("line_scores"),
                "line_boxes": _plain(result.get("line_boxes")),
                "native_block_ids": region.metadata.get("native_block_ids", []),
                "layout": _plain(region.metadata),
                "asset_path": asset_path,
                "context": context,
            }
            blocks.append(Block(
                block_id=f"{page_id}_b{region.index:04d}",
                page_id=page_id,
                block_type=_block_type(region.label),
                bbox=region.bbox,
                polygon=region.polygon,
                text=str(result.get("result") or "").strip(),
                reading_order=region.index,
                confidence=_confidence(result, str(route.get("engine")), region.metadata),
                source=SourceRef(
                    engine=engine,
                    raw_page=page_no,
                    raw_block_id=str(region.index),
                    raw_path=source_path,
                    extra={"pipeline": PIPELINE_VERSION, "region_index": region.index},
                ),
                metadata=metadata,
            ))
        page = Page(
            page_id=page_id,
            page_no=page_no,
            width=float(width),
            height=float(height),
            blocks=blocks,
            source=SourceRef(
                engine="public_layout_pipeline",
                raw_page=page_no,
                raw_path=source_path,
                extra={"pipeline": PIPELINE_VERSION},
            ),
            image_ref=image_ref,
            metadata={
                "pipeline": PIPELINE_VERSION,
                "layout_region_count": len(regions),
                "reading_order": {
                    "status": "model_provided",
                    "source": "layout_region_order",
                    "ordered_block_ids": [
                        f"{page_id}_b{region.index:04d}" for region in regions
                    ],
                },
                "recognized_region_count": len(groups) and sum(
                    len(group["images"]) for group in groups.values()
                ),
            },
        )
        # 区域序号就是上游模型顺序；不再按 bbox 做 XY-Cut 二次排序。
        page.validate()
        records.sort(key=lambda item: (item.get("page_id", 0), item.get("block_id", [0, 0])[1]))
        result = PageParseResult(
            page=page,
            records=_plain(records),
            markdown=render_markdown(page, self.config),
            layout=_plain(layout),
        )
        if output is not None:
            write_page_result(output, result, config=self.config)
        return result


def write_page_result(
    output: str | Path,
    result: PageParseResult,
    *,
    config: ParserConfig = ParserConfig(),
) -> None:
    """Write portable page artifacts after assets have been saved."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "layout.json").write_text(
        json.dumps(result.layout, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (root / "page.md").write_text(result.markdown, encoding="utf-8")
    (root / "page_result.json").write_text(
        json.dumps({
            "schema_version": "page-result.v2",
            "pipeline": PIPELINE_VERSION,
            "page": result.page.to_dict() if hasattr(result.page, "to_dict") else asdict(result.page),
            "routes": result.records,
            "config": config.to_dict(),
        }, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    with (root / "routes.jsonl").open("w", encoding="utf-8") as stream:
        for record in result.records:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def empty_document_ir(
    *,
    document_id: str,
    source_path: str | None,
    content_hash: str | None = None,
    mime_type: str | None = None,
) -> DocumentIR:
    return DocumentIR(
        document_id=document_id,
        pages=[],
        source_path=source_path,
        content_hash=content_hash,
        mime_type=mime_type,
        metadata={"pipeline": PIPELINE_VERSION},
    )
