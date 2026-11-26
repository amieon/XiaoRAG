"""单页区域路由。

调用方提供轻量 OCR 与 VLM 两个识别函数；本模块只决定每张裁图的去向，
并保留原始区域顺序和块身份。路由本身不加载任何模型。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping

from .visual_gate import VisualGate, layout_context


@dataclass(frozen=True)
class GateConfig:
    """Engineering threshold for the visual risk gate."""

    risk_threshold: float = 0.45
    max_feature_side: int = 384
    min_visual_contrast: float = 0.15
    min_crop_side: int = 12
    formula_overlap_threshold: float = 0.15
    orientation_angle_threshold: float = 15.0
    math_score_threshold: float = 0.45

    def __post_init__(self):
        if not math.isfinite(self.risk_threshold) or not 0 < self.risk_threshold <= 1:
            raise ValueError("risk_threshold must be in (0, 1]")
        if not isinstance(self.max_feature_side, int) or not 16 <= self.max_feature_side <= 384:
            raise ValueError("max_feature_side must be an integer in [16, 384]")
        if not math.isfinite(self.min_visual_contrast) or not 0 <= self.min_visual_contrast <= 1:
            raise ValueError("min_visual_contrast must be in [0, 1]")
        if not isinstance(self.min_crop_side, int) or self.min_crop_side < 1:
            raise ValueError("min_crop_side must be a positive integer")
        for name in ("formula_overlap_threshold", "math_score_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.orientation_angle_threshold) or not 0 <= self.orientation_angle_threshold <= 90:
            raise ValueError("orientation_angle_threshold must be in [0, 90]")


@dataclass
class Region:
    image: Any
    metadata: dict

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]


def detector_fields(block: Mapping[str, Any]) -> dict:
    """Keep only detector fields that are safe before recognition."""
    result = {key: block[key] for key in ("label", "angle", "vertical", "math_risk", "score") if key in block}
    box = block.get("coordinate", block.get("box"))
    result["box"] = list(box) if box is not None else None
    return result


def page_contexts(blocks, detectors):
    """Build visual-gate metadata from layout detections and overlap."""
    prepared = [detector_fields(block) for block in blocks]
    contexts = layout_context(prepared, detector_blocks=detectors)
    for block, context in zip(prepared, contexts):
        context["image_overlap"] = 0.0
        box = block["box"]
        if not box or len(box) != 4:
            continue
        for detector in detectors:
            other = detector["box"]
            if not other or len(other) != 4:
                continue
            try:
                area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                overlap = max(0.0, min(box[2], other[2]) - max(box[0], other[0])) * max(
                    0.0, min(box[3], other[3]) - max(box[1], other[1]))
                if area > 0:
                    context["image_overlap"] = max(context["image_overlap"], overlap / area)
                if area <= 0 or overlap / area < 0.5:
                    continue
                for key in ("vertical", "math_risk"):
                    if key in detector:
                        value = detector[key]
                        value = value.item() if hasattr(value, "item") else value
                        if value not in (True, False, 0, 1):
                            context["invalid_layout_fields"].append(key)
                        elif value:
                            context[key] = True
                if "angle" in detector:
                    angle = float(detector["angle"])
                    if not math.isfinite(angle):
                        context["invalid_layout_fields"].append("angle")
                    else:
                        context["angle"] = angle
            except (TypeError, ValueError, OverflowError):
                context["invalid_layout_fields"].append("detector_geometry_or_flags")
        context["box"] = box
    return contexts


def _result_text(output):
    if not isinstance(output, dict):
        return ""
    return output.get("result", "")


def _route_record(block_id, query, image, output, decision, metadata, pixel_key,
                  decision_seconds, branch_seconds, engine):
    metadata = dict(metadata) if isinstance(metadata, dict) else None
    return {
        "page_id": block_id[0] if isinstance(block_id, (list, tuple)) and block_id else None,
        "block_id": list(block_id) if isinstance(block_id, (list, tuple)) else block_id,
        "bbox": metadata.get("box") if metadata else None,
        "query": query,
        "engine": engine,
        "engine_name": {"light_ocr": "paddleocr", "vlm": "paddleocr_vl"}.get(engine, engine),
        "route_reason": decision.get("reasons", []),
        "risk_score": decision.get("risk_score"),
        "shape": list(image.shape),
        "pixel_limits": list(pixel_key),
        "result": _result_text(output),
        "line_scores": output.get("line_scores") if isinstance(output, dict) else None,
        "gate": decision,
        "gate_seconds": decision_seconds,
        "elapsed_seconds": decision_seconds + branch_seconds,
        "layout": metadata,
    }


def route_visual_groups(
    groups,
    recognize_text: Callable,
    recognize_vlm: Callable,
    *,
    gate: VisualGate,
):
    """识别前完成 native/light/VLM 分流，并保持区域 ID 与输出一一对应。"""
    records = []
    for pixel_key, group in groups.items():
        images, queries, ids = group["images"], group["queries"], group["vlm_block_ids"]
        metadata = group.get("region_metadata")
        if not len(images) == len(queries) == len(ids):
            raise ValueError("Image/prompt/ID count mismatch")
        if metadata is not None and len(metadata) != len(images):
            raise ValueError("Region metadata count mismatch")

        decisions, decision_seconds = [], []
        for index, (image, query) in enumerate(zip(images, queries)):
            start = time.perf_counter()
            meta = dict(metadata[index]) if metadata and isinstance(metadata[index], dict) else {}
            if query == "OCR:" and meta.get("native_text"):
                decision = {
                    "method": "native_text_bbox_v1", "branch": "native", "risk_score": 0.0,
                    "reasons": ["native_text_bbox_overlap"], "uses_recognized_text": False,
                }
            elif query != "OCR:":
                # 表格、公式、图表仍走 VLM；不在这里暗含另一套表格实验策略。
                decision = {
                    "method": gate.version, "branch": "vlm", "risk_score": 1.0,
                    "reasons": ["non_ocr_prompt"], "uses_recognized_text": False,
                }
            else:
                meta = metadata[index] if metadata is not None else {
                    "invalid_layout_fields": ["missing_region_metadata"]
                }
                decision = gate.assess(Region(image, meta))
            decisions.append(decision)
            decision_seconds.append(time.perf_counter() - start)

        native_ids = [i for i, item in enumerate(decisions) if item["branch"] == "native"]
        light_ids = [i for i, item in enumerate(decisions) if item["branch"] == "light"]
        vlm_ids = [i for i, item in enumerate(decisions) if item["branch"] == "vlm"]
        outputs = [None] * len(images)
        light_seconds = vlm_seconds = 0.0
        for i in native_ids:
            outputs[i] = {"result": metadata[i]["native_text"], "line_scores": []}
        if light_ids:
            start = time.perf_counter()
            values = recognize_text([images[i] for i in light_ids])
            light_seconds = time.perf_counter() - start
            if len(values) != len(light_ids):
                raise ValueError("OCR result count mismatch")
            for i, value in zip(light_ids, values):
                outputs[i] = value
        if vlm_ids:
            subset = {
                key: [group[key][i] for i in vlm_ids]
                for key in ("images", "queries", "vlm_block_ids", "figure_token_maps")
                if key in group
            }
            subset["curr_vlm_block_idx"] = 0
            start = time.perf_counter()
            recognize_vlm({pixel_key: subset})
            vlm_seconds = time.perf_counter() - start
            if len(subset.get("vlm_results", [])) != len(vlm_ids):
                raise ValueError("VLM result count mismatch")
            for i, value in zip(vlm_ids, subset["vlm_results"]):
                outputs[i] = value

        group["vlm_results"] = outputs
        for index, (query, output) in enumerate(zip(queries, outputs)):
            engine = "native_text" if index in native_ids else "light_ocr" if index in light_ids else "vlm"
            branch_seconds = 0.0 if index in native_ids else light_seconds if index in light_ids else vlm_seconds
            records.append(_route_record(
                ids[index], query, images[index], output or {}, decisions[index],
                metadata[index] if metadata is not None else None, pixel_key,
                decision_seconds[index], branch_seconds, engine,
            ))
    return records

def attach_native_text(metadata, native_blocks, minimum_coverage=0.9):
    """Attach only native text boxes almost fully contained by an OCR crop."""
    for item in metadata:
        item["native_text"] = None
        item["native_block_ids"] = []
        item["native_ids"] = []
        item["native_structures"] = []
        item["list_structures"] = []
        box = item.get("box")
        if not box or len(box) != 4:
            continue
        matched = []
        for block in native_blocks or []:
            native_box = block.get("bbox")
            text = str(block.get("text") or "").strip()
            if not text or not isinstance(native_box, (list, tuple)) or len(native_box) != 4:
                continue
            native_area = max(0.0, float(native_box[2]) - float(native_box[0])) * max(
                0.0, float(native_box[3]) - float(native_box[1]))
            if native_area <= 0:
                continue
            overlap = max(0.0, min(float(box[2]), float(native_box[2])) - max(float(box[0]), float(native_box[0]))) * max(
                0.0, min(float(box[3]), float(native_box[3])) - max(float(box[1]), float(native_box[1])))
            if overlap / native_area >= minimum_coverage:
                matched.append(block)
        if matched:
            item["native_text"] = "\n".join(str(block["text"]).strip() for block in matched)
            item["native_block_ids"] = [block.get("block_id") for block in matched]
            item["native_ids"] = [block.get("native_id") for block in matched if block.get("native_id")]
            item["native_structures"] = [
                block["native_structure"] for block in matched
                if isinstance(block.get("native_structure"), dict)
            ]
            item["list_structures"] = [
                block["list_structure"] for block in matched
                if isinstance(block.get("list_structure"), dict)
            ]

