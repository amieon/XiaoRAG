"""Small image/layout gate. No recognizer, language model, or GT is called here.

Scores are interpretable risk signals, not calibrated error probabilities.
"""
import math
from statistics import median


def _clip(value):
    return max(0.0, min(1.0, float(value)))


def _layout_box(value):
    box = [float(v) for v in value]
    if (len(box) != 4 or not all(math.isfinite(v) for v in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("Invalid layout box")
    return box


def layout_context(blocks, detector_blocks=None):
    """Use pre-recognition detector fields only, never block content."""
    contexts = []
    formulae, invalid_formula_boxes = [], False
    for candidate in (detector_blocks if detector_blocks is not None else blocks):
        if "formula" in str(candidate.get("label", "")).lower():
            try:
                formulae.append((candidate, _layout_box(candidate.get("box"))))
            except (TypeError, ValueError, OverflowError):
                invalid_formula_boxes = True
    for block in blocks:
        invalid_fields = ["formula_boxes"] if invalid_formula_boxes else []
        try:
            box = _layout_box(block.get("box"))
        except (TypeError, ValueError, OverflowError):
            box = None
            invalid_fields.append("box")
        overlap = 0.0
        if box is not None:
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            for formula, other in formulae:
                if formula is block:
                    continue
                other_area = max(0, other[2] - other[0]) * max(0, other[3] - other[1])
                intersection = (max(0, min(box[2], other[2]) - max(box[0], other[0]))
                                * max(0, min(box[3], other[3]) - max(box[1], other[1])))
                if min(area, other_area) > 0:
                    overlap = max(overlap, intersection / min(area, other_area))
        context = {"layout_label": str(block.get("label", "unknown")), "formula_overlap": float(overlap),
                   "invalid_layout_fields": invalid_fields}
        for key in ("angle", "vertical", "math_risk", "score"):
            if key in block:
                value = block[key]
                try:
                    context[key] = value.item() if hasattr(value, "item") else value
                except (TypeError, ValueError):
                    invalid_fields.append(key)
        contexts.append(context)
    return contexts


def _runs(values):
    count, active = 0, False
    for value in values:
        if value and not active:
            count += 1
        active = bool(value)
    return count


def _components(mask):
    """8-connected components on an already bounded thumbnail."""
    height, width = mask.shape
    pending = set(int(i) for i in mask.reshape(-1).nonzero()[0])
    boxes = []
    while pending:
        seed = pending.pop()
        stack = [seed]
        y, x = divmod(seed, width)
        x0 = x1 = x
        y0 = y1 = y
        area = 0
        while stack:
            current = stack.pop()
            cy, cx = divmod(current, width)
            area += 1
            x0, x1, y0, y1 = min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    neighbor = ny * width + nx
                    if neighbor in pending:
                        pending.remove(neighbor)
                        stack.append(neighbor)
        if area >= 3:
            boxes.append((x0, y0, x1 + 1, y1 + 1, area))
    return boxes


def image_features(image, max_side=384):
    import numpy as np
    from PIL import Image

    if image is None:
        raise ValueError("No crop image available")
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        # Mean is independent of RGB/BGR channel ordering in native crops.
        array = array[:, :, :3].mean(axis=2)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("Expected a finite grayscale/RGB/BGR crop")
    if array.min() >= 0 and array.max() <= 1 and array.dtype.kind == "f":
        array = array * 255
    gray = Image.fromarray(np.clip(array, 0, 255).astype("uint8"))
    original_width, original_height = gray.size
    gray.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    hist = np.bincount(pixels.astype("uint8").ravel(), minlength=256).astype(float)
    probability = hist / hist.sum()
    weights = probability.cumsum()
    means = (probability * np.arange(256)).cumsum()
    variance = (means[-1] * weights - means) ** 2 / np.maximum(weights * (1 - weights), 1e-12)
    threshold = int(variance.argmax())
    mask = pixels <= threshold
    if mask.mean() > 0.5:
        mask = ~mask
    boxes = _components(mask)
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    typical_height = median(heights) if heights else 0
    typical_width = median(widths) if widths else 0
    row_runs = _runs(mask.sum(axis=1) >= max(1, mask.shape[1] * 0.008))
    column_runs = _runs(mask.sum(axis=0) >= max(1, mask.shape[0] * 0.008))
    occupied_columns = np.flatnonzero(mask.any(axis=0))
    occupied_width = float(occupied_columns[-1] - occupied_columns[0] + 1) if len(occupied_columns) else 0
    characters_across = occupied_width / max(1, typical_width)
    small_fraction = sum(h < typical_height * 0.6 for h in heights) / len(heights) if heights else 0.0
    height_cv = float(np.std(heights) / max(1, np.mean(heights))) if heights else 0.0
    bars = sum(w > max(typical_width * 2.5, mask.shape[1] * 0.06)
               and h <= max(2, typical_height * 0.3) for w, h in zip(widths, heights))
    contrast = float(np.percentile(pixels, 99) - np.percentile(pixels, 1)) / 255
    # Projection evidence AND a narrow glyph arrangement distinguish a vertical
    # line from an ordinary tall paragraph, for which aspect ratio alone fails.
    vertical = (original_height / original_width >= 1.2 and row_runs >= 3 and column_runs <= 3
                and characters_across < 4.0)
    math_score = _clip(0.55 * min(1, bars) + 0.3 * _clip((small_fraction - 0.08) / 0.25)
                       + 0.25 * _clip((height_cv - 0.3) / 0.6)) if len(boxes) >= 4 else 0.0
    return {"contrast": contrast, "ink_fraction": float(mask.mean()), "component_count": len(boxes),
            "small_component_fraction": small_fraction, "height_cv": height_cv, "horizontal_bars": bars,
            "row_runs": row_runs, "column_runs": column_runs, "characters_across": characters_across,
            "vertical_score": float(vertical), "math_score": math_score,
            "original_width": original_width, "original_height": original_height,
            "analysis_width": gray.width, "analysis_height": gray.height}


class VisualGate:
    version = "visual_layout_v1"

    def __init__(self, config):
        self.config = config

    def assess(self, region):
        metadata, reasons, signals = region.metadata, [], {}
        layout_errors = list(metadata.get("invalid_layout_fields", []))

        def numeric(key, default=0.0):
            value = metadata.get(key)
            if value is None:
                return default
            try:
                value = float(value)
                if not math.isfinite(value) or (key == "formula_overlap" and not 0 <= value <= 1):
                    raise ValueError("Nonfinite or out-of-range layout value")
                return value
            except (TypeError, ValueError, OverflowError):
                layout_errors.append(key)
                return default

        def flag(key):
            value = metadata.get(key)
            if value is None:
                return False
            if isinstance(value, (bool, int, float)) and value in (0, 1):
                return bool(value)
            layout_errors.append(key)
            return False

        label = str(metadata.get("layout_label", "text")).lower()
        if label in {"table", "chart", "seal", "spotting", "image", "figure"} or "formula" in label:
            reasons.append("complex_layout_label")
            signals["layout"] = 1.0
        angle = numeric("angle") % 180
        if (flag("vertical") or "vertical" in label or
                min(angle, 180 - angle) >= self.config.orientation_angle_threshold):
            reasons.append("layout_orientation_risk")
            signals["orientation"] = 1.0
        formula_overlap = numeric("formula_overlap")
        if flag("math_risk") or formula_overlap >= self.config.formula_overlap_threshold:
            reasons.append("layout_formula_overlap")
            signals["formula_overlap"] = 1.0
        if layout_errors:
            reasons.append("invalid_layout_metadata")
            signals["unknown_layout"] = 1.0
        features = {}
        if not signals:
            try:
                features = image_features(region.image, self.config.max_feature_side)
                if features["vertical_score"]:
                    reasons.append("visual_vertical_arrangement")
                    signals["vertical"] = 0.95
                if features["math_score"] >= self.config.math_score_threshold:
                    reasons.append("visual_math_structure")
                    signals["math_structure"] = features["math_score"]
                if features["contrast"] < self.config.min_visual_contrast:
                    reasons.append("low_visual_contrast")
                    signals["contrast"] = 0.75
                if features["component_count"] == 0 or features["ink_fraction"] < 0.001:
                    reasons.append("insufficient_visual_evidence")
                    signals["missing_ink"] = 0.9
                if min(region.width, region.height) < self.config.min_crop_side:
                    reasons.append("tiny_crop")
                    signals["resolution"] = 0.75
            except (ValueError, TypeError, ImportError) as exc:
                reasons.append("visual_features_unavailable")
                signals["unknown"] = 1.0
                features["error"] = f"{type(exc).__name__}: {exc}"
        # Aggregate independent risks conservatively; this is not a probability.
        score = 1 - math.prod(1 - v for v in signals.values())
        branch = "vlm" if score >= self.config.risk_threshold else "light"
        return {"method": self.version, "branch": branch, "risk_score": score,
                "threshold": self.config.risk_threshold, "signals": signals, "features": features,
                "invalid_layout_fields": layout_errors,
                "reasons": reasons if reasons else ["ordinary_text_visuals"], "uses_recognized_text": False}
