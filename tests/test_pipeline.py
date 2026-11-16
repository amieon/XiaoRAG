from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doc_parser.document_ir import DocumentIR
from doc_parser.pipeline import (
    ParserConfig,
    RegionPipeline,
    public_result_dict,
)
from doc_parser.routing import GateConfig, Region, VisualGate
from doc_parser.model_backend import PaddleModelBackend


class FakeBackend:
    def __init__(self, boxes):
        self.boxes = boxes
        self.text_calls = []
        self.vlm_queries = []

    def detect_layout(self, image):
        return {"res": {"boxes": self.boxes}}

    def recognize_text(self, images):
        self.text_calls.append(list(images))
        return [
            {"result": f"light-{index}", "line_scores": [0.95, 0.85], "line_boxes": [[0, 0, 5, 5]]}
            for index, _ in enumerate(images)
        ]

    def recognize_vlm(self, groups):
        for group in groups.values():
            self.vlm_queries.extend(group["queries"])
            group["vlm_results"] = [
                {"result": "E=mc^2" if query == "Formula Recognition:" else "vlm-text"}
                for query in group["queries"]
            ]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_path = self.root / "page.png"
        image = np.full((100, 160, 3), 255, dtype=np.uint8)
        image[8:10, 8:45] = 0
        image[18:20, 8:45] = 0
        Image.fromarray(image).save(self.image_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_public_result_unwraps_res(self):
        self.assertEqual(public_result_dict({"res": {"result": "ok"}}), {"result": "ok"})

    def test_layout_adapter_drops_full_page_pixels(self):
        backend = PaddleModelBackend.__new__(PaddleModelBackend)
        backend.layout_model = type("FakeLayout", (), {
            "predict": lambda self, image: iter([{
                "input_img": image, "boxes": [],
            }]),
        })()
        result = backend.detect_layout(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(result, {"boxes": []})

    def test_formula_overlap_threshold_is_configurable(self):
        image = np.full((32, 48, 3), 255, dtype=np.uint8)
        image[8:10, 3:45] = 0
        metadata = {
            "layout_label": "text", "formula_overlap": 0.2,
            "box": [0, 0, 48, 32], "invalid_layout_fields": [],
        }
        default_decision = VisualGate(GateConfig()).assess(Region(image, metadata))
        relaxed_decision = VisualGate(GateConfig(formula_overlap_threshold=0.3)).assess(Region(image, metadata))
        self.assertEqual(default_decision["branch"], "vlm")
        self.assertEqual(relaxed_decision["branch"], "light")

    def test_three_region_types_produce_ordered_ir_and_assets(self):
        backend = FakeBackend([
            {"label": "text", "coordinate": [0, 0, 55, 40], "score": 0.9},
            {"label": "display_formula", "coordinate": [60, 0, 110, 40]},
            {"label": "image", "coordinate": [0, 50, 50, 90], "score": 0.88},
        ])
        output = self.root / "out"
        result = RegionPipeline(backend).parse_image(
            self.image_path,
            document_id="doc_test",
            page_no=1,
            source_path="page.png",
            image_ref="pages/0001.png",
            output_dir=output,
        )
        self.assertEqual(len(backend.text_calls), 1)
        self.assertEqual(backend.vlm_queries, ["Formula Recognition:"])
        self.assertEqual([block.text for block in result.page.blocks], ["light-0", "E=mc^2", ""])
        self.assertEqual(
            [block.block_type for block in result.page.blocks],
            ["paragraph", "formula", "figure"],
        )
        self.assertEqual(result.page.blocks[0].confidence, 0.85)
        self.assertEqual(result.page.blocks[0].source.engine, "paddleocr")
        self.assertEqual(result.page.blocks[1].source.engine, "paddleocr_vl")
        self.assertEqual(result.page.blocks[2].source.engine, "layout_detection")
        self.assertEqual(result.page.blocks[2].confidence, 0.88)
        self.assertTrue((output / "page.md").is_file())
        self.assertTrue((output / "layout.json").is_file())
        self.assertEqual(len(list((output / "assets").glob("*.png"))), 3)
        document = DocumentIR(document_id="doc_test", pages=[result.page])
        document.validate()
        payload = json.loads((output / "page_result.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "page-result.v2")
        self.assertEqual(len(payload["page"]["blocks"]), 3)

    def test_public_vlm_adapter_passes_items_and_generation_limits(self):
        backend = PaddleModelBackend.__new__(PaddleModelBackend)
        backend.config = ParserConfig(max_new_tokens=123)
        calls = []

        class FakeVLM:
            def predict(self, items, **kwargs):
                calls.append((items, kwargs))
                return iter([{"res": {"result": "public-vlm"}} for _ in items])

        backend.vlm_model = FakeVLM()
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        groups = {(1000, 2000): {"images": [image], "queries": ["OCR:"]}}
        backend.recognize_vlm(groups)
        self.assertEqual(groups[(1000, 2000)]["vlm_results"][0]["result"], "public-vlm")
        items, kwargs = calls[0]
        self.assertIs(items[0]["image"], image)
        self.assertEqual(items[0]["query"], "OCR:")
        self.assertEqual(kwargs["min_pixels"], 1000)
        self.assertEqual(kwargs["max_pixels"], 2000)
        self.assertEqual(kwargs["max_new_tokens"], 123)

    def test_native_bbox_skips_all_recognizers(self):
        backend = FakeBackend([
            {"label": "text", "coordinate": [0, 0, 55, 40]},
        ])
        result = RegionPipeline(backend).parse_image(
            self.image_path,
            document_id="doc_native",
            page_no=1,
            native_blocks=[{
                "block_id": "native-1", "text": "原生文本",
                "bbox": [1, 1, 54, 39],
            }],
        )
        self.assertEqual(backend.text_calls, [])
        self.assertEqual(backend.vlm_queries, [])
        self.assertEqual(result.page.blocks[0].text, "原生文本")
        self.assertEqual(result.page.blocks[0].confidence, 1.0)
        self.assertEqual(result.page.blocks[0].source.engine, "native_text")
        self.assertEqual(result.records[0]["engine"], "native_text")

    def test_config_round_trip_and_threshold_override(self):
        config = ParserConfig.from_dict({
            "gate": {"formula_overlap_threshold": 0.3},
            "ocr_pixel_limits": [1000, 2000],
        })
        self.assertEqual(config.gate.formula_overlap_threshold, 0.3)
        self.assertEqual(config.gate.risk_threshold, GateConfig().risk_threshold)
        self.assertEqual(config.ocr_pixel_limits, (1000, 2000))
        restored = ParserConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)


if __name__ == "__main__":
    unittest.main()
