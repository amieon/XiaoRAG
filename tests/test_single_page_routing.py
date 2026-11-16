import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_single_page import routing_only


class SinglePageRoutingTests(unittest.TestCase):
    def test_route_ledger_records_expected_engine_for_three_page_kinds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "page.png"
            image = np.full((80, 120, 3), 255, dtype=np.uint8)
            image[20:22, 10:110] = 0
            Image.fromarray(image).save(image_path)
            expected = {"text": "light_ocr", "formula": "vlm", "table": "vlm"}
            for kind, engine in expected.items():
                output = root / kind
                records = routing_only(image_path, output, kind, 0.45)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["engine"], engine)
                line = json.loads((output / "routes.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(line["block_id"], [0, 0])
                self.assertEqual(line["engine_name"], "paddleocr" if engine == "light_ocr" else "paddleocr_vl")
                self.assertIn("route_reason", line)
                self.assertIn("risk_score", line)


if __name__ == "__main__":
    unittest.main()
