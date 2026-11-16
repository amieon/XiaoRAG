import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from doc_parser.document_orchestration import detect_format, prepare_document
from doc_parser.page_routing import _attach_native_text, route_visual_groups


class DocumentOrchestrationTests(unittest.TestCase):
    def test_pdf_creates_page_package_and_does_not_trust_pdf_text_layer(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.pdf"
            doc = fitz.open()
            page = doc.new_page(width=100, height=200)
            page.insert_text((10, 20), "Embedded PDF text")
            doc.save(source)
            doc.close()

            manifest = prepare_document(source, root / "out", dpi=144)
            self.assertEqual(manifest["page_count"], 1)
            package_path = root / "out" / manifest["pages"][0]["package"]
            package = json.loads(package_path.read_text(encoding="utf-8"))
            self.assertEqual(package["schema_version"], "page-package.v1")
            self.assertEqual(package["image"]["coordinate_system"], "pixel_top_left")
            self.assertEqual(package["native_blocks"], [])
            self.assertEqual(package["policy"]["pdf_text_layer_policy"], "ignore")
            self.assertTrue((root / "out" / package["image"]["path"]).is_file())

    def test_page_limit_is_checked_before_pdf_rendering(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "two-pages.pdf"
            document = fitz.open()
            document.new_page()
            document.new_page()
            document.save(source)
            document.close()
            with self.assertRaisesRegex(ValueError, "page count exceeds"):
                prepare_document(source, root / "out", max_pages=1)
            self.assertFalse((root / "out" / "pages" / "0001.png").exists())

    def test_docx_native_extraction_preserves_paragraph_and_table(self):
        try:
            from docx import Document
        except ImportError:
            self.skipTest("python-docx is unavailable")
        from doc_parser.document_orchestration import _extract_docx

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.docx"
            document = Document()
            document.add_heading("Heading", level=1)
            document.add_paragraph("Native paragraph")
            table = document.add_table(rows=1, cols=1)
            table.cell(0, 0).text = "Cell value"
            document.save(source)
            extracted = _extract_docx(source)
            self.assertEqual(extracted["paragraphs"][-1]["text"], "Native paragraph")
            self.assertEqual(extracted["tables"][0][0][0][0], "Cell value")

    def test_office_render_text_is_carried_as_bbox_aligned_native_candidate(self):
        try:
            import fitz
            from docx import Document
        except ImportError:
            self.skipTest("python-docx and PyMuPDF are required")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.docx"
            document = Document()
            document.add_paragraph("Native paragraph")
            document.save(source)
            rendered_pdf = root / "rendered.pdf"
            pdf = fitz.open()
            pdf_page = pdf.new_page(width=100, height=100)
            pdf_page.insert_text((10, 20), "Native paragraph")
            pdf.save(rendered_pdf)
            pdf.close()
            with patch("doc_parser.document_orchestration._render_office", return_value=rendered_pdf):
                manifest = prepare_document(source, root / "out", dpi=144)
            package_path = root / "out" / manifest["pages"][0]["package"]
            package = json.loads(package_path.read_text(encoding="utf-8"))
            self.assertEqual(package["native_blocks"][0]["text"], "Native paragraph")
            self.assertGreater(package["native_blocks"][0]["bbox"][2], 0)

    def test_text_file_uses_native_only_result_without_model_runtime(self):
        from run_document_orchestrator import run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "notes.txt"
            source.write_text("No OCR needed", encoding="utf-8")
            result = run(source, root / "out", dpi=150, office_converter=None,
                         threshold=0.45, prepare_only=False)
            self.assertTrue(result["native_only"])
            self.assertEqual((root / "out" / "results" / "document.md").read_text(encoding="utf-8"),
                             "No OCR needed")

    def test_prepare_only_reports_rendered_page_count(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is unavailable")
        from run_document_orchestrator import run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "two-pages.pdf"
            document = fitz.open()
            document.new_page()
            document.new_page()
            document.save(source)
            document.close()
            result = run(source, root / "out", dpi=100, office_converter=None,
                         threshold=0.45, prepare_only=True)
            self.assertEqual(result["page_count"], 2)

    def test_native_text_requires_bbox_coverage(self):
        metadata = [{"box": [0, 0, 100, 40]}, {"box": [100, 0, 175, 40]}]
        native = [{"block_id": "native-1", "bbox": [10, 10, 80, 30], "text": "Already extracted"},
                  {"block_id": "native-2", "bbox": [150, 10, 190, 30], "text": "Second"}]
        _attach_native_text(metadata, native)
        self.assertEqual(metadata[0]["native_text"], "Already extracted")
        self.assertIsNone(metadata[1]["native_text"])

    def test_native_text_bypasses_ocr_while_formula_still_uses_vlm(self):
        class Gate:
            version = "test"
            def assess(self, region):
                return {"method": "test", "branch": "light", "risk_score": 0.0,
                        "reasons": [], "uses_recognized_text": False}

        calls = {"ocr": 0, "vlm": 0}
        class Crop:
            shape = (20, 40, 3)

        group = {(32, 32): {
            "images": [Crop(), Crop()],
            "queries": ["OCR:", "Formula Recognition:"],
            "vlm_block_ids": [(0, 0), (0, 1)],
            "figure_token_maps": [{}, {}],
            "region_metadata": [{"box": [0, 0, 40, 20], "native_text": "Native text"},
                                {"box": [0, 20, 40, 40]}],
        }}

        def recognize_text(images):
            calls["ocr"] += len(images)
            return [{"result": "OCR"} for _ in images]

        def recognize_vlm(groups):
            calls["vlm"] += len(groups[(32, 32)]["images"])
            groups[(32, 32)]["vlm_results"] = [{"result": "formula"}]

        records = route_visual_groups(group, recognize_text, recognize_vlm, gate=Gate())
        self.assertEqual(calls, {"ocr": 0, "vlm": 1})
        self.assertEqual(records[0]["engine"], "native_text")
        self.assertEqual(records[0]["result"], "Native text")
        self.assertEqual(records[1]["engine"], "vlm")

    def test_zip_dispatches_children_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "bundle.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("readme.txt", "contained")
            self.assertEqual(detect_format(source), "zip")
            manifest = prepare_document(source, root / "out")
            self.assertEqual(manifest["children"][0]["status"], "success")

            malicious = root / "bad.zip"
            with zipfile.ZipFile(malicious, "w") as archive:
                archive.writestr("../outside.txt", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                prepare_document(malicious, root / "bad-out")


if __name__ == "__main__":
    unittest.main()
