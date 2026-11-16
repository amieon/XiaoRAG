from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.preparation import _render_office, detect_format, prepare_document
from doc_parser.routing import attach_native_text


class PreparationTests(unittest.TestCase):
    def test_pdf_creates_page_package_without_trusting_text_layer(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.pdf"
            document = fitz.open()
            page = document.new_page(width=100, height=200)
            page.insert_text((10, 20), "Embedded PDF text")
            document.save(source)
            document.close()
            manifest = prepare_document(source, root / "out", dpi=144)
            self.assertEqual(manifest["page_count"], 1)
            package_path = root / "out" / manifest["pages"][0]["package"]
            package = json.loads(package_path.read_text(encoding="utf-8"))
            self.assertEqual(package["native_blocks"], [])
            self.assertEqual(package["image"]["coordinate_system"], "pixel_top_left")
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
            self.assertFalse((root / "out/pages/0001.png").exists())

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

    def test_office_conversion_uses_isolated_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "a.docx"
            source.write_bytes(b"fixture")
            staging = root / "unique-staging"

            def fake_run(command, **kwargs):
                target = staging / "a.pdf"
                target.write_bytes(b"%PDF-fake")
                return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            with patch("doc_parser.preparation.subprocess.run", side_effect=fake_run) as run:
                result = _render_office(source, staging, "soffice")
            self.assertEqual(result, staging / "a.pdf")
            command = run.call_args.args[0]
            self.assertIn(f"-env:UserInstallation={(staging / 'libreoffice-profile').resolve().as_uri()}", command)

    def test_native_text_requires_bbox_coverage(self):
        metadata = [{"box": [0, 0, 100, 40]}, {"box": [100, 0, 175, 40]}]
        native = [
            {"block_id": "native-1", "bbox": [10, 10, 80, 30], "text": "Already extracted"},
            {"block_id": "native-2", "bbox": [150, 10, 190, 30], "text": "Second"},
        ]
        attach_native_text(metadata, native)
        self.assertEqual(metadata[0]["native_text"], "Already extracted")
        self.assertIsNone(metadata[1]["native_text"])


if __name__ == "__main__":
    unittest.main()
