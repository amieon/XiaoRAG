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

from doc_parser.preparation import (
    _attach_docx_native_structure, _extract_docx, _extract_native,
    _render_office, detect_format, prepare_document,
)
from doc_parser.routing import attach_native_text


class PreparationTests(unittest.TestCase):
    def test_docx_heading_styles_become_native_structure_evidence(self):
        try:
            from docx import Document
        except ImportError:
            self.skipTest("python-docx is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "heading.docx"
            document = Document()
            document.add_heading("第一章 总则", level=1)
            document.add_paragraph("普通正文")
            document.save(source)
            native = _extract_docx(source)
        self.assertEqual(len(native["structure"]), 1)
        evidence = native["structure"][0]
        self.assertEqual(evidence["heading_level"], 1)
        self.assertIn(evidence["source"], {"docx_style", "docx_style_outline", "docx_outline"})
        self.assertEqual(native["paragraphs"][0]["native_structure"], evidence)

        blocks = [{"text": "第一章  总则"}, {"text": "普通正文"}]
        _attach_docx_native_structure(blocks, native)
        self.assertEqual(blocks[0]["native_structure"], evidence)
        self.assertEqual(blocks[0]["native_structure_match"], "unique_exact_text")
        self.assertNotIn("native_structure", blocks[1])

    def test_ambiguous_docx_heading_text_is_not_arbitrarily_aligned(self):
        evidence = {"source": "docx_style", "role": "heading", "native_id": "p1", "text": "总结"}
        duplicate = {**evidence, "native_id": "p2"}
        native = {"format": "docx", "structure": [evidence, duplicate]}
        blocks = [{"text": "总结"}]
        _attach_docx_native_structure(blocks, native)
        self.assertNotIn("native_structure", blocks[0])
        self.assertEqual(blocks[0]["native_structure_match"], "ambiguous_exact_text")
        self.assertEqual(blocks[0]["native_structure_candidate_ids"], ["p1", "p2"])

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

    def test_nested_zip_uses_bfs_flat_documents_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inner_buffer = root / "inner.zip"
            with zipfile.ZipFile(inner_buffer, "w") as archive:
                archive.writestr("docs/inside.txt", "nested")
            source = root / "bundle.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("readme.txt", "contained")
                archive.writestr("nested/inner.zip", inner_buffer.read_bytes())
            self.assertEqual(detect_format(source), "zip")
            manifest = prepare_document(source, root / "out")
            self.assertEqual(manifest["expansion"], "breadth_first_streaming")
            self.assertEqual(manifest["document_count"], 2)
            self.assertNotIn("children", manifest)
            self.assertEqual(
                [item["source_path"] for item in manifest["documents"]],
                ["bundle.zip/readme.txt", "bundle.zip/nested/inner.zip/docs/inside.txt"],
            )
            self.assertEqual(
                manifest["documents"][1]["archive_path"],
                "bundle.zip/nested/inner.zip",
            )
            self.assertFalse((root / "out/documents").joinpath("0002-inside/children").exists())
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

    def test_gb18030_text_is_decoded_strictly_and_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "legacy.txt"
            source.write_bytes("第一段：增量更新。\n第二段：版本管理。".encode("gb18030"))
            native = _extract_native(source, "text")
            self.assertEqual(native["encoding"], "gb18030")
            self.assertIn("增量更新", native["text"])
            manifest = prepare_document(source, root / "out")
            self.assertEqual(manifest["status"], "success")
            saved = json.loads((root / "out/native/objects.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["text"], native["text"])

    def test_office_converter_local_encoding_cannot_crash_error_reporting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "slides.pptx"
            source.write_bytes(b"fixture")
            staging = root / "staging"

            def fake_run(command, **kwargs):
                self.assertNotIn("text", kwargs)
                return type("Completed", (), {
                    "returncode": 1, "stderr": "转换失败".encode("gb18030"), "stdout": b"",
                })()

            with patch("doc_parser.preparation.subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "转换失败"):
                    _render_office(source, staging, "soffice")

    def test_legacy_pps_uses_office_pdf_renderer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "slides.pps"
            source.write_bytes(b"\xd0\xcf\x11\xe0legacy-powerpoint")
            output = root / "out"
            self.assertEqual(detect_format(source), "pps")

            def fake_pdf(path, pages_dir, dpi, max_pages=None):
                pages_dir.mkdir(parents=True)
                target = pages_dir / "0001.png"
                target.write_bytes(b"fake-image")
                return [{"page_no": 1, "path": target, "width": 100, "height": 80}]

            with patch("doc_parser.preparation._render_office", return_value=root / "slides.pdf") as convert, \
                    patch("doc_parser.preparation._render_pdf", side_effect=fake_pdf), \
                    patch("doc_parser.preparation._office_visible_blocks", return_value=[]):
                manifest = prepare_document(source, output)
            self.assertEqual(manifest["detected_format"], "pps")
            self.assertEqual(manifest["page_count"], 1)
            convert.assert_called_once()

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
