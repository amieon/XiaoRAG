import tempfile
import unittest
import json
import io
from unittest.mock import patch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import fitz
except ImportError:  # pragma: no cover - deployment dependency
    fitz = None

from run_single_page import _render_pdf_pages, _renumber_page_record, _write_document_outputs, _document_links, full_run, _read_response


@unittest.skipIf(fitz is None, "PyMuPDF is required for PDF input")
class PdfInputTests(unittest.TestCase):
    def test_pdf_pages_render_in_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.pdf"
            document = fitz.open()
            document.new_page(width=240, height=120)
            document.new_page(width=320, height=160)
            document.save(str(source))
            document.close()

            from PIL import Image
            sizes = []
            pages = []
            with patch.dict('os.environ', {'PADDLEOCR_PDF_DPI': '144'}):
                for index, path in _render_pdf_pages(source, root / 'rendered'):
                    with Image.open(path) as image:
                        sizes.append(image.size)
                    pages.append((index, path))
            self.assertEqual([index for index, _ in pages], [0, 1])
            self.assertEqual(sizes, [(480, 240), (640, 320)])
            self.assertFalse(any(path.is_file() for _, path in pages))

    def test_document_outputs_keep_page_and_engine_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.pdf"
            pages = []
            records = []
            for page in (1, 2):
                page_dir = root / "pages" / f"{page:04d}"
                page_dir.mkdir(parents=True)
                (page_dir / "page.md").write_text(f"page {page}\n", encoding="utf-8")
                pages.append({"page": page, "output": page_dir.relative_to(root).as_posix(), "route_count": 1, "engines": {"light_ocr": 1}})
                records.append(_renumber_page_record({"page_id": 0, "block_id": [0, 0], "engine": "light_ocr"}, page - 1))

            _write_document_outputs(root, source, records, pages)

            self.assertIn("page 1", (root / "document.md").read_text(encoding="utf-8"))
            self.assertIn("page 2", (root / "document.md").read_text(encoding="utf-8"))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["page_count"], 2)
            self.assertEqual(manifest["engines"], {"light_ocr": 2})
            self.assertEqual(len((root / "routes.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual([r['block_id'] for r in records], [[0, 0], [1, 0]])

    def test_asset_links_remain_valid_after_merging(self):
        text = '![chart](imgs/a.jpg) <img src="imgs/b.jpg"> [web](https://example.org)'
        result = _document_links(text, 'pages/0002')
        self.assertIn('(pages/0002/imgs/a.jpg)', result)
        self.assertIn('src="pages/0002/imgs/b.jpg"', result)
        self.assertIn('(https://example.org)', result)

    def test_pdf_rerun_refuses_stale_output_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                full_run(output / 'input.pdf', output, .45)

    def test_invalid_pdf_fails_and_does_not_render_pages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'broken.pdf').write_text('not a PDF')
            with self.assertRaises(Exception):
                list(_render_pdf_pages(root / 'broken.pdf', root / 'rendered'))
            self.assertEqual(list((root / 'rendered').glob('*.png')), [])

    def test_worker_native_logs_do_not_break_protocol(self):
        with patch('sys.stderr', new=io.StringIO()):
            self.assertEqual(_read_response(io.StringIO('native backend log\n{"outputs": []}\n')), {'outputs': []})
        with self.assertRaisesRegex(RuntimeError, 'closed stdout'):
            _read_response(io.StringIO(''))


if __name__ == "__main__":
    unittest.main()
