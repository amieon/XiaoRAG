from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doc_parser.document_ir import Block, DocumentIR, Page, SourceRef
from doc_parser.explicit_pipeline import (
    ExplicitParserConfig,
    PageParseResult,
    write_page_result,
)
from doc_parser.folder_orchestration import run_folder, run_path


class FakeRuntime:
    def __init__(self, owner):
        self.owner = owner
        self.calls = []
        self.closed = False

    def parse_page(self, image_path, *, document_id, page_no, output_dir, **kwargs):
        self.calls.append(Path(image_path).name)
        page_id = f"{document_id}_p{page_no:04d}"
        page = Page(
            page_id=page_id, page_no=page_no, width=40, height=30,
            image_ref=kwargs.get("image_ref"),
            source=SourceRef(engine="fake_public_layout", raw_page=page_no),
            blocks=[Block(
                block_id=f"{page_id}_b0000", page_id=page_id,
                block_type="paragraph", bbox=[0, 0, 40, 30],
                text=f"fake:{Path(image_path).name}", reading_order=0,
                source=SourceRef(engine="fake_public_ocr", raw_page=page_no),
                metadata={"raw_label": "text"},
            )],
        )
        result = PageParseResult(page=page, records=[], markdown=page.blocks[0].text, layout={"boxes": []})
        write_page_result(output_dir, result)
        return result

    def close(self):
        self.closed = True
        self.owner.closed = True


class FolderOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def _image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((30, 40, 3), 255, dtype=np.uint8)).save(path)

    def test_prepare_only_recurses_and_native_results_are_real_ir(self):
        (self.inputs / "a.txt").write_text("hello", encoding="utf-8")
        (self.inputs / "nested").mkdir()
        (self.inputs / "nested" / "a.json").write_text('{"x": 1}', encoding="utf-8")
        self._image(self.inputs / "nested" / "page.png")
        (self.inputs / "unknown.bin").write_bytes(b"not-a-document")
        output = self.inputs / "generated"  # Deliberately inside input: must not recurse into it.

        manifest = run_folder(self.inputs, output, prepare_only=True)
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["file_count"], 4)
        self.assertEqual(manifest["counts"]["success"], 3)
        self.assertEqual(manifest["counts"]["unsupported"], 1)
        text_ir_path = output / "documents" / "a.txt" / "results" / "document.json"
        image_ir_path = output / "documents" / "nested" / "page.png" / "results" / "document.json"
        text_ir = DocumentIR.from_dict(json.loads(text_ir_path.read_text(encoding="utf-8")))
        image_ir = DocumentIR.from_dict(json.loads(image_ir_path.read_text(encoding="utf-8")))
        self.assertEqual(text_ir.pages[0].blocks[0].text, "hello")
        self.assertEqual(image_ir.pages[0].metadata["prepared_only"], True)
        self.assertEqual(image_ir.pages[0].blocks, [])

    def test_full_folder_reuses_and_closes_one_runtime(self):
        self._image(self.inputs / "one.png")
        self._image(self.inputs / "nested" / "two.png")
        owner = type("RuntimeOwner", (), {"created": 0, "closed": False})()

        def factory():
            owner.created += 1
            return FakeRuntime(owner)

        output = self.root / "out"
        manifest = run_folder(self.inputs, output, runtime_factory=factory)
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(owner.created, 1)
        self.assertTrue(owner.closed)
        first = output / "documents" / "one.png" / "results" / "document.json"
        second = output / "documents" / "nested" / "two.png" / "results" / "document.json"
        first_text = first.read_text(encoding="utf-8")
        second_text = second.read_text(encoding="utf-8")
        self.assertIn('"text": "fake:0001.png"', first_text)
        self.assertIn('"text": "fake:0001.png"', second_text)
        self.assertIn('"source_path":', first_text)
        self.assertIn("one.png", first_text)
        self.assertIn("two.png", second_text)

    def test_bad_file_does_not_stop_later_file(self):
        (self.inputs / "a.png").write_bytes(b"broken")
        (self.inputs / "b.txt").write_text("still runs", encoding="utf-8")
        manifest = run_folder(self.inputs, self.root / "out", prepare_only=True)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["counts"]["failed"], 1)
        self.assertEqual(manifest["counts"]["success"], 1)
        self.assertIn("still runs", (self.root / "out" / "documents" / "b.txt" / "results" / "document.json").read_text(encoding="utf-8"))

    def test_run_path_maps_single_file_page_limit_argument(self):
        source = self.inputs / "one.txt"
        source.write_text("single", encoding="utf-8")
        result = run_path(source, self.root / "single-out", prepare_only=True, max_pages_per_document=2)
        self.assertEqual(result["page_count"], 1)


if __name__ == "__main__":
    unittest.main()
