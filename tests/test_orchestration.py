from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.document_ir import Block, DocumentIR, Page, SourceRef
from doc_parser.orchestration import run_folder, run_path
from doc_parser.pipeline import PageParseResult, ParserConfig, write_page_result


class FakePool:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False
        self.lock = threading.Lock()

    def parse_pages(self, jobs):
        results = []
        for job in jobs:
            image_path = Path(job["image_path"])
            document_id, page_no = job["document_id"], job["page_no"]
            with self.lock:
                self.owner.page_calls.append((document_id, page_no))
            page_id = f"{document_id}_p{page_no:04d}"
            page = Page(
                page_id=page_id, page_no=page_no, width=40, height=30,
                image_ref=job.get("image_ref"),
                source=SourceRef(engine="fake_layout", raw_page=page_no),
                blocks=[Block(
                    block_id=f"{page_id}_b0000", page_id=page_id,
                    block_type="paragraph", bbox=[0, 0, 40, 30],
                    text=f"fake:{image_path.name}", reading_order=0,
                    source=SourceRef(engine="fake_ocr", raw_page=page_no),
                    metadata={"raw_label": "text"},
                )],
            )
            result = PageParseResult(page=page, records=[], markdown=page.blocks[0].text, layout={"boxes": []})
            write_page_result(job["output_dir"], result)
            results.append(result)
        return results

    def parallelism_report(self):
        return {"worker_count": 1, "mode": "fake"}

    def close(self):
        self.closed = True
        self.owner.closed = True


class OrchestrationTests(unittest.TestCase):
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
        output = self.inputs / "generated"

        manifest = run_folder(self.inputs, output, prepare_only=True)
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["file_count"], 4)
        self.assertEqual(manifest["counts"]["success"], 3)
        self.assertEqual(manifest["counts"]["unsupported"], 1)
        self.assertEqual(manifest["parallelism"]["gpu"]["worker_count"], 0)
        text_ir = DocumentIR.from_dict(json.loads(
            (output / "documents/a.txt/results/document.json").read_text(encoding="utf-8")
        ))
        image_ir = DocumentIR.from_dict(json.loads(
            (output / "documents/nested/page.png/results/document.json").read_text(encoding="utf-8")
        ))
        self.assertEqual(text_ir.pages[0].blocks[0].text, "hello")
        self.assertTrue(image_ir.pages[0].metadata["prepared_only"])

    def test_full_folder_reuses_and_closes_one_pool(self):
        self._image(self.inputs / "one.png")
        self._image(self.inputs / "nested" / "two.png")
        owner = type("PoolOwner", (), {"created": 0, "closed": False, "page_calls": []})()

        def factory():
            owner.created += 1
            return FakePool(owner)

        output = self.root / "out"
        manifest = run_folder(self.inputs, output, runtime_pool_factory=factory, build_knowledge_units=True)
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(owner.created, 1)
        self.assertTrue(owner.closed)
        self.assertEqual(len(owner.page_calls), 2)
        self.assertEqual(manifest["parallelism"]["gpu"]["mode"], "fake")
        for relative in ("one.png", "nested/two.png"):
            path = output / "documents" / relative / "results/document.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pages"][0]["blocks"][0]["text"], "fake:0001.png")
            self.assertIn(Path(relative).name, payload["source_path"])
            knowledge_unit_root = output / "documents" / relative / "results/knowledge-units"
            self.assertTrue((knowledge_unit_root / "knowledge-units.jsonl").is_file())
            self.assertTrue((knowledge_unit_root / "element-locations.jsonl").is_file())

    def test_nested_archive_has_flat_provenance_and_real_ir(self):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as archive:
            archive.writestr("b.txt", "nested text")
        source = self.inputs / "bundle.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("a.txt", "outer text")
            archive.writestr("inner.zip", inner.getvalue())
        output = self.root / "archive-out"

        manifest = run_path(source, output, prepare_only=True, build_knowledge_units=True)

        self.assertEqual(manifest["native_page_count"], 2)
        self.assertEqual(manifest["knowledge_units"]["indexable_unit_count"], 2)
        prepared = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(prepared["document_count"], 2)
        self.assertNotIn("children", prepared)
        self.assertEqual(
            [item["source_path"] for item in prepared["documents"]],
            ["bundle.zip/a.txt", "bundle.zip/inner.zip/b.txt"],
        )
        document = DocumentIR.from_dict(json.loads(
            (output / "results/document.json").read_text(encoding="utf-8")
        ))
        self.assertEqual(
            [page.blocks[0].text for page in document.pages],
            ["outer text", "nested text"],
        )

    def test_native_ir_builds_knowledge_units_in_same_run(self):
        source = self.inputs / "native.txt"
        source.write_text("第一段。\n\n第二段。", encoding="utf-8")
        output = self.root / "native-out"
        manifest = run_path(source, output, prepare_only=True, build_knowledge_units=True)
        self.assertEqual(manifest["knowledge_units"]["indexable_unit_count"], 1)
        knowledge_unit_root = output / "results/knowledge-units"
        for name in ("knowledge-units.jsonl", "element-locations.jsonl", "document-tree.json", "manifest.json"):
            self.assertTrue((knowledge_unit_root / name).is_file())
        rows = [json.loads(line) for line in (knowledge_unit_root / "knowledge-units.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertIn("第一段。", rows[0]["text"])

    def test_bad_file_does_not_stop_later_file(self):
        (self.inputs / "a.png").write_bytes(b"broken")
        (self.inputs / "b.txt").write_text("still runs", encoding="utf-8")
        manifest = run_folder(self.inputs, self.root / "out", prepare_only=True)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["counts"]["failed"], 1)
        self.assertEqual(manifest["counts"]["success"], 1)
        self.assertIn("still runs", (self.root / "out/documents/b.txt/results/document.json").read_text(encoding="utf-8"))

    def test_run_path_maps_single_file_page_limit_argument(self):
        source = self.inputs / "one.txt"
        source.write_text("single", encoding="utf-8")
        result = run_path(source, self.root / "single-out", prepare_only=True, max_pages_per_document=2)
        self.assertEqual(result["page_count"], 1)

    def test_cpu_document_preparation_really_runs_in_parallel(self):
        for index in range(3):
            (self.inputs / f"{index}.txt").write_text(str(index), encoding="utf-8")
        from doc_parser import orchestration
        original = orchestration.prepare_document
        lock = threading.Lock()
        active = 0
        maximum = 0

        def slow_prepare(*args, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.06)
                return original(*args, **kwargs)
            finally:
                with lock:
                    active -= 1

        with patch("doc_parser.orchestration.prepare_document", side_effect=slow_prepare):
            result = run_folder(
                self.inputs, self.root / "out", prepare_only=True,
                config=ParserConfig(cpu_workers=3),
            )
        self.assertEqual(result["counts"]["success"], 3)
        self.assertGreaterEqual(maximum, 2)
        self.assertEqual(result["parallelism"]["cpu_worker_count"], 3)


if __name__ == "__main__":
    unittest.main()
