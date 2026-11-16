from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.document_ir import Block, DocumentIR, Page


def write_document(path: Path, document_id: str, text: str) -> None:
    page_id = f"{document_id}_p0001"
    document = DocumentIR(
        document_id=document_id,
        title=document_id,
        pages=[Page(
            page_id=page_id,
            page_no=1,
            width=100,
            height=100,
            blocks=[Block(
                block_id=f"{page_id}_b0000",
                page_id=page_id,
                block_type="paragraph",
                bbox=[0, 0, 100, 20],
                text=text,
                reading_order=0,
            )],
        )],
    )
    document.to_json(path)


class KnowledgeUnitCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.script = ROOT / "scripts" / "build_knowledge_units.py"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, source: Path, output: Path, *extra: str):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [sys.executable, str(self.script), "--input", str(source), "--output", str(output), *extra],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=30,
        )

    def test_single_results_directory_writes_four_files(self):
        results = self.root / "one" / "results"
        write_document(results / "document.json", "doc_one", "离线构建。")
        output = self.root / "knowledge-units"
        completed = self.run_cli(results, output, "--config", str(ROOT / "config/knowledge_units.json"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for name in ("knowledge-units.jsonl", "element-locations.jsonl", "document-tree.json", "manifest.json"):
            self.assertTrue((output / name).is_file())
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["document_id"], "doc_one")
        self.assertEqual(summary["counts"]["indexable_units"], 1)

    def test_batch_recurses_and_isolates_bad_document(self):
        batch = self.root / "batch"
        write_document(batch / "documents/a.txt/results/document.json", "doc_a", "甲。")
        write_document(batch / "documents/nested/b.txt/results/document.json", "doc_b", "乙。")
        bad = batch / "documents/bad/results/document.json"
        bad.parent.mkdir(parents=True)
        bad.write_text("{}", encoding="utf-8")
        output = self.root / "knowledge-units-batch"
        completed = self.run_cli(batch, output, "--workers", "2")
        self.assertEqual(completed.returncode, 2, completed.stderr)
        manifest = json.loads((output / "batch-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["counts"], {"success": 2, "failed": 1})
        self.assertEqual(manifest["cpu_worker_count"], 2)
        self.assertTrue((output / "documents/a.txt/results/knowledge-units/manifest.json").is_file())
        self.assertTrue((output / "documents/nested/b.txt/results/knowledge-units/manifest.json").is_file())
        self.assertTrue((output / "documents/bad/results/knowledge-units-error.json").is_file())


if __name__ == "__main__":
    unittest.main()
