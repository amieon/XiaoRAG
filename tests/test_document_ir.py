from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.document_ir import (
    Block, DocumentIR, DocumentIRValidationError, Page, Relation, SourceRef, iter_blocks,
)


def fixture():
    page_id = "doc_p0001"
    blocks = [
        Block(
            block_id=f"{page_id}_b0001", page_id=page_id, block_type="paragraph",
            bbox=[0, 30, 100, 50], text="第二块", reading_order=1,
        ),
        Block(
            block_id=f"{page_id}_b0000", page_id=page_id, block_type="document_title",
            bbox=[0, 0, 100, 20], text="标题", reading_order=0,
            source=SourceRef(engine="native_extraction"),
        ),
    ]
    return DocumentIR(
        document_id="doc", title="标题",
        pages=[Page(page_id=page_id, page_no=1, width=100, height=100, blocks=blocks)],
    )


class DocumentIRTests(unittest.TestCase):
    def test_round_trip_and_reading_order(self):
        document = fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "document.json"
            document.to_json(path)
            restored = DocumentIR.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(restored, document)
        self.assertEqual([block.text for block in iter_blocks(restored)], ["标题", "第二块"])

    def test_relation_must_reference_known_objects(self):
        document = fixture()
        document.relations.append(Relation(
            relation_id="r1", relation_type="references",
            source_id="doc_p0001_b0000", target_id="missing",
        ))
        with self.assertRaises(DocumentIRValidationError):
            document.validate()

    def test_invalid_bbox_is_rejected(self):
        document = fixture()
        document.pages[0].blocks[0].bbox = [0, 0, -1, 10]
        with self.assertRaises(DocumentIRValidationError):
            document.validate()


if __name__ == "__main__":
    unittest.main()
