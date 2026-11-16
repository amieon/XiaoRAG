import json
import tempfile
import unittest
from pathlib import Path

from doc_parser.document_ir import (
    DocumentIR,
    DocumentIRValidationError,
    Relation,
    from_mineru_middle_json,
    from_paddleocr_json,
    iter_blocks,
)


class DocumentIRTest(unittest.TestCase):
    def test_mineru_middle_json_is_normalized(self):
        middle = {
            "_backend": "pipeline",
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [1000, 1400],
                    "para_blocks": [
                        {
                            "type": "title",
                            "bbox": [10, 20, 500, 60],
                            "lines": [{
                                "spans": [{"content": "第一章 总则"}],
                            }],
                        },
                        {
                            "type": "table",
                            "bbox": [10, 100, 900, 500],
                            "table_caption": "付款表",
                        },
                    ],
                }
            ],
        }
        document = from_mineru_middle_json(
            middle, document_id="doc_001", source_path="contract_middle.json"
        )
        self.assertEqual(document.pages[0].page_id, "doc_001_p0001")
        self.assertEqual(document.pages[0].blocks[0].block_type, "section_heading")
        self.assertEqual(document.pages[0].blocks[0].text, "第一章 总则")
        self.assertEqual(document.pages[0].blocks[1].block_type, "table")
        self.assertEqual(document.pages[0].blocks[1].text, "付款表")
        self.assertEqual(document.pages[0].blocks[0].source.engine, "mineru")

    def test_round_trip_json_preserves_ir(self):
        document = from_mineru_middle_json(
            {"pdf_info": [{"page_idx": 0, "page_size": [10, 20], "para_blocks": []}]},
            document_id="doc_002",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document_ir.json"
            document.to_json(path)
            restored = DocumentIR.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(restored.to_dict(), document.to_dict())

    def test_relation_must_reference_document_objects(self):
        document = from_mineru_middle_json(
            {"pdf_info": [{"page_idx": 0, "page_size": [10, 20], "para_blocks": []}]},
            document_id="doc_003",
        )
        document.relations.append(
            Relation(
                relation_id="rel_1",
                relation_type="section_contains",
                source_id="missing",
                target_id=document.pages[0].page_id,
            )
        )
        with self.assertRaises(DocumentIRValidationError):
            document.to_dict()

    def test_iter_blocks_follows_page_and_reading_order(self):
        middle = {
            "pdf_info": [
                {
                    "page_idx": 1,
                    "page_size": [10, 20],
                    "para_blocks": [{"type": "text", "bbox": [0, 0, 1, 1], "text": "p2"}],
                },
                {
                    "page_idx": 0,
                    "page_size": [10, 20],
                    "para_blocks":[{"type": "text", "bbox": [0, 0, 1, 1], "text": "p1"}],
                },
            ]
        }
        document = from_mineru_middle_json(middle, document_id="doc_004")
        self.assertEqual([block.text for block in iter_blocks(document)], ["p1", "p2"])

    def test_paddleocr_json_is_normalized(self):
        document = from_paddleocr_json(
            [
                {
                    "page_index": 0,
                    "width": 1200,
                    "height": 1600,
                    "input_path": "contract.pdf",
                    "parsing_res_list": [
                        {
                            "block_id": 3,
                            "block_label": "text",
                            "block_content": "付款条件",
                            "block_bbox": [10, 20, 300, 50],
                            "block_polygon_points": [[10, 20], [300, 20], [300, 50], [10, 50]],
                            "block_order": 1,
                        },
                        {
                            "block_label": "table",
                            "block_content": "<table></table>",
                            "block_bbox": [10, 60, 1000, 400],
                        },
                    ],
                }
            ],
            document_id="doc_005",
        )
        self.assertEqual(document.pages[0].blocks[0].block_type, "paragraph")
        self.assertEqual(document.pages[0].blocks[1].block_type, "table")
        self.assertEqual(document.pages[0].blocks[0].source.engine, "paddleocr")
        self.assertEqual(len(document.pages[0].blocks[0].polygon), 4)


if __name__ == "__main__":
    unittest.main()
