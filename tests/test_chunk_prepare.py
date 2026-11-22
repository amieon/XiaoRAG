from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_chunking import normalize_document, prepare_document


def raw_block(page_id, ident, text, order, kind="paragraph", **extra):
    return {
        "block_id": ident,
        "page_id": page_id,
        "block_type": kind,
        "bbox": [10, 10 + order, 90, 20 + order],
        "text": text,
        "reading_order": order,
        **extra,
    }


class ChunkPrepareTests(unittest.TestCase):
    def test_tree_bfs_or_dfs_never_replaces_reading_order(self):
        page1, page2 = "p1", "p2"
        ir = {
            "document_id": "doc",
            "pages": [
                {"page_id": page2, "page_no": 2, "width": 100, "height": 100,
                 "blocks": [raw_block(page2, "h2", "第二章", 0, "section_heading", metadata={"heading_level": 1})]},
                {"page_id": page1, "page_no": 1, "width": 100, "height": 100,
                 "blocks": [
                     raw_block(page1, "body", "正文", 2),
                     raw_block(page1, "h1-1", "1.1 范围", 1, "section_heading", metadata={"heading_level": 2}),
                     raw_block(page1, "h1", "第一章", 0, "section_heading", metadata={"heading_level": 1}),
                 ]},
            ],
        }
        # 同一 IR 可分别序列化为 BFS/DFS 章节树；二者都不参与物理阅读排序。
        bfs = deepcopy(ir)
        bfs["serialized_structure"] = {"traversal": "bfs", "block_ids": ["h1", "h2", "h1-1"]}
        dfs = deepcopy(ir)
        dfs["serialized_structure"] = {"traversal": "dfs", "block_ids": ["h1", "h1-1", "h2"]}
        first = normalize_document(bfs)
        second = normalize_document(dfs)
        self.assertEqual(first, second)
        self.assertEqual([block["block_id"] for block in first], ["h1", "h1-1", "body", "h2"])
        self.assertEqual(first[0]["sources"][0]["bbox"], [10, 10, 90, 20])
        self.assertEqual(first[1]["level"], 2)

    def test_canonical_source_and_raw_label_are_adapted_without_mutation(self):
        page_id = "p1"
        ir = {"document_id": "doc", "pages": [{"page_id": page_id, "page_no": 1, "width": 100, "height": 100,
            "blocks": [
                raw_block(page_id, "header", "内部页眉", 0, metadata={"raw_label": "header"}),
                raw_block(page_id, "body", "正文", 1, polygon=[[10, 11], [90, 11], [90, 21]],
                          source={"engine": "fixture", "raw_block_id": "native-7", "raw_path": "/blocks/7"}),
            ]}]}
        snapshot = deepcopy(ir)
        result = prepare_document(ir)
        self.assertEqual(ir, snapshot)
        self.assertEqual([block["block_id"] for block in result["blocks"]], ["body"])
        self.assertEqual(result["excluded"][0]["exclusion_reason"], "explicit_furniture")
        source = result["blocks"][0]["sources"][0]
        self.assertEqual(source["native_id"], "native-7")
        self.assertEqual(source["source"]["raw_path"], "/blocks/7")
        self.assertEqual(source["polygon"], [[10, 11], [90, 11], [90, 21]])
        self.assertIsNone(result["format_result"])
        self.assertNotIn("parents", result)
        self.assertNotIn("children", result)

    def test_cross_page_merge_requires_explicit_acceptance(self):
        def document(decision):
            return {"document_id": "doc", "pages": [
                {"page_id": "p1", "page_no": 1, "width": 100, "height": 100,
                 "blocks": [raw_block("p1", "a", "前半段", 0)]},
                {"page_id": "p2", "page_no": 2, "width": 100, "height": 100,
                 "blocks": [raw_block("p2", "b", "后半段", 0)]},
            ], "relations": [{"relation_id": "r1", "relation_type": "paragraph_continuation",
                              "source_id": "a", "target_id": "b", "metadata": {"decision": decision}}]}
        accepted = prepare_document(document("accepted"))
        self.assertEqual(len(accepted["blocks"]), 1)
        self.assertEqual(accepted["blocks"][0]["text"], "前半段\n后半段")
        self.assertEqual([ref["block_id"] for ref in accepted["blocks"][0]["sources"]], ["a", "b"])
        candidate = prepare_document(document("candidate"))
        self.assertEqual([block["text"] for block in candidate["blocks"]], ["前半段", "后半段"])
        self.assertEqual(candidate["merges"][0]["reason"], "relation_not_explicitly_accepted")

    def test_ppt_mime_routes_and_notes_reenter_only_format_channel(self):
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        ir = {"document_id": "slides", "mime_type": mime, "pages": [{"page_id": "s1", "page_no": 1,
            "width": 960, "height": 540, "metadata": {"unit_type": "slide"}, "blocks": [
                raw_block("s1", "title", "页标题", 0, "section_heading", metadata={"heading_level": 1}),
                raw_block("s1", "body", "页正文", 1),
                raw_block("s1", "notes", "备注说明", 2, "speaker_notes"),
            ]}]}
        native = {"slides": [{"slide": 1, "notes": "备注说明", "notes_observed": True}]}
        result = prepare_document(ir, native)
        self.assertEqual([block["block_id"] for block in result["blocks"]], ["title", "body"])
        self.assertEqual(result["excluded"][0]["block_id"], "notes")
        self.assertEqual(result["format_result"]["channel"], "presentation")
        group = result["format_result"]["groups"][0]
        self.assertEqual(group["context"]["slide_ids"], ["s1"])
        self.assertEqual(group["text_override"].count("备注说明"), 1)

    def test_excel_mime_routes_to_native_rows(self):
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ir = {"document_id": "book", "mime_type": mime, "pages": [{"page_id": "sheet1",
            "metadata": {"unit_type": "worksheet"}, "blocks": [
                raw_block("sheet1", "table", "原始表格", 0, "table", metadata={"sheet_name": "记录", "region_id": "r1"}),
            ]}]}
        native = {"sheets": [{"name": "记录", "state": "visible", "merged_cells": [],
            "rows": [
                {"row_number": 1, "values": ["编号", "名称"], "cells": []},
                {"row_number": 2, "values": [1, "甲"], "cells": [
                    {"coordinate": "A2", "column": 1, "value": 1},
                    {"coordinate": "B2", "column": 2, "value": "甲"},
                ]},
            ],
            "data_regions": [{"region_id": "r1", "range": "A1:B2", "header_rows": [1]}],
        }]}
        result = prepare_document(ir, native)
        self.assertEqual(result["format_result"]["channel"], "spreadsheet")
        self.assertEqual(len(result["format_result"]["groups"]), 1)
        text = result["format_result"]["groups"][0]["text_override"]
        self.assertIn("编号: 1; 名称: 甲", text)
        self.assertTrue(any(ref.get("cell") == "B2" for ref in result["format_result"]["groups"][0]["native_refs"]))

    def test_default_config_is_not_shared_between_runs(self):
        config_path = ROOT / "src" / "doc_chunking" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        ir = {"document_id": "doc", "pages": []}
        first = prepare_document(ir, config=config)
        first["config"]["preprocess"]["strip_toc"] = False
        second = prepare_document(ir, config=config)
        self.assertTrue(second["config"]["preprocess"]["strip_toc"])
        self.assertTrue(config["preprocess"]["strip_toc"])


if __name__ == "__main__":
    unittest.main()
