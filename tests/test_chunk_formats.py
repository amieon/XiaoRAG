import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_chunking.formats import format_groups


def block(ident, text="", kind="paragraph", *, unit=1, unit_type="slide", **metadata):
    unit_id = f"unit_{unit}"
    return {"block_id": ident, "kind": kind, "native_kind": kind, "text": text,
            "unit_index": unit, "unit_id": unit_id, "unit_type": unit_type,
            "page_number": None, "order": 0, "level": 1 if kind == "heading" else None,
            "metadata": metadata, "sources": [{"block_id": ident, "unit_id": unit_id,
            "unit_index": unit, "unit_type": unit_type, "page_number": None,
            "bbox": None, "source_type": "native", "confidence": None, "native_id": ident}]}


def rows(values):
    result = []
    for number, values_row in enumerate(values, 1):
        result.append({"row_number": number, "values": values_row,
                       "is_blank": not any(value is not None for value in values_row),
                       "cells": [{"coordinate": f"{chr(64 + col)}{number}", "column": col,
                                  "value": value} for col, value in enumerate(values_row, 1) if value is not None]})
    return result


def sheet_fixture(values=None, **sheet_fields):
    values = values or [["编号", "名称"], [1, "甲"], [2, "乙"], [3, "丙"]]
    region = {"region_id": "r1", "range": f"A1:B{len(values)}", "start_row": 1,
              "end_row": len(values), "start_column": 1, "end_column": 2,
              "header_rows": [1, 2], "header_status": "heuristic_candidates_not_gold"}
    sheet = {"name": "记录", "state": "visible", "rows": rows(values), "data_regions": [region], "merged_cells": []}
    sheet.update(sheet_fields)
    return {"format": "xlsx", "sheets": [sheet]}


def run_excel(native, blocks=None, **options):
    if blocks is None:
        blocks = [block("table", "ORIGINAL REGION TABLE", "table", unit_type="worksheet", sheet_name="记录", region_id="r1")]
    return format_groups(blocks, {"format": "xlsx", "pages": []}, native, {}, {"formats": options})


class ChunkFormatTests(unittest.TestCase):
    def test_slide_contains_title_text_table_image_and_deduplicated_notes(self):
        blocks = [block("title", "标题", "heading"), block("body", "正文"),
                  block("table", "| A | B |", "table"), block("image", kind="image", refs=["native/assets/a.png"]),
                  block("notes", "备注说明", "speaker_notes", excluded_from_body=True),
                  block("header", "不应进正文", "header"), block("toc", "目录项", "toc")]
        native = {"slides": [{"slide": 1, "notes": "备注说明", "notes_observed": True}]}
        result = format_groups(blocks, {"format": "pptx", "pages": []}, native, {}, {})
        self.assertEqual(len(result["groups"]), 1)
        group = result["groups"][0]
        for value in ("标题", "正文", "| A | B |", "native/assets/a.png", "备注说明"):
            self.assertIn(value, group["text_override"])
        self.assertEqual(group["text_override"].count("备注说明"), 1)
        self.assertEqual({b["block_id"] for b in group["blocks"]}, {"title", "body", "table", "image", "notes"})
        self.assertEqual(group["context"]["slide_ids"], ["unit_1"])
        self.assertEqual(group["native_refs"][0]["pointer"], "/slides/0/notes")

    def test_native_notes_supplement_ir_and_short_slides_merge_only_when_enabled(self):
        blocks = [block("one", "短页1"), block("two", "短页2", unit=2), block("four", "短页4", unit=4)]
        native = {"slides": [{"slide": 1, "notes": "原生备注", "notes_observed": True}]}
        ordinary = format_groups(blocks, {"format": "pptx"}, native, {}, {})
        self.assertEqual(len(ordinary["groups"]), 3)
        merged = format_groups(blocks, {"format": "pptx"}, native, {},
                               {"formats": {"slide_merge_below_chars": 100, "slide_merge_max_chars": 200}})
        self.assertEqual(len(merged["groups"]), 2)
        self.assertEqual(merged["groups"][0]["context"]["slide_ids"], ["unit_1", "unit_2"])
        self.assertIn("原生备注", merged["groups"][0]["text_override"])
        self.assertEqual(merged["groups"][1]["context"]["slide_numbers"], [4])

    def test_slide_merge_honors_full_character_budget_and_source_identity(self):
        blocks = [block("one", "A  B"), block("two", "C  D", unit=2)]
        result = format_groups(blocks, {"source_format": "pptx"}, {}, {},
                               {"formats": {"slide_merge_below_chars": 50, "slide_merge_max_chars": 8}})
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual({b["block_id"] for g in result["groups"] for b in g["blocks"]}, {"one", "two"})

    def test_native_slide_images_are_supplemented_but_toc_is_not_restored(self):
        native = {"slides": [{"slide": 1, "notes": "演讲补充信息", "notes_observed": True,
                              "shapes": [{"text": "原生正文"}, {"asset_ref": "native/assets/fallback.png"}]}]}
        ir = {"format": "pptx", "pages": [{"page_id": "unit_1", "metadata": {"unit_type": "slide"}}]}
        result = format_groups([block("body", "IR正文")], ir, native, {}, {})
        self.assertIn("native/assets/fallback.png", result["groups"][0]["text_override"])
        self.assertNotIn("原生正文", result["groups"][0]["text_override"])
        ir["pages"][0]["metadata"]["is_toc"] = True
        omitted = format_groups([block("body", "IR目录")], ir, native, {}, {})
        self.assertEqual(omitted["groups"], [])
        ir["pages"][0]["metadata"].pop("is_toc")
        native_only = format_groups([], ir, native, {}, {})
        self.assertIn("原生正文", native_only["groups"][0]["text_override"])
        self.assertEqual(native_only["groups"][0]["text_override"].count("演讲补充信息"), 1)

    def test_compatibility_notes_pointer_and_excluded_images_remain_honest(self):
        native = {"slides": [{"slide": 1, "notes_text": "兼容备注", "notes_observed": True,
                              "shapes": [{"shape_id": 7, "asset_ref": "native/assets/excluded.png"}]}]}
        blocks = [block("body", "正文"), block("excluded", kind="image", refs=["native/assets/excluded.png"], excluded_from_body=True)]
        result = format_groups(blocks, {"format": "pptx"}, native, {}, {})
        group = result["groups"][0]
        self.assertNotIn("excluded.png", group["text_override"])
        self.assertEqual(group["native_refs"][0]["pointer"], "/slides/0/notes_text")
        ir = {"format": "pptx", "pages": [{"page_id": "unit_1", "metadata": {"unit_type": "slide"}}]}
        native["slides"][0].pop("notes_text")
        native["slides"][0]["shapes"] = [{"text": "明确排除原文"}]
        excluded = [block("excluded", "明确排除原文", excluded_from_body=True)]
        self.assertEqual(format_groups(excluded, ir, native, {}, {})["groups"], [])

    def test_excel_repeats_only_first_candidate_header_and_retains_data_cells(self):
        native = sheet_fixture()
        snapshot = copy.deepcopy(native)
        result = run_excel(native, excel_rows_per_group=1)
        self.assertEqual(len(result["groups"]), 3)
        self.assertEqual([g["context"]["row_numbers"] for g in result["groups"]], [[2], [3], [4]])
        for group in result["groups"]:
            self.assertEqual(group["context"]["header_rows"], [1])
            self.assertEqual(group["context"]["header_status"], "heuristic")
            self.assertIn("重复表头 行1", group["text_override"])
            self.assertNotIn("ORIGINAL REGION TABLE", group["text_override"])
        self.assertIn("编号: 1; 名称: 甲", result["groups"][0]["text_override"])
        refs = result["groups"][0]["native_refs"]
        self.assertTrue(any(ref.get("cell") == "B2" and ref["pointer"] == "/sheets/0/rows/1/cells/1" for ref in refs))
        self.assertEqual(native, snapshot)

    def test_excel_vertical_merges_extend_row_group_and_preserve_anchor_evidence(self):
        native = sheet_fixture([["编号", "名称"], [1, "合并内容"], [2, None], [3, None], [4, "结束"]], merged_cells=["B2:B4"])
        result = run_excel(native, excel_rows_per_group=1)
        first = result["groups"][0]
        self.assertEqual(first["context"]["row_numbers"], [2, 3, 4])
        self.assertTrue(first["context"]["row_limit_exceeded_for_merged_cells"])
        self.assertIn("[合并 B2:B4; 锚点 B2: 合并内容]", first["text_override"])
        self.assertTrue(any(ref.get("merged_range") == "B2:B4" for ref in first["native_refs"]))
        self.assertEqual(result["groups"][1]["context"]["row_numbers"], [5])

    def test_excel_merge_across_region_boundary_keeps_all_region_sources(self):
        native = sheet_fixture([["编号", "名称"], [1, "跨度"], [None, None], [2, None]],
            merged_cells=["B2:B4"], data_regions=[
                {"region_id": "r1", "range": "A1:B2", "header_rows": [1]},
                {"region_id": "r2", "range": "A4:B4", "header_rows": []}])
        blocks = [block("first", "first table", "table", unit_type="worksheet", sheet_name="记录", region_id="r1"),
                  block("second", "second table", "table", unit_type="worksheet", sheet_name="记录", region_id="r2")]
        result = run_excel(native, blocks, excel_rows_per_group=1)
        self.assertEqual(result["metrics"]["data_region_count"], 1)
        self.assertEqual(set(result["groups"][0]["context"]["region_ids"]), {"r1", "r2"})
        self.assertEqual({b["block_id"] for g in result["groups"] for b in g["blocks"]}, {"first", "second"})
        self.assertIn(3, result["groups"][0]["context"]["row_numbers"])

    def test_excel_missing_native_falls_back_and_sheet_extra_text_is_never_dropped(self):
        blocks = [block("table", "原IR表格", "table", unit_type="worksheet", sheet_name="记录", region_id="r1"),
                  block("title", "工作表额外说明", "heading", unit_type="worksheet", sheet_name="记录"),
                  block("footer", "页脚不可入正文", "footer", unit_type="worksheet")]
        fallback = run_excel({}, blocks)
        self.assertEqual(fallback["strategy"], "ir_table_fallback")
        self.assertEqual({b["block_id"] for g in fallback["groups"] for b in g["blocks"]}, {"table", "title"})
        self.assertFalse(fallback["groups"][0]["context"]["row_grouping_performed"])
        ordinary = run_excel(sheet_fixture(), blocks)
        self.assertIn("工作表额外说明", ordinary["groups"][0]["text_override"])
        self.assertEqual({b["block_id"] for g in ordinary["groups"] for b in g["blocks"]}, {"table", "title"})

    def test_excel_formulas_merged_headers_and_unlisted_rows_are_observable(self):
        native = sheet_fixture([["合并标题", None], [1, "=SUM(A2:A2)"], [2, "不能丢"]],
            merged_cells=["A1:B1"], data_regions=[{"region_id": "r1", "range": "A1:B2", "header_rows": [1]}])
        result = run_excel(native, excel_rows_per_group=1)
        text = "\n".join(group["text_override"] for group in result["groups"])
        self.assertIn("=SUM(A2:A2)", text)
        self.assertIn("不能丢", text)
        self.assertIn("A1:B1", text)
        self.assertIn("excel_region_supplemented_from_native_cells", result["warnings"])

    def test_native_excel_does_not_restore_explicitly_excluded_ir_region(self):
        native = sheet_fixture([["编号", "名称"], [1, "明确排除内容"]])
        blocks = [block("table", "明确排除内容", "table", unit_type="worksheet", sheet_name="记录",
                        region_id="r1", excluded_from_body=True)]
        result = run_excel(native, blocks)
        self.assertEqual(result["groups"], [])
        self.assertIn("excel_explicit_excluded_region_not_rendered", result["warnings"])

    def test_absent_merge_metadata_remains_unknown(self):
        native = sheet_fixture()
        native["sheets"][0].pop("merged_cells")
        result = run_excel(native)
        self.assertIn("excel_merge_metadata_unavailable", result["warnings"])
        self.assertFalse(result["groups"][0]["context"]["merged_cells_observed"])

    def test_routing_uses_explicit_format_instead_of_filename_or_labels(self):
        blocks = [block("ordinary", "正文", unit_type="logical_document")]
        self.assertIsNone(format_groups(blocks, {"source_path": "pretend.xlsx", "format": "markdown"}, {},
                                       {"category": "spreadsheet", "expected_structures": ["sheet"]}, {}))
        self.assertEqual(format_groups(blocks, {}, {}, {"detected_format": "xlsx"}, {})["channel"], "spreadsheet")


if __name__ == "__main__":
    unittest.main()
