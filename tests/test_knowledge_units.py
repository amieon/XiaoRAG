from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.document_ir import Block, DocumentIR, Page, SourceRef
from doc_parser.knowledge_units import KnowledgeUnitConfig, build_knowledge_units, validate_knowledge_units, write_knowledge_units


def block(page_id, index, text, kind="paragraph", **kwargs):
    return Block(
        block_id=f"{page_id}_b{index:04d}", page_id=page_id,
        block_type=kind, bbox=[10, index * 10, 90, index * 10 + 8],
        text=text, reading_order=index,
        source=SourceRef(engine="fixture"), metadata=kwargs.pop("metadata", {}),
        **kwargs,
    )


def rich_document():
    page1 = "doc_p0001"
    page2 = "doc_p0002"
    return DocumentIR(
        document_id="doc", content_hash="source-hash", title="企业手册",
        metadata={"pipeline": "fixture-pipeline"},
        pages=[
            Page(page_id=page1, page_no=1, width=800, height=1200, blocks=[
                block(page1, 0, "企业手册", "document_title"),
                block(page1, 1, "第一章 安全", "section_heading", metadata={"heading_level": 1}),
                block(page1, 2, "第一条规则。"),
                block(page1, 3, "第二条规则。"),
                block(page1, 4, "表1 权限", metadata={"raw_label": "table_title"}),
                block(page1, 5, "<table><tr><td>管理员</td></tr></table>", "table"),
                block(page1, 6, "图1 架构", metadata={"raw_label": "figure_title"}),
                block(page1, 7, "", "figure", metadata={"asset_path": "assets/arch.png"}),
                block(page1, 8, "E=mc^2", "formula"),
                block(page1, 9, "内部页眉", metadata={"raw_label": "header"}),
                block(page1, 10, ""),
            ]),
            Page(page_id=page2, page_no=2, width=800, height=1200, blocks=[
                block(page2, 0, "第二章 检索", "section_heading", metadata={"heading_level": 1}),
                block(page2, 1, "混合检索。"),
            ]),
        ],
    )


class KnowledgeUnitTests(unittest.TestCase):
    def test_rich_document_tree_atomic_units_and_locations(self):
        document = rich_document()
        result = build_knowledge_units(document)
        counts = result.manifest["counts"]
        self.assertEqual(counts["source_blocks"], 13)
        self.assertEqual(counts["mapped_blocks"], 11)
        self.assertEqual(counts["omitted_blocks"], 2)
        self.assertEqual(counts["container_units"], 3)
        self.assertEqual(counts["indexable_units"], 5)
        self.assertEqual(counts["caption_merges"], 2)
        self.assertEqual(result.manifest["coverage_ratio"], 11 / 13)

        indexable = [unit for unit in result.units if unit.metadata["indexable"]]
        by_type = {}
        for unit in indexable:
            by_type.setdefault(unit.unit_type, []).append(unit)
        self.assertEqual(len(by_type["paragraph"]), 2)
        self.assertIn("表1 权限", by_type["table"][0].text)
        self.assertIn("图1 架构", by_type["figure"][0].text)
        self.assertEqual(by_type["figure"][0].asset_paths, ["assets/arch.png"])
        self.assertEqual(by_type["formula"][0].text, "E=mc^2")
        self.assertEqual(by_type["paragraph"][0].heading_path, ["企业手册", "第一章 安全"])
        self.assertIn("企业手册 > 第一章 安全", by_type["paragraph"][0].context_prefix)

        locations = {(row.unit_id, row.block_id): row for row in result.locations}
        table = by_type["table"][0]
        self.assertEqual(len(table.block_ids), 2)
        for block_id in table.block_ids:
            self.assertIn((table.unit_id, block_id), locations)
        first_root = result.tree["children"][0]
        self.assertEqual(first_root["unit_type"], "document_title")
        self.assertEqual([child["unit_type"] for child in first_root["children"]], ["section", "section"])

    def test_long_block_splits_into_nonoverlapping_ranges_deterministically(self):
        page_id = "doc_p0001"
        document = DocumentIR(
            document_id="doc", pages=[Page(
                page_id=page_id, page_no=1, width=100, height=100,
                blocks=[block(page_id, 0, "甲。乙。丙。丁。戊。己。庚。辛。")],
            )],
        )
        config = KnowledgeUnitConfig(min_tokens=2, target_tokens=4, max_tokens=6)
        first = build_knowledge_units(document, config)
        second = build_knowledge_units(document, config)
        units = [unit for unit in first.units if unit.metadata["indexable"]]
        self.assertGreater(len(units), 1)
        self.assertTrue(all(unit.metadata["shared_block_split"] for unit in units))
        self.assertTrue(all(unit.metadata["split_reason"] == "length_boundary" for unit in units))
        rows = sorted(first.locations, key=lambda row: row.text_start)
        self.assertEqual(rows[0].text_start, 0)
        self.assertEqual(rows[-1].text_end, len(document.pages[0].blocks[0].text))
        for left, right in zip(rows, rows[1:]):
            self.assertLessEqual(left.text_end, right.text_start)
        self.assertEqual(
            [asdict(unit) for unit in first.units], [asdict(unit) for unit in second.units],
        )
        self.assertEqual(first.manifest, second.manifest)

    def test_hard_length_fallback_is_explicit(self):
        page_id = "doc_p0001"
        document = DocumentIR(
            document_id="doc", pages=[Page(
                page_id=page_id, page_no=1, width=100, height=100,
                blocks=[block(page_id, 0, "甲乙丙丁戊己庚辛壬癸")],
            )],
        )
        result = build_knowledge_units(document, KnowledgeUnitConfig(min_tokens=1, target_tokens=2, max_tokens=4))
        units = [unit for unit in result.units if unit.metadata["indexable"]]
        self.assertEqual(len(units), 3)
        self.assertTrue(any(unit.metadata["split_reason"] == "length_fallback" for unit in units))

    def test_paragraphs_do_not_merge_across_pages(self):
        p1, p2 = "doc_p0001", "doc_p0002"
        document = DocumentIR(document_id="doc", pages=[
            Page(page_id=p1, page_no=1, width=100, height=100, blocks=[block(p1, 0, "第一页")]),
            Page(page_id=p2, page_no=2, width=100, height=100, blocks=[block(p2, 0, "第二页")]),
        ])
        result = build_knowledge_units(document)
        units = [unit for unit in result.units if unit.metadata["indexable"]]
        self.assertEqual(len(units), 2)
        self.assertEqual([unit.page_start for unit in units], [1, 2])

    def test_heading_default_and_jump_are_diagnosed(self):
        page_id = "doc_p0001"
        document = DocumentIR(document_id="doc", pages=[Page(
            page_id=page_id, page_no=1, width=100, height=100, blocks=[
                block(page_id, 0, "一级标题", "section_heading"),
                block(page_id, 1, "三级标题", "section_heading", metadata={"heading_level": 3}),
                block(page_id, 2, "正文"),
            ],
        )])
        result = build_knowledge_units(document)
        types = [row["type"] for row in result.manifest["diagnostics"]]
        self.assertIn("heading_level_defaulted", types)
        self.assertIn("heading_level_jump", types)

    def test_validation_rejects_dangling_source_reference(self):
        result = build_knowledge_units(rich_document())
        result.locations[0].block_id = "missing-block"
        with self.assertRaisesRegex(ValueError, "Dangling element location"):
            validate_knowledge_units(rich_document(), result)

    def test_low_confidence_marks_review_and_output_is_byte_deterministic(self):
        document = rich_document()
        document.pages[1].blocks[1].confidence = 0.2
        result = build_knowledge_units(document)
        self.assertEqual(result.manifest["counts"]["needs_review_units"], 1)
        with tempfile.TemporaryDirectory() as temp:
            one, two = Path(temp) / "one", Path(temp) / "two"
            write_knowledge_units(result, one)
            write_knowledge_units(build_knowledge_units(document), two)
            for name in ("knowledge-units.jsonl", "element-locations.jsonl", "document-tree.json", "manifest.json"):
                self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())
            rows = [json.loads(line) for line in (one / "knowledge-units.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), len(result.units))


if __name__ == "__main__":
    unittest.main()
