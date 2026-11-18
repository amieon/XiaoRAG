from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.document_ir import Block, Page, iter_blocks, DocumentIR
from doc_parser.reading_order import (
    ReadingOrderConfig,
    plan_reading_order,
    restore_page_reading_order,
)


def make_page(specs, *, width=200, height=240):
    blocks = [
        Block(
            block_id=block_id,
            page_id="doc_p0001",
            block_type=block_type,
            bbox=bbox,
            text=block_id,
            reading_order=index,
            metadata={"raw_label": raw_label} if raw_label else {},
        )
        for index, (block_id, bbox, block_type, raw_label) in enumerate(specs)
    ]
    return Page(
        page_id="doc_p0001", page_no=1, width=width, height=height, blocks=blocks,
    )


def ordered_ids(page):
    return [block.block_id for block in sorted(page.blocks, key=lambda item: item.reading_order)]


class ReadingOrderTests(unittest.TestCase):
    def test_single_column_is_top_to_bottom_even_if_input_is_shuffled(self):
        page = make_page([
            ("bottom", [20, 100, 180, 130], "paragraph", None),
            ("title", [20, 10, 180, 30], "section_heading", None),
            ("middle", [20, 50, 180, 80], "paragraph", None),
        ])
        result = restore_page_reading_order(page)
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(ordered_ids(page), ["title", "middle", "bottom"])
        self.assertEqual(result.ordered_block_ids, ordered_ids(page))

    def test_aligned_paragraph_gaps_do_not_interleave_two_columns(self):
        page = make_page([
            ("right_1", [110, 20, 190, 40], "paragraph", None),
            ("left_2", [10, 60, 90, 80], "paragraph", None),
            ("right_2", [110, 60, 190, 80], "paragraph", None),
            ("left_1", [10, 20, 90, 40], "paragraph", None),
        ])
        result = restore_page_reading_order(page)
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(ordered_ids(page), ["left_1", "left_2", "right_1", "right_2"])
        self.assertIn("vertical_cut", result.strategies)

    def test_full_width_title_and_lower_band_surround_columns(self):
        page = make_page([
            ("bottom", [10, 140, 190, 170], "paragraph", None),
            ("right", [110, 60, 190, 110], "paragraph", None),
            ("title", [10, 10, 190, 30], "section_heading", None),
            ("left", [10, 60, 90, 110], "paragraph", None),
        ])
        result = restore_page_reading_order(page)
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(ordered_ids(page), ["title", "left", "right", "bottom"])
        self.assertEqual(result.strategies, ["horizontal_cut", "vertical_cut"])

    def test_rtl_direction_reverses_column_order(self):
        page = make_page([
            ("left", [10, 20, 90, 100], "paragraph", None),
            ("right", [110, 20, 190, 100], "paragraph", None),
        ])
        result = restore_page_reading_order(page, ReadingOrderConfig(direction="rtl"))
        self.assertEqual(result.ordered_block_ids, ["right", "left"])

    def test_single_figure_wrap_uses_rectangles_and_groups_caption(self):
        page = make_page([
            ("figure", [110, 30, 190, 90], "figure", "image"),
            ("bottom", [10, 115, 190, 150], "paragraph", "text"),
            ("caption", [110, 90, 190, 115], "figure_caption", "figure_caption"),
            ("left", [10, 30, 90, 115], "paragraph", "text"),
        ])
        result = restore_page_reading_order(page)
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(ordered_ids(page), ["left", "figure", "caption", "bottom"])
        self.assertIn("single_figure_wrap", result.strategies)
        self.assertEqual(page.blocks[0].bbox, [110, 30, 190, 90])

    def test_headers_are_preserved_but_kept_out_of_main_flow(self):
        page = make_page([
            ("header", [10, 0, 190, 10], "header", "header"),
            ("right", [110, 40, 190, 100], "paragraph", "text"),
            ("left", [10, 40, 90, 100], "paragraph", "text"),
            ("page_number", [90, 220, 110, 230], "number", "number"),
        ])
        result = restore_page_reading_order(page)
        self.assertEqual(result.main_block_ids, ["left", "right"])
        self.assertEqual(result.auxiliary_block_ids, ["header", "page_number"])
        self.assertEqual(result.ordered_block_ids, ["left", "right", "header", "page_number"])

    def test_unresolved_overlap_is_risky_and_never_drops_blocks(self):
        page = make_page([
            ("a", [10, 20, 120, 100], "paragraph", None),
            ("b", [40, 30, 180, 90], "paragraph", None),
        ])
        result = restore_page_reading_order(page)
        self.assertEqual(result.status, "risky")
        self.assertTrue(any(reason.startswith("overlapping_text_blocks:") for reason in result.risks))
        self.assertIn("no_valid_geometric_cut", result.risks)
        self.assertCountEqual(result.ordered_block_ids, ["a", "b"])
        self.assertEqual(len(set(result.ordered_block_ids)), 2)

    def test_invalid_geometry_falls_back_without_losing_identity(self):
        page = make_page([
            ("empty", [20, 20, 20, 50], "paragraph", None),
            ("valid", [20, 10, 100, 30], "paragraph", None),
        ])
        result = restore_page_reading_order(page)
        self.assertEqual(result.status, "risky")
        self.assertIn("coordinate_fallback_after_invalid_geometry", result.risks)
        self.assertCountEqual(result.ordered_block_ids, ["empty", "valid"])

    def test_original_region_order_and_iter_blocks_remain_traceable(self):
        page = make_page([
            ("right", [110, 20, 190, 100], "paragraph", None),
            ("left", [10, 20, 90, 100], "paragraph", None),
        ])
        restore_page_reading_order(page)
        self.assertEqual(page.blocks[0].metadata["layout_reading_order"], 0)
        self.assertEqual(page.blocks[1].metadata["layout_reading_order"], 1)
        document = DocumentIR(document_id="doc", pages=[page])
        self.assertEqual([block.block_id for block in iter_blocks(document)], ["left", "right"])
        self.assertEqual(page.metadata["reading_order"]["status"], "confirmed")

    def test_disabled_mode_preserves_original_region_order(self):
        page = make_page([
            ("right", [110, 20, 190, 100], "paragraph", None),
            ("left", [10, 20, 90, 100], "paragraph", None),
        ])
        result = plan_reading_order(page, ReadingOrderConfig(enabled=False))
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.ordered_block_ids, ["right", "left"])


if __name__ == "__main__":
    unittest.main()
