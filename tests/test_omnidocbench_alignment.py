import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "align_omnidoctbench.py"
spec = importlib.util.spec_from_file_location("alignment", SCRIPT)
alignment = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(alignment)


class CategoryMappingTests(unittest.TestCase):
    def test_requested_gold_mappings(self):
        cases = {
            "text_block": "body_text", "text_mask": "body_text", "list_group": "body_text",
            "title": "title", "table": "table", "table_mask": "table",
            "table_caption": "table_caption", "table_footnote": "table_caption",
            "figure": "figure", "chart_mask": "figure",
            "figure_caption": "figure_caption", "figure_footnote": "figure_caption",
            "equation_isolated": "formula", "equation_semantic": "formula",
            "header": "header", "footer": "footer", "page_footnote": "footer",
            "page_number": "page_number", "reference": "reference",
        }
        for source, expected in cases.items():
            self.assertEqual(alignment.normalize_gold_category(source), expected)

    def test_requested_mineru_mappings(self):
        cases = {
            "text": "body_text", "list": "body_text",
            "paragraph_title": "title", "title": "title", "doc_title": "title",
            "table": "table", "table_title": "table_caption", "table_caption": "table_caption",
            "image": "figure", "figure": "figure", "chart": "figure",
            "figure_title": "figure_caption", "figure_caption": "figure_caption",
            "display_formula": "formula", "interline_equation": "formula",
            "header": "header", "footer": "footer", "number": "page_number",
            "page_number": "page_number", "reference": "reference",
        }
        for source, expected in cases.items():
            self.assertEqual(alignment.normalize_mineru_category(source), expected)

    def test_unmapped_internal_labels_are_excluded(self):
        self.assertIsNone(alignment.normalize_mineru_category("ocr_text"))
        self.assertIsNone(alignment.normalize_mineru_category("inline_formula"))
        self.assertIsNone(alignment.normalize_gold_category("abandon"))

    def test_matching_uses_unified_category(self):
        result = alignment.match_page(
            [{"label": "display_formula", "bbox": [0, 0, 100, 100]}],
            [{"category_type": "equation_isolated", "poly": [0, 0, 100, 0, 100, 100, 0, 100]}],
            0.5,
        )
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["matches"][0]["unified_category"], "formula")


if __name__ == "__main__":
    unittest.main()
