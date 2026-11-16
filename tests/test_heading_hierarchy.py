import copy
import unittest

from doc_parser.heading_hierarchy import enhance_middle_json, generate_outputs, parse_numbering


def block(text, *, kind="title", level=2, y=20, height=16, x=80):
    return {
        "type": kind,
        "level": level,
        "bbox": [x, y, 600, y + height],
        "lines": [
            {
                "bbox": [x, y, 600, y + height],
                "spans": [
                    {
                        "type": "text",
                        "bbox": [x, y, 600, y + height],
                        "content": text,
                    }
                ],
            }
        ],
    }


def middle(blocks):
    return {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [1000, 1400],
                "para_blocks": blocks,
                "discarded_blocks": [],
            }
        ],
        "_backend": "pipeline",
    }


class NumberingTest(unittest.TestCase):
    def test_decimal_depth(self):
        self.assertEqual(parse_numbering("2 Experimental Setup")["depth"], 1)
        self.assertEqual(parse_numbering("2.1 Experimental Setup")["depth"], 2)
        self.assertEqual(parse_numbering("2.1.3 Reactor")["depth"], 3)

    def test_chinese_patterns(self):
        self.assertEqual(parse_numbering("第一章 总则")["depth"], 1)
        self.assertEqual(parse_numbering("第一节 范围")["depth"], 2)
        self.assertEqual(parse_numbering("一、目标")["depth"], 1)
        self.assertEqual(parse_numbering("（一）范围")["depth"], 2)
        self.assertEqual(parse_numbering("（1）任务")["depth"], 3)

    def test_plain_decimal_value_is_not_numbering(self):
        self.assertIsNone(parse_numbering("3.14"))
        self.assertIsNone(parse_numbering("2025.10"))


class ResolutionTest(unittest.TestCase):
    def test_numbering_sets_levels_and_preserves_original(self):
        source = middle(
            [
                block("2 Experimental Setup", height=18),
                block("2.1 Reactor", height=14, y=60),
                block("2.2 Sewage", height=14, y=100),
            ]
        )
        original = copy.deepcopy(source)
        enhanced, inventory, registry, tree = enhance_middle_json(source)
        levels = [item["level"] for item in enhanced["pdf_info"][0]["para_blocks"]]
        self.assertEqual(levels, [2, 3, 3])
        self.assertEqual(source, original, "input must not be mutated")
        self.assertEqual(inventory["headings"][1]["resolution"]["method"], "numbering")
        self.assertEqual(inventory["headings"][1]["original_level"], 2)
        self.assertEqual({item["resolved_level"] for item in registry}, {2, 3})
        self.assertEqual(tree[0]["children"][0]["text"], "2.1 Reactor")
        content, content_v2, markdown = generate_outputs(enhanced)
        title_items = [item for item in content if "text_level" in item]
        self.assertEqual([item["text_level"] for item in title_items], [2, 3, 3])
        v2_titles = [item for item in content_v2[0] if item["type"] == "title"]
        self.assertEqual([item["content"]["level"] for item in v2_titles], [2, 3, 3])
        self.assertIn("### 2.1 Reactor", markdown)

    def test_unnumbered_heading_uses_numbered_style_anchor(self):
        source = middle(
            [
                block("body", kind="text", height=10, y=5),
                block("2 Experimental Setup", height=18, y=40),
                block("2.1 Reactor", height=13, y=80),
                block("Conclusion", height=18, y=120),
            ]
        )
        enhanced, inventory, _, _ = enhance_middle_json(source)
        conclusion = inventory["headings"][-1]
        self.assertEqual(conclusion["resolution"]["resolved_level"], 2)
        self.assertEqual(conclusion["resolution"]["method"], "document_style_mapping")
        self.assertEqual(enhanced["pdf_info"][0]["para_blocks"][-1]["level"], 2)

    def test_unnumbered_without_anchor_keeps_mineru_default(self):
        source = middle([block("Background", level=4)])
        _, inventory, registry, _ = enhance_middle_json(source)
        self.assertEqual(registry, [])
        result = inventory["headings"][0]["resolution"]
        self.assertEqual(result["resolved_level"], 4)
        self.assertEqual(result["method"], "mineru_default")
        self.assertIn("insufficient_heading_evidence", result["warnings"])

    def test_page_style_clustering_creates_two_levels(self):
        source = middle([
            block("body", kind="text", height=10, y=5),
            block("Document Section", height=28, y=30),
            block("First Subsection", height=15, y=70),
            block("Second Subsection", height=14, y=110),
        ])
        enhanced, inventory, registry, tree = enhance_middle_json(source)
        levels = [x["resolution"]["resolved_level"] for x in inventory["headings"]]
        self.assertEqual(levels, [2, 3, 3])
        self.assertTrue(all(x["resolution"]["method"] == "page_style_clustering" for x in inventory["headings"]))
        self.assertEqual(len(registry), 2)
        self.assertEqual(len(tree[0]["children"]), 2)
        self.assertEqual([b["level"] for b in enhanced["pdf_info"][0]["para_blocks"] if b["type"] == "title"], levels)

    def test_single_line_style_uses_layout_height_not_noisy_ocr_height(self):
        parent = block("Parent", height=28, y=20)
        topic1 = block("Topic 1", height=14, y=70)
        topic2 = block("Topic 2", height=14, y=110)
        # Simulate OCR boxes inconsistently absorbing a decorative underline.
        topic1["lines"][0]["bbox"] = [80, 72, 600, 81]  # OCR height 9
        topic2["lines"][0]["bbox"] = [80, 109, 600, 121]  # OCR height 12
        source = middle([block("body", kind="text", height=10, y=5), parent, topic1, topic2])
        _, inventory, _, _ = enhance_middle_json(source)
        visual = [x["visual"] for x in inventory["headings"]]
        self.assertEqual([x["line_height"] for x in visual], [28.0, 14.0, 14.0])
        self.assertEqual([x["ocr_line_height"] for x in visual], [28.0, 9.0, 12.0])
        self.assertTrue(all(x["font_height_source"] == "layout_block_bbox" for x in visual))
        self.assertEqual([x["resolution"]["resolved_level"] for x in inventory["headings"]], [2, 3, 3])

    def test_adjacent_table_title_is_excluded_from_tree(self):
        table = {"type": "table", "bbox": [80, 100, 800, 300], "blocks": []}
        source = middle([
            block("Report Section", height=28, y=20),
            block("Ratings", height=14, y=80),
            table,
            block("Next Section", height=14, y=340),
        ])
        enhanced, inventory, _, tree = enhance_middle_json(source)
        rating = inventory["headings"][1]
        self.assertEqual(rating["role_resolution"]["resolved_role"], "table_title")
        self.assertEqual(rating["resolution"]["resolved_level"], 0)
        self.assertNotIn("Ratings", str(tree))
        content, content_v2, markdown = generate_outputs(enhanced)
        self.assertNotIn("## Ratings", markdown)
        self.assertIn("Ratings", markdown)
        self.assertTrue(any(x.get("text") == "Ratings" and "text_level" not in x for x in content))
        self.assertTrue(any(x["type"] == "paragraph" for x in content_v2[0]))


if __name__ == "__main__":
    unittest.main()
