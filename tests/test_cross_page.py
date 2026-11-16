import unittest

from doc_parser.cross_page import recover_cross_page_relations
from doc_parser.document_ir import Block, DocumentIR, Page, SourceRef


def block(block_id, page_id, kind, bbox, text, order):
    return Block(
        block_id=block_id,
        page_id=page_id,
        block_type=kind,
        bbox=list(bbox),
        text=text,
        reading_order=order,
        source=SourceRef(engine="test"),
    )


class CrossPageRecoveryTest(unittest.TestCase):
    def test_unfinished_paragraph_on_consecutive_pages_is_linked(self):
        pages = [
            Page("doc_p0001", 1, 1000, 1000, [block("b1", "doc_p0001", "paragraph", [100, 700, 900, 980], "This sentence continues", 0)]),
            Page("doc_p0002", 2, 1000, 1000, [block("b2", "doc_p0002", "paragraph", [100, 30, 900, 180], "on the next page.", 0)]),
        ]
        document = DocumentIR("doc", pages)
        recovered, stats = recover_cross_page_relations(document)
        self.assertEqual(stats.relation_counts["paragraph_continuation"], 1)
        self.assertEqual(recovered.relations[0].source_id, "b1")
        self.assertEqual(recovered.relations[0].target_id, "b2")
        self.assertEqual(recovered.relations[0].metadata["decision"], "accept")
        self.assertEqual(document.relations, [], "input document must not be mutated")

    def test_sparse_pages_are_not_linked(self):
        pages = [
            Page("doc_p0001", 1, 1000, 1000, [block("b1", "doc_p0001", "paragraph", [100, 700, 900, 980], "unfinished", 0)]),
            Page("doc_p0003", 3, 1000, 1000, [block("b3", "doc_p0003", "paragraph", [100, 30, 900, 180], "continuation", 0)]),
        ]
        recovered, stats = recover_cross_page_relations(DocumentIR("doc", pages))
        self.assertEqual(stats.adjacent_page_pairs, 0)
        self.assertEqual(recovered.relations, [])

    def test_aligned_tables_are_linked_as_candidates(self):
        pages = [
            Page("doc_p0001", 1, 1000, 1000, [block("t1", "doc_p0001", "table", [100, 600, 900, 980], "A B", 0)]),
            Page("doc_p0002", 2, 1000, 1000, [block("t2", "doc_p0002", "table", [100, 20, 900, 350], "A B", 0)]),
        ]
        recovered, stats = recover_cross_page_relations(DocumentIR("doc", pages))
        self.assertEqual(stats.relation_counts["cross_page_table"], 1)
        self.assertEqual(recovered.relations[0].relation_type, "cross_page_table")


if __name__ == "__main__":
    unittest.main()
