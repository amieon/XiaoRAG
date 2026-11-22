"""Layer-zero behavior: preserve evidence and refuse uncertain assembly."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doc_chunking.preprocess import preprocess_document


def block(ident, kind="paragraph", page=1, *, unit_index=None, text=None, metadata=None):
    unit_index = page if unit_index is None else unit_index
    return {"block_id": ident, "kind": kind, "native_kind": kind, "text": text or ident,
            "level": None, "unit_index": unit_index, "unit_id": f"doc-p{unit_index}",
            "unit_type": "physical_page", "page_number": page, "order": 0,
            "metadata": deepcopy(metadata or {}),
            "sources": [{"block_id": ident, "unit_id": f"doc-p{unit_index}", "unit_index": unit_index,
                         "unit_type": "physical_page", "page_number": page, "bbox": [1, 2, 3, 4],
                         "source_type": "native", "confidence": None, "native_id": f"raw-{ident}"}]}


def relation(left, right, *, kind="paragraph_continuation", decision="accept", **extra):
    result = {"relation_id": f"{left}-{right}", "relation_type": kind,
              "source_id": left, "target_id": right, "metadata": {}}
    if decision is not None:
        result["metadata"]["decision"] = decision
    result.update(deepcopy(extra))
    return result


def run(blocks, relations=(), *, native=None, config=None, pages=None):
    return preprocess_document(blocks, {"document_id": "doc", "relations": list(relations), "pages": pages or []},
                               native or {}, config or {})


class ChunkPreprocessTests(unittest.TestCase):
    def test_explicit_furniture_toc_and_notes_are_preserved_as_metadata(self):
        furniture = [block(kind, kind=kind, metadata={"security_marking": "internal", "file_number": "ref-7"})
                     for kind in ("header", "footer", "page_number", "watermark", "toc", "speaker_notes")]
        repeated = [block("body1", text="same sentence"), block("body2", page=2, text="same sentence")]
        inputs = furniture + repeated
        original = deepcopy(inputs)
        output = run(inputs)
        self.assertEqual([b["block_id"] for b in output["blocks"]], ["body1", "body2"])
        self.assertEqual(len(output["excluded"]), len(furniture))
        for excluded, source in zip(output["excluded"], furniture):
            self.assertEqual(excluded["text"], source["text"])
            self.assertEqual(excluded["metadata"], source["metadata"])
            self.assertEqual(excluded["sources"], source["sources"])
        output["excluded"][0]["metadata"]["security_marking"] = "changed"
        self.assertEqual(inputs, original)
        self.assertEqual(output["merges"], [])

    def test_explicit_toc_page_is_removed_but_a_contents_heading_is_not_guessed(self):
        toc = block("toc-line", text="Chapter One 3")
        title = block("body-title", kind="heading", page=2, text="Contents")
        pages = [{"page_id": toc["unit_id"], "metadata": {"is_toc": True}}]
        output = run([toc, title], pages=pages)
        self.assertEqual([b["block_id"] for b in output["blocks"]], ["body-title"])
        self.assertEqual(output["excluded"][0]["exclusion_reason"], "explicit_toc")

    def test_unordered_accepted_relations_assemble_a_chain_without_mutating_inputs(self):
        blocks = [block("a", metadata={"security": "secret", "fragments": ["upstream metadata"]}),
                  block("b", page=2), block("c", page=3)]
        relations = [relation("b", "c", decision=None, metadata={"status": "accepted"}),
                     relation("a", "b", decision=None, accepted=True)]
        original_blocks, original_relations = deepcopy(blocks), deepcopy(relations)
        output = run(blocks, relations)
        merged, = output["blocks"]
        self.assertEqual(merged["block_id"], "a")
        self.assertEqual(merged["text"], "a\nb\nc")
        self.assertEqual([s["block_id"] for s in merged["sources"]], ["a", "b", "c"])
        self.assertEqual(merged["metadata"]["fragments"], original_blocks)
        self.assertEqual(merged["metadata"]["security"], "secret")
        self.assertEqual([m["status"] for m in output["merges"]], ["merged", "merged"])
        merged["metadata"]["fragments"][0]["sources"][0]["bbox"][0] = 999
        self.assertEqual(blocks, original_blocks)
        self.assertEqual(relations, original_relations)

    def test_candidate_unknown_and_conflicting_acceptance_are_rejected(self):
        for edge in (relation("a", "b", decision="candidate", confidence=1.0),
                     relation("a", "b", decision=None, confidence=1.0),
                     relation("a", "b", decision="candidate", accepted=True),
                     relation("a", "b", accepted=False)):
            with self.subTest(edge=edge):
                output = run([block("a"), block("b", page=2)], [edge])
                self.assertEqual(len(output["blocks"]), 2)
                self.assertEqual(output["merges"][0]["reason"], "relation_not_explicitly_accepted")

    def test_missing_pages_and_nonphysical_units_are_never_joined(self):
        for left, right, expected in (
            (block("a"), block("b", page=3, unit_index=2), "nonconsecutive_physical_pages"),
            (block("a"), block("b", page=2, unit_index=3), "nonconsecutive_units"),
            (dict(block("a"), unit_type="slide"), dict(block("b", page=2), unit_type="slide"), "physical_pages_required"),
            (dict(block("a"), page_number=None), block("b", page=2), "nonconsecutive_physical_pages"),
        ):
            with self.subTest(expected=expected):
                output = run([left, right], [relation("a", "b")])
                self.assertEqual(len(output["blocks"]), 2)
                self.assertEqual(output["merges"][0]["reason"], expected)

    def test_foreign_document_and_intervening_content_are_not_crossed(self):
        output = run([block("a"), block("b", page=2, metadata={"document_id": "other"})], [relation("a", "b")])
        self.assertEqual(output["merges"][0]["reason"], "cross_document_relation")
        output = run([block("a"), block("b", page=2)], [relation("a", "b", target_document_id="other")])
        self.assertEqual(output["merges"][0]["reason"], "cross_document_relation")
        output = run([block("a"), block("heading", kind="heading", page=2), block("b", page=2)], [relation("a", "b")])
        self.assertEqual(output["merges"][0]["reason"], "intervening_body_block")
        self.assertEqual([b["block_id"] for b in output["blocks"]], ["a", "heading", "b"])

    def test_table_chain_removes_only_explicit_matching_repeated_headers(self):
        header = ["Name", "Value"]
        blocks = [block(ident, "table", page=page, metadata={"rows": [header, [ident, page]], "header_row_count": 1})
                  for page, ident in enumerate(("a", "b", "c"), 1)]
        original = deepcopy(blocks)
        output = run(blocks, [relation("b", "c", kind="cross_page_table"), relation("a", "b", kind="table_continuation")])
        merged, = output["blocks"]
        self.assertEqual(merged["metadata"]["rows"], [header, ["a", 1], ["b", 2], ["c", 3]])
        self.assertEqual(merged["metadata"]["row_count"], 4)
        self.assertEqual(merged["text"].count("Name"), 1)
        self.assertEqual([row["repeated_header_rows_removed"] for row in output["merges"]], [1, 1])
        self.assertEqual(merged["metadata"]["fragments"], original)
        self.assertEqual(blocks, original)

    def test_unverified_or_mismatched_table_headers_remain_in_content(self):
        for right_header, status, expected_warning in ((["Name"], "unverified", None),
                                                       (["Different"], "confirmed", "declared_repeated_header_mismatch_preserved")):
            with self.subTest(status=status):
                left = block("a", "table", metadata={"rows": [["Name"], ["A"]], "header_row_count": 1, "header_status": status})
                right = block("b", "table", page=2, metadata={"rows": [right_header, ["B"]], "header_row_count": 1, "header_status": status})
                output = run([left, right], [relation("a", "b", kind="cross_page_table")])
                self.assertEqual(len(output["blocks"][0]["metadata"]["rows"]), 4)
                self.assertEqual(output["merges"][0]["repeated_header_rows_removed"], 0)
                self.assertEqual(output["merges"][0].get("header_warning"), expected_warning)

    def test_native_table_rows_are_bound_by_native_id_and_explicit_relation_header(self):
        blocks = [block("a", "table"), block("b", "table", page=2)]
        native = {"tables": [{"raw_id": "raw-a", "rows": [["Header"], ["A"]]},
                              {"raw_id": "raw-b", "rows": [["Header"], ["B"]]}]}
        original_native = deepcopy(native)
        edge = relation("a", "b", kind="table_continuation", repeat_header_rows=1)
        output = run(blocks, [edge], native=native)
        self.assertEqual(output["blocks"][0]["metadata"]["rows"], [["Header"], ["A"], ["B"]])
        self.assertEqual(output["blocks"][0]["metadata"]["header_row_count"], 1)
        self.assertEqual(native, original_native)

    def test_unstructured_table_fragments_are_preserved_without_header_guessing(self):
        blocks = [block("a", "table", text="unknown table first"), block("b", "table", page=2, text="unknown table second")]
        output = run(blocks, [relation("a", "b", kind="cross_page_table")])
        self.assertEqual(output["blocks"][0]["text"], "unknown table first\nunknown table second")
        self.assertEqual(output["merges"][0]["header_handling"], "unstructured_fragments_preserved")
        self.assertEqual(output["blocks"][0]["metadata"]["fragments"], blocks)

    def test_horizontal_tables_and_conflicting_column_counts_stay_separate(self):
        for right_columns, extra, expected in (
            (2, {"metadata": {"status": "accepted", "orientation": "horizontal"}}, "horizontal_table_merge_not_supported"),
            (3, {}, "incompatible_table_columns"),
        ):
            with self.subTest(expected=expected):
                blocks = [block("a", "table", metadata={"column_count": 2}),
                          block("b", "table", page=2, metadata={"column_count": right_columns})]
                output = run(blocks, [relation("a", "b", kind="cross_page_table", **extra)])
                self.assertEqual(output["blocks"], blocks)
                self.assertEqual(output["merges"][0]["reason"], expected)

    def test_accepted_branches_are_not_resolved_by_picking_the_nearest_block(self):
        blocks = [block("a"), block("b", page=2), block("c", page=2)]
        output = run(blocks, [relation("a", "b"), relation("a", "c")])
        self.assertEqual(output["blocks"], blocks)
        self.assertEqual(output["merges"][0]["reason"], "ambiguous_relation_branch")
        duplicate = run(blocks[:2], [relation("a", "b"), relation("a", "b")])
        self.assertEqual(len(duplicate["blocks"]), 1)
        self.assertEqual([row["status"] for row in duplicate["merges"]], ["merged", "rejected"])
        self.assertEqual(duplicate["merges"][1]["reason"], "duplicate_relation_edge")

    def test_header_cell_types_must_match_before_removing_a_row(self):
        blocks = [block("a", "table", metadata={"rows": [[1], ["A"]], "header_row_count": 1}),
                  block("b", "table", page=2, metadata={"rows": [[True], ["B"]], "header_row_count": 1})]
        output = run(blocks, [relation("a", "b", kind="cross_page_table")])
        self.assertEqual(output["blocks"][0]["metadata"]["rows"], [[1], ["A"], [True], ["B"]])
        self.assertEqual(output["merges"][0]["repeated_header_rows_removed"], 0)

    def test_disabled_steps_and_invalid_relations_are_auditable(self):
        blocks = [block("header", "header", metadata={"excluded_from_body": True}),
                  block("a"), block("b", page=2), block("toc", "toc")]
        output = run(blocks, [relation("a", "b"), None], config={"preprocess": {"strip_furniture": False, "strip_toc": False, "merge_cross_page": False}})
        self.assertEqual(output["blocks"], blocks)
        self.assertEqual(output["excluded"], [])
        self.assertEqual(output["merges"][0]["reason"], "cross_page_merge_disabled")
        self.assertEqual(output["warnings"], ["invalid_relation_object_ignored"])


if __name__ == "__main__":
    unittest.main()
