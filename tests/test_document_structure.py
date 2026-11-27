from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import classify_headings
from chapter_tree.heading_classifier import HeadingExample
from chapter_tree.heading_hierarchy import predict_document_structure
from chapter_tree.tree import build_chapter_tree


def example(block_id: str, document_id: str, text: str, raw_label: str) -> HeadingExample:
    return HeadingExample(
        block_id=block_id,
        page_id=f"{document_id}_p0001",
        document_id=document_id,
        text=text,
        context={},
        raw_label=raw_label,
        reading_order=0,
        source_page_result="fake/page_result.json",
    )


class FakeTwoStageClassifier:
    model = "fake-qwen"
    max_tokens = 4096
    use_response_format = False
    disable_thinking = True

    def __init__(self):
        self.hierarchy_calls = []

    def predict(self, examples, *, batch_size=8):
        return [{
            "block_id": item.block_id,
            "page_id": item.page_id,
            "document_id": item.document_id,
            "reading_order": item.reading_order,
            "label": "heading" if item.raw_label == "paragraph_title" else "body",
            "heading_level": 1 if item.raw_label == "paragraph_title" else None,
            "native_heading_count": 0,
            "native_evidence": [],
            "mode": "zero_shot_llm",
            "model": self.model,
        } for item in examples]

    def _post(self, payload):
        user = payload["messages"][1]["content"]
        import json
        current = json.loads(user)["current"]
        self.hierarchy_calls.append(current["text"])
        return {"choices": [{"message": {"content": '{"parent_id":"h000"}'}}]}


class DocumentStructureTests(unittest.TestCase):
    def test_same_classifier_runs_two_stages_with_document_isolation(self):
        items = [
            example("a0", "doc_a", "第一章 总则", "paragraph_title"),
            example("b0", "doc_b", "引言", "paragraph_title"),
            example("a1", "doc_a", "研究背景", "paragraph_title"),
            example("b1", "doc_b", "1.1 方法", "paragraph_title"),
            example("a2", "doc_a", "这是一段正文。", "text"),
        ]
        classifier = FakeTwoStageClassifier()
        rows = predict_document_structure(
            items, classifier, hierarchy_mode="streaming",
        )
        self.assertEqual([row["block_id"] for row in rows], ["a0", "b0", "a1", "b1", "a2"])
        by_id = {row["block_id"]: row for row in rows}
        self.assertEqual(by_id["a0"]["hierarchy_source"], "explicit_numbering")
        self.assertEqual(by_id["a1"]["heading_level"], 2)
        self.assertEqual(by_id["a1"]["hierarchy_parent_id"], "a0")
        self.assertEqual(by_id["a1"]["hierarchy_source"], "streaming_model")
        self.assertEqual(by_id["b0"]["heading_level"], 1)
        self.assertEqual(by_id["b0"]["hierarchy_source"], "document_root")
        self.assertEqual(by_id["b1"]["heading_level"], 2)
        self.assertEqual(by_id["b1"]["hierarchy_parent_id"], "b0")
        self.assertEqual(classifier.hierarchy_calls, ["研究背景"])

    def test_batch_predictions_are_filtered_for_current_document_tree(self):
        blocks = [{"block_id": "a0", "kind": "paragraph", "text": "正文", "metadata": {}}]
        predictions = [
            {"block_id": "b0", "document_id": "doc_b", "label": "heading", "heading_level": 1},
            {"block_id": "a0", "document_id": "doc_a", "label": "body", "heading_level": None},
        ]
        tree = build_chapter_tree(blocks, document_id="doc_a", predictions=predictions)
        self.assertEqual(tree["stats"]["heading_block_count"], 0)
        self.assertEqual(tree["stats"]["flat_document"], True)

    def test_cli_resume_skips_documents_already_saved_in_the_checkpoint(self):
        items = [
            example("a0", "doc_a", "已完成文档", "paragraph_title"),
            example("b0", "doc_b", "待处理文档", "paragraph_title"),
        ]

        class FakeClassifier:
            def __init__(self):
                self.seen_documents = []

            def predict(self, examples, *, batch_size=8):
                self.seen_documents.append({item.document_id for item in examples})
                return [{
                    "block_id": item.block_id,
                    "page_id": item.page_id,
                    "document_id": item.document_id,
                    "reading_order": item.reading_order,
                    "label": "heading",
                    "heading_level": 1,
                    "native_heading_count": 0,
                    "native_evidence": [],
                } for item in examples]

        classifier = FakeClassifier()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            completed = {
                "block_id": "a0", "document_id": "doc_a", "label": "heading",
                "heading_level": 1, "hierarchy_source": "document_root",
            }
            output.write_text(json.dumps(completed) + "\n", encoding="utf-8")
            arguments = [
                "classify_headings.py", "--input", "unused", "--mode", "zero-shot",
                "--local-model", "fake", "--hierarchy-mode", "none", "--resume",
                "--output", str(output),
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(classify_headings, "load_heading_examples", return_value=items),
                mock.patch.object(classify_headings, "LocalHFHeadingClassifier", return_value=classifier),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                classify_headings.main()
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(classifier.seen_documents, [{"doc_b"}])
        self.assertEqual([row["document_id"] for row in rows], ["doc_a", "doc_b"])

    def test_legacy_prediction_without_document_id_still_rejects_unknown_blocks(self):
        blocks = [{"block_id": "a0", "kind": "paragraph", "text": "正文", "metadata": {}}]
        with self.assertRaisesRegex(ValueError, "predictions_reference_unknown_blocks"):
            build_chapter_tree(
                blocks,
                document_id="doc_a",
                predictions=[{"block_id": "b0", "label": "heading", "heading_level": 1}],
            )


if __name__ == "__main__":
    unittest.main()
