from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chapter_tree.context import build_crop_context
from chapter_tree.heading_classifier import (
    HeadingExample,
    TransformerHeadingClassifier,
    binary_metrics,
    format_model_input,
    load_heading_examples,
    split_examples_by_document,
    train_transformer,
)
from chapter_tree.native_classifier import NativeHeadingClassifier
from chapter_tree.reading_order import upstream_reading_order_metadata


def example(document_id: str, label: str = "body", text: str = "当前") -> HeadingExample:
    return HeadingExample(
        block_id=f"{document_id}_p0001_b0000",
        page_id=f"{document_id}_p0001", document_id=document_id,
        text=text,
        context={
            "previous": {"text": "前文"},
            "next": {"text": "后文"},
        },
        raw_label="text", reading_order=0,
        source_page_result="synthetic", label=label,
    )


class ChapterTreeTests(unittest.TestCase):
    def test_context_uses_nearest_original_text_and_audit_offsets(self):
        regions = [SimpleNamespace(index=index, label="text") for index in range(3)]
        context = build_crop_context(
            regions, ["abcdefg", "当前", "hijklmn"], 1,
            page_id="doc_p0001", max_chars=3,
        )
        self.assertEqual(context["previous"]["text"], "efg")
        self.assertEqual(context["previous"]["char_start"], 4)
        self.assertEqual(context["previous"]["char_end"], 7)
        self.assertTrue(context["previous"]["truncated"])
        self.assertEqual(context["next"]["text"], "hij")
        self.assertEqual(context["next"]["block_id"], "doc_p0001_b0002")
        self.assertIsNone(build_crop_context(
            regions, ["a", "b", "c"], 0, page_id="doc_p0001", max_chars=3,
        )["previous"])

    def test_reading_order_metadata_does_not_reorder_regions(self):
        regions = [SimpleNamespace(index=3), SimpleNamespace(index=7)]
        metadata = upstream_reading_order_metadata(regions, page_id="doc_p0001")
        self.assertEqual(metadata["status"], "model_provided")
        self.assertEqual(
            metadata["ordered_block_ids"],
            ["doc_p0001_b0003", "doc_p0001_b0007"],
        )
        self.assertEqual([region.index for region in regions], [3, 7])

    def test_model_input_puts_current_crop_before_context(self):
        text = format_model_input(example("doc", text="短标题"))
        self.assertTrue(text.startswith("[当前块]\n短标题\n[前一块]\n前文"))
        self.assertIn("[后一块]\n后文", text)

    def test_page_result_loader_joins_separate_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "page": {
                    "page_id": "doc_demo_p0001",
                    "blocks": [
                        {
                            "block_id": "doc_demo_p0001_b0000", "page_id": "doc_demo_p0001",
                            "text": "第一章", "reading_order": 0, "block_type": "paragraph",
                            "metadata": {
                                "raw_label": "text", "context": {"next": {"text": "正文"}},
                                "native_structures": [{
                                    "source": "docx_style", "role": "heading", "heading_level": 1,
                                }],
                            },
                        },
                        {
                            "block_id": "doc_demo_p0001_b0001", "page_id": "doc_demo_p0001",
                            "text": "正文", "reading_order": 1, "block_type": "paragraph",
                            "metadata": {"raw_label": "text", "context": {"previous": {"text": "第一章"}}},
                        },
                    ],
                },
                "routes": [],
            }
            (root / "page_result.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8",
            )
            labels = root / "labels.jsonl"
            labels.write_text(
                '{"block_id":"doc_demo_p0001_b0000","label":"section_heading"}\n'
                '{"block_id":"doc_demo_p0001_b0001","label":"paragraph"}\n',
                encoding="utf-8",
            )
            examples = load_heading_examples(root, labels, require_labels=True)
            self.assertEqual([item.label for item in examples], ["heading", "body"])
            self.assertEqual({item.document_id for item in examples}, {"doc_demo"})
            self.assertEqual(examples[0].context["next"]["text"], "正文")
            self.assertEqual(examples[0].native_structures[0]["heading_level"], 1)

    def test_split_is_deterministic_and_document_disjoint(self):
        examples = [
            item for document_index in range(5)
            for item in (
                example(f"doc_{document_index}", "heading"),
                HeadingExample(**{
                    **example(f"doc_{document_index}", "body").__dict__,
                    "block_id": f"doc_{document_index}_p0001_b0001",
                }),
            )
        ]
        train_a, val_a = split_examples_by_document(examples, val_ratio=0.4, seed=7)
        train_b, val_b = split_examples_by_document(examples, val_ratio=0.4, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertTrue({item.document_id for item in train_a}.isdisjoint(
            item.document_id for item in val_a
        ))
        self.assertEqual(len(val_a), 4)

    def test_native_mode_trusts_docx_evidence_without_model_probabilities(self):
        heading = example("doc_native", "body", "第一章")
        heading.native_structures.append({
            "source": "docx_style", "role": "heading", "heading_level": 1,
            "native_id": "paragraph/0", "text": "第一章",
        })
        body = HeadingExample(**{
            **example("doc_native", "body", "普通正文").__dict__,
            "block_id": "doc_native_p0001_b0001",
        })
        predictions = NativeHeadingClassifier().predict([heading, body])
        self.assertEqual([item["label"] for item in predictions], ["heading", "body"])
        self.assertEqual(predictions[0]["heading_level"], 1)
        self.assertEqual(predictions[0]["decision_source"], "docx_native_trust")
        self.assertIsNone(predictions[0]["confidence"])
        self.assertIsNone(predictions[0]["scores"])

    def test_metrics_and_training_guards_do_not_need_ml_dependencies(self):
        metrics = binary_metrics([0, 0, 1, 1], [0, 1, 1, 0])
        self.assertEqual(metrics["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(metrics["accuracy"], 0.5)
        with self.assertRaisesRegex(ValueError, "both body and heading"):
            train_transformer(
                [example("doc", "body")], [], base_model="unused", output="unused",
            )
        with self.assertRaisesRegex(ValueError, "model directory"):
            TransformerHeadingClassifier("/definitely/not/a/model")


if __name__ == "__main__":
    unittest.main()
