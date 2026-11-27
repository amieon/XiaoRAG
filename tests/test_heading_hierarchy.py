from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_docx_heading_dataset import build_docx_heading_dataset
from chapter_tree.heading_classifier import HeadingExample
from chapter_tree.heading_hierarchy import HeadingHierarchyResolver
from chapter_tree.zero_shot_classifier import OpenAICompatibleHeadingClassifier
from docx import Document


def example(position: int, text: str) -> HeadingExample:
    return HeadingExample(
        block_id=f"doc_p0001_b{position:04d}", page_id="doc_p0001", document_id="doc",
        text=text,
        context={"previous": {"text": f"前文{position}"}, "next": {"text": f"后文{position}"}},
        raw_label="text", reading_order=position, source_page_result="synthetic",
    )


def prediction(item: HeadingExample, *, level: int = 1, native_level: int | None = None):
    native = native_level is not None
    return {
        "block_id": item.block_id, "label": "heading", "heading_level": level,
        "native_heading_count": int(native),
        "native_evidence": ([{
            "source": "docx_style", "role": "heading", "heading_level": native_level,
        }] if native else []),
    }


def json_response(value) -> dict:
    return {"choices": [{"message": {"content": json.dumps(value, ensure_ascii=False)}}]}


def client_for(transport) -> OpenAICompatibleHeadingClassifier:
    return OpenAICompatibleHeadingClassifier(
        base_url="http://localhost/v1", model="fake-hierarchy-model", transport=transport,
    )


class HeadingHierarchyTests(unittest.TestCase):
    def test_none_flattens_native_numbered_and_model_headings_without_model_calls(self):
        items = [example(0, "原生标题"), example(1, "1.1 明确编号"), example(2, "普通标题")]
        rows = [
            prediction(items[0], level=4, native_level=3),
            prediction(items[1], level=5),
            prediction(items[2], level=6),
        ]

        def must_not_call(payload):
            raise AssertionError("none baseline must not call the hierarchy model")

        resolved = HeadingHierarchyResolver(client_for(must_not_call), mode="none").resolve(items, rows)
        self.assertEqual([row["heading_level"] for row in resolved], [1, 1, 1])
        self.assertEqual([row["hierarchy_parent_id"] for row in resolved], [None, None, None])
        self.assertEqual({row["hierarchy_source"] for row in resolved}, {"baseline_no_tree"})
        self.assertEqual({row["hierarchy_mode"] for row in resolved}, {"none"})

    def test_global_sends_full_sequence_and_derives_levels_from_parents(self):
        items = [example(index, text) for index, text in enumerate([
            "平台方案", "总体设计", "详细设计", "实施保障",
        ])]
        calls = []

        def transport(payload):
            user = json.loads(payload["messages"][1]["content"])
            calls.append(user)
            parents = {"h001": "h000", "h002": "h001", "h003": "h000"}
            return json_response({"predictions": [
                {"heading_id": heading_id, "parent_id": parent_id}
                for heading_id, parent_id in parents.items()
            ]})

        resolved = HeadingHierarchyResolver(client_for(transport), mode="global").resolve(
            items, [prediction(item, level=6) for item in items],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual([row["heading_id"] for row in calls[0]["headings"]], [
            "h000", "h001", "h002", "h003",
        ])
        self.assertEqual(calls[0]["query_heading_ids"], ["h001", "h002", "h003"])
        self.assertEqual([row["heading_level"] for row in resolved], [1, 2, 3, 2])
        self.assertEqual([row["hierarchy_parent_id"] for row in resolved], [
            None, items[0].block_id, items[1].block_id, items[0].block_id,
        ])
        self.assertEqual([row["hierarchy_source"] for row in resolved], [
            "document_root", "global_model", "global_model", "global_model",
        ])

    def test_streaming_uses_only_current_and_active_path_and_updates_stack(self):
        items = [example(index, text) for index, text in enumerate([
            "平台方案", "总体设计", "详细设计", "接口设计",
        ])]
        calls = []
        parents = {"总体设计": "h000", "详细设计": "h000", "接口设计": "h002"}

        def transport(payload):
            user = json.loads(payload["messages"][1]["content"])
            calls.append(user)
            return json_response({"parent_id": parents[user["current"]["text"]]})

        resolved = HeadingHierarchyResolver(client_for(transport), mode="streaming").resolve(
            items, [prediction(item, level=6) for item in items],
        )
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(set(call) == {"current", "active_path"} for call in calls))
        self.assertTrue(all(set(call["current"]) == {"text"} for call in calls))
        self.assertEqual([node["heading_id"] for node in calls[1]["active_path"]], ["root", "h000", "h001"])
        # h002 与 h001 同级并挂到 h000，因此 h001 已关闭，不在下一次活动路径中。
        self.assertEqual([node["heading_id"] for node in calls[2]["active_path"]], ["root", "h000", "h002"])
        self.assertEqual([row["heading_level"] for row in resolved], [1, 2, 2, 3])
        self.assertEqual([row["hierarchy_parent_id"] for row in resolved], [
            None, items[0].block_id, items[0].block_id, items[2].block_id,
        ])

    def test_streaming_never_exposes_current_local_id_as_a_parent_candidate(self):
        items = [example(0, "总论"), example(1, "承诺人：")]
        calls = []

        def transport(payload):
            calls.append(payload)
            return json_response({"parent_id": "h000"})

        resolved = HeadingHierarchyResolver(client_for(transport), mode="streaming").resolve(
            items, [prediction(item) for item in items],
        )
        self.assertEqual(len(calls), 1)
        self.assertNotIn("h001", json.dumps(calls[0]["messages"], ensure_ascii=False))
        self.assertEqual(resolved[1]["hierarchy_parent_id"], items[0].block_id)

    def test_streaming_context_adds_local_text_and_recent_closed_headings(self):
        items = [example(index, text) for index, text in enumerate([
            "平台方案", "总体设计", "详细设计", "接口设计",
        ])]
        calls = []
        parents = {"总体设计": "h000", "详细设计": "h000", "接口设计": "h002"}

        def transport(payload):
            user = json.loads(payload["messages"][1]["content"])
            calls.append(user)
            return json_response({"parent_id": parents[user["current"]["text"]]})

        HeadingHierarchyResolver(client_for(transport), mode="streaming-context").resolve(
            items, [prediction(item) for item in items],
        )
        self.assertEqual(calls[1]["previous_text"], "前文2")
        self.assertEqual(calls[1]["next_text"], "后文2")
        # h001 在 h002 完成挂载后才关闭，因此从下一次请求开始可见。
        self.assertEqual(calls[1]["recent_closed_siblings"], [])
        self.assertEqual(calls[2]["recent_closed_siblings"], [
            {"heading_id": "h001", "text": "总体设计"},
        ])
        self.assertTrue(all(set(call) == {
            "current", "active_path", "previous_text", "next_text", "recent_closed_siblings",
        } for call in calls))

    def test_native_and_explicit_numbering_anchors_do_not_call_hierarchy_model(self):
        items = [example(index, text) for index, text in enumerate([
            "原生总论", "模型子节", "第二章 明确编号", "模型新子节",
        ])]
        rows = [
            prediction(items[0], native_level=1), prediction(items[1]),
            prediction(items[2]), prediction(items[3]),
        ]
        calls = []
        parents = {"模型子节": "h000", "模型新子节": "h002"}

        def transport(payload):
            user = json.loads(payload["messages"][1]["content"])
            calls.append(user)
            return json_response({"parent_id": parents[user["current"]["text"]]})

        resolved = HeadingHierarchyResolver(client_for(transport), mode="streaming").resolve(items, rows)
        self.assertEqual(len(calls), 2)
        self.assertEqual([row["heading_level"] for row in resolved], [1, 2, 1, 2])
        self.assertEqual([row["hierarchy_source"] for row in resolved], [
            "native", "streaming_model", "explicit_numbering", "streaming_model",
        ])
        self.assertEqual([row["hierarchy_parent_id"] for row in resolved], [
            None, items[0].block_id, None, items[2].block_id,
        ])

    def test_global_rejects_future_unknown_duplicate_and_missing_rows(self):
        items = [example(index, text) for index, text in enumerate(["总论", "设计", "实施"])]
        cases = {
            "future": ([
                {"heading_id": "h001", "parent_id": "h002"},
                {"heading_id": "h002", "parent_id": None},
            ], "must precede"),
            "unknown_parent": ([
                {"heading_id": "h001", "parent_id": "h999"},
                {"heading_id": "h002", "parent_id": None},
            ], "Unknown global parent_id"),
            "duplicate": ([
                {"heading_id": "h001", "parent_id": None},
                {"heading_id": "h001", "parent_id": None},
            ], "Unknown or duplicate"),
            "missing": ([
                {"heading_id": "h001", "parent_id": None},
            ], "missing headings"),
        }
        for name, (response, message) in cases.items():
            with self.subTest(name=name):
                resolver = HeadingHierarchyResolver(
                    client_for(lambda payload, value=response: json_response({"predictions": value})),
                    mode="global",
                )
                with self.assertRaisesRegex(ValueError, message):
                    resolver.resolve(items, [prediction(item) for item in items])

    def test_streaming_rejects_self_and_closed_parent(self):
        items = [example(index, text) for index, text in enumerate(["总论", "设计", "实施"])]
        resolver = HeadingHierarchyResolver(
            client_for(lambda payload: json_response({"parent_id": "h001"})), mode="streaming",
        )
        with self.assertRaisesRegex(ValueError, "not in active path"):
            resolver.resolve(items[:2], [prediction(item) for item in items[:2]])

        parents = iter(["root"])

        def closed_parent(payload):
            return json_response({"parent_id": next(parents, "h000")})

        resolver = HeadingHierarchyResolver(client_for(closed_parent), mode="streaming")
        with self.assertRaisesRegex(ValueError, "not in active path"):
            resolver.resolve(items, [prediction(item) for item in items])

    def test_streaming_retries_an_invalid_example_id_with_explicit_candidates(self):
        items = [example(index, text) for index, text in enumerate(["总论", "子节"])]
        calls = []

        def transport(payload):
            calls.append(payload)
            return json_response({"parent_id": "h003" if len(calls) == 1 else "h000"})

        resolved = HeadingHierarchyResolver(client_for(transport), mode="streaming").resolve(
            items, [prediction(item) for item in items],
        )
        self.assertEqual(len(calls), 2)
        self.assertNotIn("h003", calls[0]["messages"][0]["content"])
        correction = json.loads(calls[1]["messages"][-1]["content"])
        self.assertEqual(correction["allowed_ids"], ["root", "h000"])
        self.assertNotIn("current_heading_id", correction)
        self.assertNotIn("output", correction)
        self.assertEqual(resolved[1]["hierarchy_parent_id"], items[0].block_id)

    def test_streaming_persistent_invalid_parent_fails_with_current_block_and_candidates(self):
        items = [example(index, text) for index, text in enumerate(["总论", "子节"])]
        calls = []

        def transport(payload):
            calls.append(payload)
            return json_response({"parent_id": "h003"})

        resolver = HeadingHierarchyResolver(client_for(transport), mode="streaming", max_retries=1)
        with self.assertRaisesRegex(ValueError, "doc_p0001_b0001.*allowed.*h000"):
            resolver.resolve(items, [prediction(item) for item in items])
        self.assertEqual(len(calls), 2)

    def test_none_dataset_and_downstream_cli_replay_stay_flat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "demo.docx"
            document = Document()
            document.add_paragraph("原生总论", style="Heading 2")
            document.add_paragraph("正文一。")
            document.add_paragraph("模型标题")
            document.add_paragraph("正文二。")
            document.save(source)

            def transport(payload):
                user = json.loads(payload["messages"][1]["content"])
                self.assertIn("blocks", user)
                return json_response({"predictions": [
                    {
                        "block_id": item["block_id"],
                        "label": "heading" if item["text"] == "模型标题" else "body",
                        "heading_level": 4 if item["text"] == "模型标题" else None,
                    } for item in user["blocks"]
                ]})

            output = root / "none-exp01"
            manifest = build_docx_heading_dataset(
                source, output, classifier=client_for(transport), hierarchy_mode="none",
            )
            document_output = output / "documents/demo"
            tree = json.loads((document_output / "chapter-tree.json").read_text(encoding="utf-8"))
            self.assertEqual(tree["stats"]["max_depth"], 1)
            self.assertTrue(all(node["level"] == 1 for node in tree["nodes"][1:]))
            self.assertEqual(manifest["documents"][0]["hierarchy_model_decision_count"], 0)
            styled = Document(output / manifest["documents"][0]["styled_docx"])
            self.assertEqual(styled.paragraphs[0].style.name, "Heading 1")
            self.assertEqual(styled.paragraphs[2].style.name, "Heading 1")

            common = [
                "--input", str(document_output / "document.json"),
                "--predictions", str(document_output / "heading-predictions.jsonl"),
                "--mode", "hybrid", "--flat-headings",
            ]
            rebuilt = root / "tree-none-exp01"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/build_chapter_tree.py"),
                *common, "--output", str(rebuilt),
            ], check=True, capture_output=True, text=True)
            rebuilt_tree = json.loads((rebuilt / "chapter-tree.json").read_text(encoding="utf-8"))
            self.assertEqual(rebuilt_tree["stats"]["max_depth"], 1)

            chunks = root / "chunks-none-exp01"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/build_chunks.py"),
                *common, "--output", str(chunks),
            ], check=True, capture_output=True, text=True)
            chunk_tree = json.loads((chunks / "chapter-tree.json").read_text(encoding="utf-8"))
            self.assertEqual(chunk_tree["stats"]["max_depth"], 1)
            self.assertTrue(all(node["level"] == 1 for node in chunk_tree["nodes"][1:]))


if __name__ == "__main__":
    unittest.main()
