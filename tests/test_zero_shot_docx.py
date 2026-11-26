from __future__ import annotations

import json
import os
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
from chapter_tree.zero_shot_classifier import OpenAICompatibleHeadingClassifier
from doc_parser.docx_native_ir import native_docx_to_ir
from docx import Document


def example(block_id: str, text: str, *, native: bool = False) -> HeadingExample:
    structures = [{
        "source": "docx_style", "role": "heading", "heading_level": 1,
    }] if native else []
    return HeadingExample(
        block_id=block_id, page_id="doc_p0001", document_id="doc", text=text,
        context={"previous": {"text": "前文"}, "next": {"text": "后文"}},
        raw_label="text", reading_order=0, source_page_result="synthetic",
        native_structures=structures,
    )


def response_for(payload: dict) -> dict:
    user = json.loads(payload["messages"][1]["content"])
    if "current" in user:
        # 流式层级请求：测试替身总是挂到当前活动路径末尾。
        parent_id = user["active_path"][-1]["heading_id"]
        return {"choices": [{"message": {"content": json.dumps({"parent_id": parent_id})}}]}
    items = user["blocks"]
    rows = []
    for item in items:
        heading = item["text"] in {"系统建设方案", "总体架构", "无编号标题"}
        rows.append({
            "block_id": item["block_id"],
            "label": "heading" if heading else "body",
            "heading_level": (1 if item["text"] != "总体架构" else 2) if heading else None,
        })
    return {"choices": [{"message": {"content": json.dumps({"predictions": rows}, ensure_ascii=False)}}]}


class ZeroShotHeadingTests(unittest.TestCase):
    def test_native_heading_is_not_sent_to_model_and_has_no_fake_confidence(self):
        calls = []

        def transport(payload):
            calls.append(payload)
            return response_for(payload)

        classifier = OpenAICompatibleHeadingClassifier(
            base_url="http://localhost:8000/v1", model="fake-qwen", transport=transport,
        )
        rows = classifier.predict([
            example("doc_p0001_b0000", "已有标题", native=True),
            example("doc_p0001_b0001", "无编号标题"),
            example("doc_p0001_b0002", "这是完整正文。"),
        ])
        self.assertEqual(len(calls), 1)
        sent = json.loads(calls[0]["messages"][1]["content"])["blocks"]
        self.assertEqual([item["block_id"] for item in sent], ["b0", "b1"])
        self.assertEqual([row["block_id"] for row in rows], [
            "doc_p0001_b0000", "doc_p0001_b0001", "doc_p0001_b0002",
        ])
        self.assertEqual(rows[0]["decision_source"], "docx_native_trust")
        self.assertEqual(rows[1]["heading_level"], 1)
        self.assertIsNone(rows[1]["confidence"])
        self.assertEqual(rows[2]["label"], "body")

    def test_invalid_or_missing_model_ids_fail_closed(self):
        def missing(payload):
            return {"choices": [{"message": {"content": '{"predictions":[]}'}}]}

        classifier = OpenAICompatibleHeadingClassifier(
            base_url="http://localhost/v1", model="fake", transport=missing,
        )
        with self.assertRaisesRegex(ValueError, "every input block"):
            classifier.predict([example("doc_p0001_b0000", "无编号标题")])

        def unknown(payload):
            return {"choices": [{"message": {"content": json.dumps({"predictions": [{
                "block_id": "invented", "label": "heading", "heading_level": 1,
            }]})}}]}

        classifier.transport = unknown
        with self.assertRaisesRegex(ValueError, "unknown block_id"):
            classifier.predict([example("doc_p0001_b0000", "无编号标题")])

    def test_reasoning_fallback_accepts_only_valid_prediction_json(self):
        def reasoning_only(payload):
            self.assertEqual(payload["max_tokens"], 8192)
            self.assertFalse(payload["enable_thinking"])
            self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
            response = response_for(payload)
            return {
                "choices": [{"finish_reason": "stop", "message": {
                    "content": "",
                    "reasoning_content": response["choices"][0]["message"]["content"],
                }}],
                "usage": {"completion_tokens": 66},
            }

        classifier = OpenAICompatibleHeadingClassifier(
            base_url="http://localhost/v1", model="fake-reasoner",
            max_tokens=8192, disable_thinking=True, transport=reasoning_only,
        )
        rows = classifier.predict([example("doc_p0001_b0000", "无编号标题")])
        self.assertEqual(rows[0]["block_id"], "doc_p0001_b0000")
        self.assertEqual(rows[0]["label"], "heading")

        classifier.transport = lambda payload: {
            "choices": [{"finish_reason": "stop", "message": {
                "content": "", "reasoning_content": "这里只是思考，不是最终JSON",
            }}],
        }
        with self.assertRaisesRegex(ValueError, "did not return valid JSON"):
            classifier.predict([example("doc_p0001_b0000", "无编号标题")])

    def test_response_format_can_be_disabled(self):
        def transport(payload):
            self.assertNotIn("response_format", payload)
            return response_for(payload)

        classifier = OpenAICompatibleHeadingClassifier(
            base_url="http://localhost/v1", model="fake-qwen",
            use_response_format=False, transport=transport,
        )
        rows = classifier.predict([example("doc_p0001_b0000", "无编号标题")])
        self.assertEqual(rows[0]["label"], "heading")

    def test_end_to_end_docx_dataset_builds_tree_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "demo.docx"
            document = Document()
            document.add_paragraph("系统建设方案")
            document.add_paragraph("本文说明系统建设范围和主要功能。")
            document.add_paragraph("第一章 已有结构", style="Heading 1")
            document.add_paragraph("这一节已经有原生标题样式。")
            document.add_paragraph("总体架构")
            document.add_paragraph("平台由解析、检索和版本管理模块组成。")
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "模块"
            table.cell(0, 1).text = "功能"
            document.save(source)

            classifier = OpenAICompatibleHeadingClassifier(
                base_url="http://localhost/v1", model="fake-qwen", transport=response_for,
            )
            output = root / "styled-exp01"
            manifest = build_docx_heading_dataset(source, output, classifier=classifier)
            self.assertEqual(manifest["document_count"], 1)
            self.assertFalse(manifest["ocr_required"])
            self.assertFalse(manifest["source_files_modified"])
            self.assertEqual(manifest["documents"][0]["native_heading_count"], 1)
            self.assertEqual(manifest["documents"][0]["model_heading_count"], 2)
            self.assertEqual(manifest["documents"][0]["tree_section_count"], 3)
            self.assertEqual(manifest["documents"][0]["hierarchy_model_decision_count"], 1)

            # 原文件不被修改，新文件才具有模型补齐的标题样式。
            original = Document(source)
            self.assertEqual(original.paragraphs[0].style.name, "Normal")
            styled_path = output / manifest["documents"][0]["styled_docx"]
            styled = Document(styled_path)
            self.assertEqual(styled.paragraphs[0].style.name, "Heading 1")
            self.assertEqual(styled.paragraphs[2].style.name, "Heading 1")
            self.assertEqual(styled.paragraphs[4].style.name, "Heading 2")
            self.assertEqual(styled.paragraphs[1].style.name, "Normal")
            self.assertEqual(len(styled.tables), 1)

            # 回写后的 DOCX 单靠原生样式即可恢复相同章节树，不再依赖模型。
            rebuilt = native_docx_to_ir(styled_path)
            native_headings = [
                (block.text, block.metadata["native_structures"][0]["heading_level"])
                for page in rebuilt.pages for block in page.blocks
                if block.metadata.get("native_structures")
            ]
            self.assertEqual(native_headings, [
                ("系统建设方案", 1), ("第一章 已有结构", 1), ("总体架构", 2),
            ])
            self.assertTrue((output / "documents/demo/chapter-tree.json").is_file())
            self.assertTrue((output / "documents/demo/heading-predictions.jsonl").is_file())

            # 独立切块 CLI 必须消费同一份预测，不能退回只信原生标题。
            chunks = root / "chunks-exp01"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/build_chunks.py"),
                "--input", str(output / "documents/demo/document.json"),
                "--predictions", str(output / "documents/demo/heading-predictions.jsonl"),
                "--mode", "hybrid", "--output", str(chunks),
            ], check=True, capture_output=True, text=True)
            chunk_tree = json.loads((chunks / "chapter-tree.json").read_text(encoding="utf-8"))
            self.assertEqual(chunk_tree["stats"]["section_count"], 3)
            self.assertTrue((chunks / "chunks.jsonl").is_file())

    def test_native_only_cli_needs_no_model_and_ignores_numbered_plain_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "native.docx"
            document = Document()
            document.add_paragraph("原生总论", style="Heading 1")
            document.add_paragraph("原生子节", style="Heading 2")
            document.add_paragraph("1.1 普通编号文本")
            document.add_paragraph("无样式标题")
            document.add_paragraph("正文。")
            document.save(source)

            output = root / "native-only-exp01"
            environment = dict(os.environ)
            environment.pop("HEADING_MODEL_BASE_URL", None)
            environment.pop("HEADING_MODEL_NAME", None)
            environment.pop("HEADING_MODEL_API_KEY", None)
            subprocess.run([
                sys.executable, str(ROOT / "scripts/build_docx_heading_dataset.py"),
                "--input", str(source), "--output", str(output), "--native-only",
            ], check=True, capture_output=True, text=True, env=environment)

            manifest = json.loads((output / "dataset-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["heading_mode"], "native")
            self.assertEqual(manifest["hierarchy_mode"], "native")
            self.assertIsNone(manifest["model"])
            self.assertEqual(manifest["documents"][0]["native_heading_count"], 2)
            self.assertEqual(manifest["documents"][0]["model_heading_count"], 0)
            self.assertEqual(manifest["documents"][0]["hierarchy_model_decision_count"], 0)
            self.assertEqual(manifest["documents"][0]["tree_section_count"], 2)

            tree = json.loads((output / "documents/native/chapter-tree.json").read_text(encoding="utf-8"))
            self.assertEqual([node["title"] for node in tree["nodes"][1:]], ["原生总论", "原生子节"])
            self.assertEqual([node["level"] for node in tree["nodes"][1:]], [1, 2])
            predictions = [
                json.loads(line) for line in
                (output / "documents/native/heading-predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["label"] for row in predictions], [
                "heading", "heading", "body", "body", "body",
            ])
            styled = Document(output / manifest["documents"][0]["styled_docx"])
            self.assertEqual(styled.paragraphs[0].style.name, "Heading 1")
            self.assertEqual(styled.paragraphs[1].style.name, "Heading 2")
            self.assertEqual(styled.paragraphs[2].style.name, "Normal")
            self.assertEqual(styled.paragraphs[3].style.name, "Normal")


if __name__ == "__main__":
    unittest.main()
