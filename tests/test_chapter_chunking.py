from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chapter_tree.tree import ChapterTreeConfig, build_chapter_tree_from_ir
from doc_chunking.engine import chunk_document, load_config
from doc_chunking.validation import validate_result
from doc_parser.preparation import _attach_docx_native_structure, _extract_docx


def ir_with_texts(items: list[tuple[str, str, dict]]) -> dict:
    blocks = []
    for index, (block_id, text, metadata) in enumerate(items):
        blocks.append({
            "block_id": block_id, "page_id": "p0", "block_type": "text", "text": text,
            "reading_order": index, "bbox": [0, index * 10, 10, index * 10 + 5],
            "metadata": metadata,
        })
    return {
        "document_id": "chunks", "schema_version": "0.1",
        "pages": [{"page_id": "p0", "page_no": 1, "width": 100, "height": 1000, "blocks": blocks}],
    }


def test_clause_is_child_boundary_not_extra_chapter_or_parent():
    ir = ir_with_texts([
        ("b0", "第一章 总则", {}),
        ("b1", "第一条 为明确责任，制定本合同。", {}),
        ("b2", "第二条 本合同适用于软件交付。", {}),
        ("b3", "第三条 双方应当遵守约定。", {}),
    ])
    result = chunk_document(ir, config=load_config())
    assert result["decision"]["strategy"] == "chapter_tree"
    assert len(result["parents"]) == 1
    assert len(result["children"]) == 3
    assert [child["clause_path"] for child in result["children"]] == [["第一条"], ["第二条"], ["第三条"]]
    assert result["chapter_tree"]["stats"]["section_count"] == 1
    assert validate_result(result, ir)["status"] == "passed"


def test_native_list_is_not_heading_and_explicit_continuation_stays_together():
    list_metadata = {"list_structures": [{
        "source": "docx_numbering", "num_id": "7", "level": 0,
        "ordinal": 1, "rendered_marker": "1.", "number_format": "decimal",
    }]}
    ir = ir_with_texts([
        ("b0", "1. 安装依赖", list_metadata),
        ("b1", "安装完成后检查版本。", {"list_continuation": True}),
        ("b2", "2. 启动服务", {"list_structures": [{**list_metadata["list_structures"][0], "ordinal": 2}]}),
    ])
    tree, _ = build_chapter_tree_from_ir(ir)
    assert tree["stats"]["heading_block_count"] == 0
    result = chunk_document(ir, config=load_config())
    assert len(result["parents"]) == 1
    assert len(result["children"]) == 1
    child = result["children"][0]
    assert child["metadata"]["split_reasons"] == ["atomic_list_item", "explicit_list_continuation"]
    assert [part["logical_block_id"] for part in child["metadata"]["body_parts"]] == ["b0", "b1", "b2"]
    assert validate_result(result, ir)["status"] == "passed"


def test_real_docx_numbering_is_extracted_and_aligned_by_rendered_marker(tmp_path):
    import pytest
    pytest.importorskip("docx")
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    path = tmp_path / "numbered.docx"
    document = Document()
    for text in ("安装依赖", "启动服务"):
        paragraph = document.add_paragraph(text)
        properties = OxmlElement("w:numPr")
        level = OxmlElement("w:ilvl")
        level.set(qn("w:val"), "0")
        number = OxmlElement("w:numId")
        number.set(qn("w:val"), "1")
        properties.extend([level, number])
        paragraph._p.get_or_add_pPr().append(properties)
    document.save(path)
    native = _extract_docx(path)
    assert [item["ordinal"] for item in native["lists"]] == [1, 2]
    assert all(item["source"] == "docx_numbering" for item in native["lists"])
    marker = native["lists"][0]["rendered_marker"]
    rendered = [{"text": f"{marker} 安装依赖"}]
    attached = _attach_docx_native_structure(rendered, native)
    assert attached[0]["list_structure"]["num_id"] == "1"
    assert attached[0]["native_structure_match"] == "unique_exact_text"


def test_empty_heading_and_body_blocks_keep_source_coverage_without_adding_text():
    ir = ir_with_texts([("b0", "", {}), ("b1", "正文。", {}), ("b2", "", {})])
    result = chunk_document(
        ir, config=load_config(),
        predictions=[{"block_id": "b0", "label": "heading", "heading_level": 1}],
    )
    assert len(result["children"]) == 1
    child = result["children"][0]
    assert child["body_text"] == "正文。"
    assert child["source_block_ids"] == ["b0", "b1", "b2"]
    assert child["metadata"]["source_only_empty_block_ids"] == ["b0", "b2"]
    assert [part["logical_block_id"] for part in child["metadata"]["body_parts"]] == ["b1"]
    assert validate_result(result, ir)["status"] == "passed"


def test_wholly_empty_group_has_provenance_but_no_fabricated_text():
    ir = ir_with_texts([("b0", "", {}), ("b1", "", {})])
    result = chunk_document(ir, config=load_config())
    assert len(result["children"]) == 1
    child = result["children"][0]
    assert child["text"] == "" and child["body_text"] == ""
    assert child["source_block_ids"] == ["b0", "b1"]
    assert child["metadata"]["body_parts"] == []
    assert validate_result(result, ir)["status"] == "passed"


def test_chunk_cli_emits_structural_chunks_and_tree(tmp_path):
    ir = ir_with_texts([("b0", "第一章 总则", {}), ("b1", "第一条 正文。", {})])
    source = tmp_path / "document.json"
    source.write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "out"
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "build_chunks.py"),
        "--input", str(source), "--output", str(output), "--mode", "upstream",
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    summary = json.loads(completed.stdout)
    assert summary["parent_count"] == 1 and summary["child_count"] == 1
    lines = (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["level"] for line in lines} == {"parent", "child"}
    assert json.loads((output / "self_check.json").read_text(encoding="utf-8"))["integrity"]["status"] == "passed"
