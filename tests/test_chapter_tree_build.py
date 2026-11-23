from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chapter_tree.groups import chapter_groups_document, chapter_tree_to_groups
from chapter_tree.tree import ChapterTreeConfig, build_chapter_tree, build_chapter_tree_from_ir


ROOT = Path(__file__).resolve().parents[1]


def block(
    block_id: str, text: str, *, kind: str = "paragraph", level: int | None = None,
    native_kind: str | None = None, native: bool = False,
) -> dict:
    metadata = {}
    if level is not None:
        metadata["heading_level"] = level
    if native:
        metadata["native_structures"] = [{
            "source": "docx_style", "native_id": f"p/{block_id}",
            "role": "document_title" if native_kind == "document_title" else "heading",
            "heading_level": level or 1,
        }]
    return {
        "block_id": block_id, "kind": kind, "text": text,
        "unit_index": 0, "order": int(block_id[1:]), "input_order": int(block_id[1:]),
        "metadata": metadata, "native_kind": native_kind or kind,
    }


def test_no_heading_document_is_flat_and_special_objects_stay_body():
    blocks = [
        block("b0", "没有标题的正文。"),
        block("b1", "1、这是列表项", kind="list_item"),
        block("b2", "2.1 这是表格", kind="table"),
        block("b3", "3.1 这是公式", kind="formula"),
    ]
    tree = build_chapter_tree(blocks, document_id="flat")
    groups = chapter_tree_to_groups(tree, blocks)
    assert tree["stats"]["flat_document"] is True
    assert tree["stats"]["heading_block_count"] == 0
    assert len(groups) == 1
    assert groups[0]["kind"] == "flat"
    assert groups[0]["source_block_ids"] == ["b0", "b1", "b2", "b3"]
    assert tree["nodes"][0]["block_ids"] == ["b0", "b1", "b2", "b3"]


def test_native_headings_build_deep_tree_and_warn_on_level_jump():
    blocks = [
        block("b0", "项目总则", level=1, native=True, native_kind="document_title"),
        block("b1", "第一章 范围", level=1, native=True),
        block("b2", "1.1.1 深层条款", level=3, native=True),
        block("b3", "正文"),
    ]
    tree = build_chapter_tree(blocks, document_id="deep", config=ChapterTreeConfig(heading_mode="native"))
    assert tree["stats"]["section_count"] == 3
    assert tree["stats"]["max_depth"] == 3
    assert tree["nodes"][3]["title_path"] == ["项目总则", "第一章 范围", "1.1.1 深层条款"]
    assert any(item["type"] == "heading_level_jump" and item["block_id"] == "b2" for item in tree["warnings"])
    assert not any("sec" in (item.get("title") or "") and item.get("title") == "" for item in tree["warnings"])


def test_empty_ancestors_follow_first_contentful_descendant():
    blocks = [
        block("b0", "文档标题", native_kind="document_title", level=1, native=True),
        block("b1", "第一章", level=1, native=True),
        block("b2", "1.1 范围", level=2, native=True),
        block("b3", "本章正文"),
        block("b4", "1.2 空尾章", level=2, native=True),
    ]
    tree = build_chapter_tree(blocks, document_id="empty")
    grouped = chapter_groups_document(tree, blocks)
    assert len(grouped["groups"]) == 2
    first, last = grouped["groups"]
    assert first["source_block_ids"] == ["b0", "b1", "b2", "b3"]
    assert first["title_path"] == ["文档标题", "第一章", "1.1 范围"]
    assert first["context"]["inherited_heading_block_ids"] == ["b0", "b1"]
    assert first["strategy"] == "chapter_tree"
    assert last["source_block_ids"] == ["b4"]
    assert last["context"]["heading_only_group"] is True
    assert grouped["self_check"]["represented_block_count"] == len(blocks)


def test_preamble_and_numbered_heading_promotion():
    blocks = [
        block("b0", "前言正文"),
        block("b1", "1 总则"),
        block("b2", "总则正文"),
        block("b3", "1.1 范围"),
        block("b4", "范围正文"),
        block("b5", "1.2 这句话是正文。"),
        block("b6", "1.3 空尾章"),
    ]
    tree = build_chapter_tree(blocks, document_id="numbered")
    groups = chapter_tree_to_groups(tree, blocks)
    assert [group["kind"] for group in groups] == ["preamble", "section", "section", "section"]
    assert groups[1]["source_block_ids"] == ["b1", "b2"]
    assert groups[2]["source_block_ids"] == ["b3", "b4", "b5"]
    assert groups[2]["title_path"] == ["1 总则", "1.1 范围"]
    assert tree["block_decisions"][-2]["is_heading"] is False
    assert groups[-1]["source_block_ids"] == ["b6"]
    assert groups[-1]["context"]["heading_only_group"] is True


def test_upstream_labels_work_without_native_metadata():
    blocks = [
        block("b0", "上游标题", kind="heading", native_kind="paragraph_title", level=2),
        block("b1", "正文"),
    ]
    tree = build_chapter_tree(blocks, document_id="upstream", config=ChapterTreeConfig(heading_mode="upstream"))
    assert tree["block_decisions"][0]["source"] == "upstream_layout_label"
    assert tree["nodes"][1]["level"] == 2
    assert tree["warnings"][0]["type"] == "heading_level_jump"


def test_transformer_prediction_can_supply_heading_and_veto_upstream():
    blocks = [
        block("b0", "模型标题", kind="heading", native_kind="paragraph_title"),
        block("b1", "模型补出的标题"),
        block("b2", "正文"),
    ]
    predictions = [
        {"block_id": "b0", "label": "body", "mode": "transformer"},
        {"block_id": "b1", "label": "heading", "heading_level": 1, "mode": "transformer"},
    ]
    tree = build_chapter_tree(blocks, document_id="pred", predictions=predictions)
    assert [item["is_heading"] for item in tree["block_decisions"]] == [False, True, False]
    assert tree["nodes"][1]["heading_source"] == "transformer"


def test_cli_round_trip_does_not_modify_document_ir(tmp_path):
    ir = {
        "document_id": "cli", "source_path": "cli.docx", "source_hash": "abc",
        "schema_version": "0.1", "metadata": {},
        "pages": [{
            "page_id": "p0", "page_no": 1, "width": 100, "height": 100,
            "blocks": [
                {"block_id": "b0", "page_id": "p0", "block_type": "paragraph_title",
                 "text": "第一章 总则", "reading_order": 0, "bbox": [0, 0, 10, 10],
                 "metadata": {"native_structures": [{"source": "docx_style", "role": "heading", "heading_level": 1}]}},
                {"block_id": "b1", "page_id": "p0", "block_type": "text", "text": "正文",
                 "reading_order": 1, "bbox": [0, 10, 10, 20], "metadata": {}},
            ],
        }],
    }
    source = tmp_path / "document.json"
    source.write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
    before = source.read_bytes()
    output = tmp_path / "tree-output"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_chapter_tree.py"),
         "--input", str(source), "--output", str(output), "--mode", "native"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout)["group_count"] == 1
    tree = json.loads((output / "chapter-tree.json").read_text(encoding="utf-8"))
    groups = json.loads((output / "chapter-groups.json").read_text(encoding="utf-8"))
    assert tree["schema_version"] == "chapter-tree.v1"
    assert groups["groups"][0]["source_block_ids"] == ["b0", "b1"]
    assert source.read_bytes() == before
