from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.merge import merge_retrieval_exports


def write_export(root: Path, name: str, document_id: str, producer: str) -> Path:
    output = root / name
    output.mkdir(parents=True)
    prefix = f"{producer}:{document_id}"
    parent = {
        "schema_version": "retrieval-parent.v1", "parent_id": prefix + ":p1",
        "document_id": document_id, "text": "完整章节", "body_text": "完整章节",
        "provenance": {"producer": producer},
    }
    unit = {
        "schema_version": "retrieval-unit.v1", "unit_id": prefix + ":c1",
        "document_id": document_id, "parent_id": parent["parent_id"],
        "embedding_text": document_id + " 正文", "body_text": "正文",
        "provenance": {"producer": producer},
    }
    (output / "retrieval-units.jsonl").write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "retrieval-parents.jsonl").write_text(json.dumps(parent, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": "retrieval-preparation.v1", "document_id": document_id,
        "producer": producer, "source_chunks_sha256": "abc",
        "unit_count": 1, "parent_count": 1,
    }), encoding="utf-8")
    return output


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_merge_multiple_documents_and_producers_without_rewriting(tmp_path):
    first = write_export(tmp_path, "one", "doc1", "chapter_tree")
    second = write_export(tmp_path, "two", "doc1", "senior_strategy")
    third = write_export(tmp_path, "three", "doc2", "chapter_tree")
    output = tmp_path / "merged"
    manifest = merge_retrieval_exports([first, second, third], output)
    units = read_jsonl(output / "retrieval-units.jsonl")
    parents = read_jsonl(output / "retrieval-parents.jsonl")
    assert manifest["document_count"] == 2
    assert manifest["producer_document_count"] == 3
    assert manifest["producer_counts"] == {"chapter_tree": 2, "senior_strategy": 1}
    assert [item["unit_id"] for item in units] == [
        "chapter_tree:doc1:c1", "senior_strategy:doc1:c1", "chapter_tree:doc2:c1",
    ]
    assert len(parents) == 3 and manifest["records_rewritten"] is False
    assert units[0]["embedding_text"] == "doc1 正文"


def test_duplicate_id_and_duplicate_document_producer_are_rejected(tmp_path):
    first = write_export(tmp_path, "one", "doc1", "chapter_tree")
    duplicate = write_export(tmp_path, "two", "doc1", "chapter_tree")
    with pytest.raises(ValueError, match="duplicate_document_producer"):
        merge_retrieval_exports([first, duplicate], tmp_path / "out")

    other = write_export(tmp_path, "three", "doc2", "chapter_tree")
    unit_path = other / "retrieval-units.jsonl"
    unit = json.loads(unit_path.read_text(encoding="utf-8"))
    unit["unit_id"] = "chapter_tree:doc1:c1"
    unit_path.write_text(json.dumps(unit), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_global_id"):
        merge_retrieval_exports([first, other], tmp_path / "out2")


def test_dangling_parent_and_empty_input_fail_before_output(tmp_path):
    first = write_export(tmp_path, "one", "doc1", "chapter_tree")
    unit_path = first / "retrieval-units.jsonl"
    unit = json.loads(unit_path.read_text(encoding="utf-8"))
    unit["parent_id"] = "missing"
    unit_path.write_text(json.dumps(unit), encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="dangling_parent"):
        merge_retrieval_exports([first], output)
    assert not output.exists()

    second = write_export(tmp_path, "two", "doc2", "chapter_tree")
    (second / "retrieval-units.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty_units_or_parents"):
        merge_retrieval_exports([second], tmp_path / "out2")


def test_merge_cli_then_existing_build_input_contract(tmp_path):
    first = write_export(tmp_path, "one", "doc1", "chapter_tree")
    second = write_export(tmp_path, "two", "doc2", "chapter_tree")
    output = tmp_path / "merged"
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "merge_retrieval_units.py"),
        "--inputs", str(first), str(second), "--output", str(output),
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result["status"] == "passed" and result["unit_count"] == 2
    assert (output / "retrieval-units.jsonl").is_file()
    assert (output / "retrieval-parents.jsonl").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["merge_complete"] is True and len(manifest["inputs"]) == 2
