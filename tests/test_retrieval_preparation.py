from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.adapter import adapt_structural_chunks


SECURITY = {"classification": "internal", "allowed_principals": ["group:engineering"]}


def chunks():
    refs = [{"block_id": "b0", "page_number": 1, "unit_id": "p0", "char_start": 0, "char_end": 4}]
    common = {
        "schema_version": "structural-chunk.v1", "document_id": "doc1",
        "strategy": "chapter_tree", "format_channel": "document", "kind": "section",
        "title_path": ["第一章"], "clause_path": ["第一条"],
        "context_prefix": "第一章 > 第一条", "source_refs": refs,
        "source_block_ids": ["b0"], "native_refs": [], "page_numbers": [1],
        "page_start": 1, "page_end": 1, "unit_ids": ["p0"],
    }
    parent = {
        **common, "chunk_id": "doc1:p00001", "level": "parent", "parent_id": None,
        "child_ids": ["doc1:p00001:c0001"], "body_text": "测试正文",
        "text": "第一章 > 第一条\n\n测试正文",
        "length": {"characters": 20, "unit": "unicode_codepoints", "token_count": None},
        "metadata": {"split_reasons": []},
    }
    child = {
        **common, "chunk_id": "doc1:p00001:c0001", "level": "child",
        "parent_id": "doc1:p00001", "child_ids": [], "body_text": "测试正文",
        "text": "第一章 > 第一条\n\n测试正文",
        "length": {"characters": 20, "unit": "unicode_codepoints", "token_count": None},
        "metadata": {"split_reasons": ["whole_block"], "oversize_character_budget": False},
    }
    return [parent, child]


def test_both_producers_share_schema_and_only_ids_and_provenance_differ():
    senior = adapt_structural_chunks(chunks(), security=SECURITY, producer="senior_strategy")
    ours = adapt_structural_chunks(chunks(), security=SECURITY, producer="chapter_tree")
    a, b = senior["retrieval_units"][0], ours["retrieval_units"][0]
    assert a["schema_version"] == b["schema_version"] == "retrieval-unit.v1"
    assert a["embedding_text"] == b["embedding_text"] == chunks()[1]["text"]
    assert a["content_hash"] == b["content_hash"]
    assert a["unit_id"] != b["unit_id"]
    assert a["provenance"]["producer"] == "senior_strategy"
    assert b["provenance"]["producer"] == "chapter_tree"
    assert a["parent_id"] == senior["parents"][0]["parent_id"]
    assert b["parent_id"] == ours["parents"][0]["parent_id"]


def test_adapter_preserves_security_refs_and_does_not_rewrite_text():
    result = adapt_structural_chunks(chunks(), security=SECURITY, producer="chapter_tree")
    unit = result["retrieval_units"][0]
    assert unit["security"] == SECURITY
    assert unit["source_refs"] == chunks()[1]["source_refs"]
    assert unit["body_text"] == "测试正文"
    assert result["self_check"]["text_rewritten"] is False


def test_missing_security_fails_closed():
    with pytest.raises(ValueError, match="security"):
        adapt_structural_chunks(chunks(), security={}, producer="chapter_tree")


def test_dangling_parent_and_missing_source_are_rejected():
    broken = chunks()
    broken[1]["parent_id"] = "missing"
    with pytest.raises(ValueError, match="dangling_parent"):
        adapt_structural_chunks(broken, security=SECURITY, producer="chapter_tree")
    no_source = chunks()
    no_source[0]["source_refs"] = []
    no_source[1]["source_refs"] = []
    with pytest.raises(ValueError, match="locatable_source"):
        adapt_structural_chunks(no_source, security=SECURITY, producer="chapter_tree")


def test_export_cli_round_trip(tmp_path):
    source = tmp_path / "chunks.jsonl"
    source.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in chunks()), encoding="utf-8")
    policy = tmp_path / "security.json"
    policy.write_text(json.dumps(SECURITY, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "out"
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "export_retrieval_units.py"),
        "--input", str(source), "--security", str(policy),
        "--producer", "chapter_tree", "--output", str(output),
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)["status"] == "passed"
    unit = json.loads((output / "retrieval-units.jsonl").read_text(encoding="utf-8").strip())
    parent = json.loads((output / "retrieval-parents.jsonl").read_text(encoding="utf-8").strip())
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert unit["parent_id"] == parent["parent_id"]
    assert manifest["embedding_field"] == "embedding_text"
    assert manifest["vectorization_performed"] is False
