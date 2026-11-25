from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_chunking.cdc import split_cdc
from doc_chunking.engine import chunk_document, load_config
from doc_chunking.validation import validate_result
from hybrid_retrieval.adapter import adapt_structural_chunks
from versioning import publish_version
from test_chapter_chunking import ir_with_texts


def cdc_config():
    config = load_config()
    config["sizing"].update(
        min_chars=60, target_chars=100, max_chars=180,
        window_chars=8, boundary_snap_chars=24,
    )
    return config


def sample_text():
    return "".join(
        f"第{index:03d}节用于说明系统版本与增量更新的完整处理过程。"
        for index in range(80)
    )


def test_cdc_is_deterministic_lossless_and_respects_hard_limit():
    sizing = cdc_config()["sizing"]
    text = sample_text()
    first, second = split_cdc(text, sizing), split_cdc(text, deepcopy(sizing))
    assert first == second
    assert first[0][0] == 0 and first[-1][1] == len(text)
    assert all(left[1] == right[0] for left, right in zip(first, first[1:]))
    assert "".join(text[start:end] for start, end, _ in first) == text
    assert all(0 < end - start <= sizing["max_chars"] for start, end, _ in first)
    assert all(text[end - 1] == "。" for _, end, reason in first[:-1])


def test_cdc_without_natural_boundary_has_explicit_hard_cut():
    text = "甲" * 25
    sizing = {
        "min_chars": 4, "target_chars": 8, "max_chars": 10,
        "window_chars": 2, "boundary_snap_chars": 2,
    }
    spans = split_cdc(text, sizing)
    assert [(start, end) for start, end, _ in spans] == [(0, 10), (10, 20), (20, 25)]
    assert [reason for _, _, reason in spans] == [
        "cdc_hard_character_boundary", "cdc_hard_character_boundary", "cdc_tail",
    ]


def test_cdc_spans_multiple_ir_blocks_without_gap_or_overlap():
    text = sample_text()
    sentences = [text[index:index + 26] for index in range(0, len(text), 26)]
    ir = ir_with_texts([(f"b{index}", value, {}) for index, value in enumerate(sentences)])
    result = chunk_document(ir, config=cdc_config())
    assert len(result["children"]) > 2
    assert "".join(part["text"] for child in result["children"] for part in child["metadata"]["body_parts"]) == text
    assert validate_result(result, ir)["status"] == "passed"
    assert result["decision"]["length_policy"]["algorithm"] == "cdc-natural-text-v1"


def test_local_edit_only_changes_nearby_child_instead_of_shifting_all_suffixes():
    text = sample_text()
    changed = text.replace("第000节", "第0000节", 1)
    old_ir = ir_with_texts([("b0", text, {})])
    new_ir = ir_with_texts([("b0", changed, {})])
    old = chunk_document(old_ir, config=cdc_config())
    new = chunk_document(new_ir, config=cdc_config())
    old_bodies = [child["body_text"] for child in old["children"]]
    new_bodies = [child["body_text"] for child in new["children"]]
    assert old_bodies[0] != new_bodies[0]
    assert old_bodies[1:] == new_bodies[1:]
    assert validate_result(old, old_ir)["status"] == "passed"
    assert validate_result(new, new_ir)["status"] == "passed"


class TinyEmbedder:
    model_id = "tiny-cdc-version-vectors"

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def write_export(root: Path, name: str, result) -> Path:
    output = root / name
    output.mkdir()
    chunks = result["parents"] + result["children"]
    adapted = adapt_structural_chunks(
        chunks, producer="chapter_tree",
        security={"classification": "internal", "allowed_principals": ["group:dev"]},
    )
    for filename, records in (
        ("retrieval-units.jsonl", adapted["retrieval_units"]),
        ("retrieval-parents.jsonl", adapted["parents"]),
    ):
        (output / filename).write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8",
        )
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": "retrieval-preparation.v1", "document_id": "chunks",
        "producer": "chapter_tree", "unit_count": len(adapted["retrieval_units"]),
        "parent_count": len(adapted["parents"]),
        "source_chunks_sha256": hashlib.sha256(
            "".join(json.dumps(chunk, sort_keys=True) for chunk in chunks).encode("utf-8")
        ).hexdigest(),
        "chunking": chunks[0]["metadata"]["chunking"],
    }), encoding="utf-8")
    return output


def read_version_units(store: Path, revision: int):
    current_path = next((store / "documents").glob("doc_*/current.json"))
    root = current_path.parent / "versions" / f"r{revision:06d}"
    units = [json.loads(line) for line in (root / "retrieval-units.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return units, manifest


def test_version_publishing_reuses_cdc_suffix_vectors_and_records_algorithm(tmp_path):
    text = sample_text()
    changed = text.replace("第000节", "第0000节", 1)
    old = chunk_document(ir_with_texts([("b0", text, {})]), config=cdc_config())
    new = chunk_document(ir_with_texts([("b0", changed, {})]), config=cdc_config())
    export0 = write_export(tmp_path, "export0", old)
    export1 = write_export(tmp_path, "export1", new)
    source = tmp_path / "document.txt"
    store = tmp_path / "store"
    embedder = TinyEmbedder()
    source.write_text("版本零", encoding="utf-8")
    publish_version(source, export0, store, embedder=embedder)
    old_units, _ = read_version_units(store, 0)
    source.write_text("版本一", encoding="utf-8")
    result = publish_version(source, export1, store, embedder=embedder)
    new_units, manifest = read_version_units(store, 1)
    old_by_text = {unit["body_text"]: unit for unit in old_units}
    assert result["reused_from_previous_count"] == len(old_units) - 1
    assert result["newly_encoded_text_count"] == 1
    assert len(embedder.calls) == 2 and len(embedder.calls[1]) == 1
    for unit in new_units[1:]:
        assert unit["vector_ref"] == old_by_text[unit["body_text"]]["vector_ref"]
    assert manifest["chunking"]["algorithm"] == "cdc-natural-text-v1"
    assert manifest["chunking"]["length_unit"] == "unicode_codepoints"
