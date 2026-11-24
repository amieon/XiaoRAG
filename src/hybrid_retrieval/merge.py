"""把多个单文档检索单元导出合并为一份可建库语料，不改写记录。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .adapter import PRODUCERS
from .index import read_jsonl


MERGE_SCHEMA = "retrieval-merge.v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records
    ), encoding="utf-8")


def _required_string(record: dict, key: str, error: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(error)
    return value


def merge_retrieval_exports(
    inputs: Sequence[str | Path], output: str | Path,
) -> dict[str, Any]:
    """校验并合并导出目录；同一文档允许不同 producer，但不允许重复导入同一版本。"""
    if not inputs:
        raise ValueError("merge_requires_at_least_one_input")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("merge_output_not_empty_choose_new_directory")
    directories = [Path(item).resolve() for item in inputs]
    if len(set(directories)) != len(directories):
        raise ValueError("merge_duplicate_input_directory")

    units: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []
    input_summaries = []
    document_producers = set()
    for directory in directories:
        manifest_path = directory / "manifest.json"
        units_path = directory / "retrieval-units.jsonl"
        parents_path = directory / "retrieval-parents.jsonl"
        if not all(path.is_file() for path in (manifest_path, units_path, parents_path)):
            raise ValueError(f"merge_incomplete_export:{directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("schema_version") != "retrieval-preparation.v1":
            raise ValueError(f"merge_unsupported_input_manifest:{directory}")
        document_id = _required_string(manifest, "document_id", "merge_document_id_required")
        producer = _required_string(manifest, "producer", "merge_producer_required")
        if producer not in PRODUCERS:
            raise ValueError("merge_unknown_producer")
        if (producer, document_id) in document_producers:
            raise ValueError(f"merge_duplicate_document_producer:{producer}:{document_id}")
        document_producers.add((producer, document_id))

        current_units = read_jsonl(units_path)
        current_parents = read_jsonl(parents_path)
        if not current_units or not current_parents:
            raise ValueError(f"merge_empty_units_or_parents:{directory}")
        if manifest.get("unit_count") != len(current_units) or manifest.get("parent_count") != len(current_parents):
            raise ValueError(f"merge_manifest_count_mismatch:{directory}")
        for record in current_units:
            if record.get("schema_version") != "retrieval-unit.v1":
                raise ValueError("merge_requires_retrieval-unit.v1")
            if record.get("document_id") != document_id or (record.get("provenance") or {}).get("producer") != producer:
                raise ValueError("merge_unit_document_or_producer_mismatch")
            _required_string(record, "unit_id", "merge_unit_id_required")
            _required_string(record, "parent_id", "merge_child_parent_id_required")
            _required_string(record, "embedding_text", "merge_embedding_text_required")
        for record in current_parents:
            if record.get("schema_version") != "retrieval-parent.v1":
                raise ValueError("merge_requires_retrieval-parent.v1")
            if record.get("document_id") != document_id or (record.get("provenance") or {}).get("producer") != producer:
                raise ValueError("merge_parent_document_or_producer_mismatch")
            _required_string(record, "parent_id", "merge_parent_id_required")
        units.extend(current_units)
        parents.extend(current_parents)
        input_summaries.append({
            "path": str(directory), "document_id": document_id, "producer": producer,
            "source_chunks_sha256": manifest.get("source_chunks_sha256"),
            "unit_count": len(current_units), "parent_count": len(current_parents),
        })

    # unit 与 parent 共用命中装配命名空间，跨类型 ID 冲突也拒绝。
    seen: dict[str, str] = {}
    for kind, records, key in (("unit", units, "unit_id"), ("parent", parents, "parent_id")):
        for record in records:
            record_id = record[key]
            if record_id in seen:
                raise ValueError(f"merge_duplicate_global_id:{record_id}")
            seen[record_id] = kind
    parent_by_id = {record["parent_id"]: record for record in parents}
    child_counts = {record_id: 0 for record_id in parent_by_id}
    for unit in units:
        parent = parent_by_id.get(unit["parent_id"])
        if parent is None:
            raise ValueError(f"merge_dangling_parent_id:{unit['parent_id']}")
        if parent["document_id"] != unit["document_id"]:
            raise ValueError("merge_child_parent_document_mismatch")
        child_counts[parent["parent_id"]] += 1
    empty_parents = [record_id for record_id, count in child_counts.items() if count == 0]
    if empty_parents:
        raise ValueError(f"merge_parent_without_children:{empty_parents[0]}")

    producer_counts = {
        producer: sum(item["producer"] == producer for item in input_summaries)
        for producer in sorted(PRODUCERS) if any(item["producer"] == producer for item in input_summaries)
    }
    manifest = {
        "schema_version": MERGE_SCHEMA, "merge_complete": True,
        "input_count": len(input_summaries), "document_count": len({item["document_id"] for item in input_summaries}),
        "producer_document_count": len(input_summaries), "producer_counts": producer_counts,
        "unit_count": len(units), "parent_count": len(parents),
        "records_rewritten": False, "embedding_field": "embedding_text",
        "inputs": input_summaries,
    }
    # 所有输入与全局关系校验完成后才写出，避免留下看似可用的半成品。
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "retrieval-units.jsonl", units)
    _write_jsonl(output / "retrieval-parents.jsonl", parents)
    _write_json(output / "manifest.json", manifest)
    return manifest
