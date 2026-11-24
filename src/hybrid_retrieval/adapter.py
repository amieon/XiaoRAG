"""structural-chunk.v1 → M2 统一检索单元；不改写文本，不推断权限。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Iterable


UNIT_SCHEMA = "retrieval-unit.v1"
PARENT_SCHEMA = "retrieval-parent.v1"
PRODUCERS = {"senior_strategy", "chapter_tree"}


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_security(value: Any) -> dict[str, Any]:
    """权限必须显式提供；空主体列表不自动等价于公开。"""
    if not isinstance(value, dict):
        raise ValueError("security_policy_requires_object")
    classification = value.get("classification")
    principals = value.get("allowed_principals")
    if not isinstance(classification, str) or not classification.strip():
        raise ValueError("security_classification_required")
    if not isinstance(principals, list) or any(
        not isinstance(item, str) or not item.strip() for item in principals
    ):
        raise ValueError("security_allowed_principals_requires_string_list")
    result = deepcopy(value)
    result["classification"] = classification.strip()
    result["allowed_principals"] = list(dict.fromkeys(item.strip() for item in principals))
    return result


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"chunk_{key}_required")
    return value


def _string_list(record: dict[str, Any], key: str) -> list[str]:
    value = record.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"chunk_{key}_requires_string_list")
    return deepcopy(value)


def _source_fields(record: dict[str, Any]) -> dict[str, Any]:
    refs = record.get("source_refs")
    native_refs = record.get("native_refs", [])
    if not isinstance(refs, list) or not isinstance(native_refs, list):
        raise ValueError("chunk_source_refs_require_lists")
    if not refs and not native_refs:
        raise ValueError("chunk_requires_locatable_source")
    pages = record.get("page_numbers", [])
    if not isinstance(pages, list) or any(type(page) is not int or page < 1 for page in pages):
        raise ValueError("chunk_page_numbers_require_positive_integers")
    return {
        "source_refs": deepcopy(refs), "native_refs": deepcopy(native_refs),
        "page_numbers": pages,
    }


def _provenance(record: dict[str, Any], producer: str) -> dict[str, Any]:
    return {
        "producer": producer,
        "strategy": record.get("strategy"),
        "format_channel": record.get("format_channel"),
        "source_schema_version": record.get("schema_version"),
    }


def adapt_structural_chunks(
    chunks: Iterable[dict[str, Any]], *, security: dict[str, Any], producer: str,
) -> dict[str, Any]:
    """转换一份文档的 chunks；parent/child 原 ID 只作为来源 ID 保存。"""
    if producer not in PRODUCERS:
        raise ValueError("unknown_chunk_producer")
    policy = validate_security(security)
    records = list(chunks)
    if not records:
        raise ValueError("chunks_jsonl_is_empty")
    if any(not isinstance(item, dict) for item in records):
        raise ValueError("chunk_record_requires_object")
    for item in records:
        if item.get("schema_version") != "structural-chunk.v1":
            raise ValueError("adapter_requires_structural-chunk.v1")
    document_ids = {_required_string(item, "document_id") for item in records}
    if len(document_ids) != 1:
        raise ValueError("one_adapter_run_requires_one_document")
    document_id = document_ids.pop()
    chunk_ids = [_required_string(item, "chunk_id") for item in records]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("duplicate_chunk_id")
    parents = {item["chunk_id"]: item for item in records if item.get("level") == "parent"}
    children = [item for item in records if item.get("level") == "child"]
    if not parents or not children or len(parents) + len(children) != len(records):
        raise ValueError("adapter_requires_parent_and_child_levels")
    for child in children:
        if _required_string(child, "parent_id") not in parents:
            raise ValueError("dangling_parent_id")
    actual_membership = {
        parent_id: [child["chunk_id"] for child in children if child["parent_id"] == parent_id]
        for parent_id in parents
    }
    for parent_id, parent in parents.items():
        declared = _string_list(parent, "child_ids")
        if declared != actual_membership[parent_id]:
            raise ValueError("parent_child_membership_mismatch")

    def namespaced(chunk_id: str) -> str:
        return f"{producer}:{document_id}:{chunk_id}"

    parent_units = []
    for parent in parents.values():
        text = _required_string(parent, "text")
        body = parent.get("body_text")
        if not isinstance(body, str):
            raise ValueError("parent_body_text_requires_string")
        parent_units.append({
            "schema_version": PARENT_SCHEMA,
            "parent_id": namespaced(parent["chunk_id"]),
            "source_parent_id": parent["chunk_id"],
            "document_id": document_id,
            "text": text,
            "body_text": body,
            "context_prefix": parent.get("context_prefix", ""),
            "title_path": _string_list(parent, "title_path"),
            "clause_path": _string_list(parent, "clause_path"),
            "kind": parent.get("kind"),
            "length": deepcopy(parent.get("length")),
            **_source_fields(parent),
            "security": deepcopy(policy),
            "provenance": _provenance(parent, producer),
            "content_hash": _text_hash(text),
        })

    retrieval_units = []
    for child in children:
        embedding_text = _required_string(child, "text")
        body = child.get("body_text")
        if not isinstance(body, str):
            raise ValueError("child_body_text_requires_string")
        metadata = child.get("metadata") or {}
        retrieval_units.append({
            "schema_version": UNIT_SCHEMA,
            "unit_id": namespaced(child["chunk_id"]),
            "source_chunk_id": child["chunk_id"],
            "document_id": document_id,
            "parent_id": namespaced(child["parent_id"]),
            "embedding_text": embedding_text,
            "body_text": body,
            "context_prefix": child.get("context_prefix", ""),
            "title_path": _string_list(child, "title_path"),
            "clause_path": _string_list(child, "clause_path"),
            "kind": child.get("kind"),
            "length": deepcopy(child.get("length")),
            "split_reasons": deepcopy(metadata.get("split_reasons", [])),
            "oversize_character_budget": bool(metadata.get("oversize_character_budget")),
            **_source_fields(child),
            "security": deepcopy(policy),
            "provenance": _provenance(child, producer),
            "content_hash": _text_hash(embedding_text),
        })
    return {
        "document_id": document_id,
        "producer": producer,
        "retrieval_units": retrieval_units,
        "parents": parent_units,
        "self_check": {
            "status": "passed", "input_chunk_count": len(records),
            "retrieval_unit_count": len(retrieval_units), "parent_count": len(parent_units),
            "text_rewritten": False, "security_source": "explicit_policy",
        },
    }
