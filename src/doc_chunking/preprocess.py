"""Conservative layer-zero cleanup of explicit document structure.

No layout detection or relation recovery happens here. The caller supplies
normalized IR blocks and already accepted cross-page relations. Originals are
retained in metadata, so cleanup and logical-object assembly remain auditable.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any


FURNITURE = {"header", "page_header", "footer", "page_footer", "page_number", "number", "watermark"}
TOC_KINDS = {"toc", "toc_entry", "table_of_contents", "table_of_contents_entry"}
NOTE_KINDS = {"speaker_notes", "speaker_note", "slide_notes"}
CONTINUATIONS = {"paragraph_continuation", "table_continuation", "cross_page_table"}
ACCEPTED = {"accept", "accepted"}


def _metadata(value: dict) -> dict:
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _accepted(relation: dict) -> bool:
    metadata = _metadata(relation)
    flags = [value for value in (relation.get("accepted"), metadata.get("accepted")) if type(value) is bool]
    statuses = [str(value).strip().lower() for value in
                (relation.get("decision"), relation.get("status"), metadata.get("decision"), metadata.get("status"))
                if value is not None]
    # A candidate or contradictory decision always wins over an acceptance flag.
    return False not in flags and not any(value not in ACCEPTED for value in statuses) and (True in flags or bool(statuses))


def _document_id(block: dict, fallback: str) -> Any:
    return block.get("document_id", _metadata(block).get("document_id", fallback))


def _toc_units(ir: dict) -> set[str]:
    result = set()
    for page in ir.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        metadata = _metadata(page)
        role = metadata.get("page_type", metadata.get("role", page.get("page_type")))
        if metadata.get("is_toc") is True or metadata.get("toc_page") is True or isinstance(role, str) and role in TOC_KINDS:
            ident = page.get("page_id", page.get("unit_id"))
            if ident is not None:
                result.add(ident)
    return result


def _exclusion(block: dict, toc_units: set[str], settings: dict) -> str | None:
    metadata = _metadata(block)
    kinds = {value for value in (block.get("kind"), block.get("native_kind"), metadata.get("role")) if isinstance(value, str)}
    if kinds & NOTE_KINDS:
        return "speaker_notes_metadata"
    if kinds & FURNITURE:
        return "explicit_furniture" if settings.get("strip_furniture", True) else None
    if kinds & TOC_KINDS or metadata.get("is_toc") is True or block.get("unit_id") in toc_units:
        return "explicit_toc" if settings.get("strip_toc", True) else None
    if metadata.get("excluded_from_body") is True:
        return "explicit_excluded_from_body"
    return None


def _rows(block: dict, native: dict) -> list[list] | None:
    metadata = _metadata(block)
    table = metadata.get("table") if isinstance(metadata.get("table"), dict) else {}
    rows = metadata.get("rows", table.get("rows"))
    if rows is None:
        native_ids = {source.get("native_id") for source in block.get("sources", []) if isinstance(source, dict)}
        native_ids.discard(None)
        matches = [table for table in native.get("tables", []) or [] if isinstance(table, dict)
                   and table.get("raw_id", table.get("block_id")) in native_ids]
        if len(matches) == 1:
            rows = matches[0].get("rows")
    if isinstance(rows, list) and rows and all(isinstance(row, list) for row in rows):
        return deepcopy(rows)
    return None


def _header_count(block: dict) -> int | None:
    metadata = _metadata(block)
    status = str(metadata.get("header_status", "")).lower()
    if any(word in status for word in ("candidate", "unverified", "heuristic")):
        return None
    count = metadata.get("header_row_count")
    if type(count) is int and count >= 0:
        return count
    return None


def _repeat_header_count(left: dict, right: dict, relation: dict) -> int | None:
    metadata = _metadata(relation)
    for container in (relation, metadata):
        for key in ("repeat_header_rows", "repeated_header_rows"):
            count = container.get(key)
            if type(count) is int and count >= 0:
                return count
    left_count, right_count = _header_count(left), _header_count(right)
    return left_count if left_count is not None and left_count == right_count else None


def _same_cells(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_cells(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_cells(value, right[key]) for key, value in left.items())
    return left == right


def _columns(block: dict, native: dict) -> int | None:
    count = _metadata(block).get("column_count")
    if type(count) is int and count > 0:
        return count
    rows = _rows(block, native)
    widths = {len(row) for row in rows} if rows else set()
    return next(iter(widths)) if len(widths) == 1 else None


def _horizontal_table(relation: dict) -> bool:
    for container in (relation, _metadata(relation)):
        for key in ("orientation", "direction", "action", "table_merge_action"):
            value = str(container.get(key, "")).lower()
            if "horizontal" in value or value in {"side_by_side", "append_columns", "merge_columns"}:
                return True
    return False


def _render_rows(rows: list[list], header_count: int | None) -> str:
    """Produce logical table text while keeping lossless cell arrays separately."""
    def cell(value: Any) -> str:
        return ("" if value is None else str(value)).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
    lines = ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    if header_count == 1 and rows:
        lines.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")
    return "\n".join(lines)


def _join(left: dict, right: dict, relation: dict, native: dict,
          fragments: list[dict], relation_ids: list) -> tuple[dict, dict]:
    result = deepcopy(left)
    metadata = result.setdefault("metadata", {})
    metadata["fragments"] = deepcopy(fragments)
    result["sources"] = deepcopy(left.get("sources", [])) + deepcopy(right.get("sources", []))
    source_ids = [fragment["block_id"] for fragment in fragments]
    metadata["chunk_preprocess"] = {"source_block_ids": source_ids, "relation_ids": list(relation_ids),
                                     "join_separator": "\n", "logical_object": left["kind"]}
    detail = {"source_block_ids": source_ids}
    if left["kind"] == "table":
        left_rows, right_rows = _rows(left, native), _rows(right, native)
        if left_rows is not None and right_rows is not None:
            count = _repeat_header_count(left, right, relation)
            removed = count if count and _same_cells(left_rows[:count], right_rows[:count]) and count <= min(len(left_rows), len(right_rows)) else 0
            rows = left_rows + right_rows[removed:]
            metadata["rows"] = rows
            header_count = _header_count(left)
            if header_count is None and removed:
                header_count = removed
                metadata["header_status"] = "accepted_cross_page_relation"
            if header_count is not None:
                metadata["header_row_count"] = header_count
            metadata["row_count"] = len(rows)
            result["text"] = _render_rows(rows, header_count)
            detail.update(repeated_header_rows_removed=removed,
                          header_handling="explicit_matching_header_removed" if removed else "all_rows_preserved")
            if count and not removed:
                detail["header_warning"] = "declared_repeated_header_mismatch_preserved"
        else:
            result["text"] = left.get("text", "") + "\n" + right.get("text", "")
            detail.update(repeated_header_rows_removed=0, header_handling="unstructured_fragments_preserved")
    else:
        result["text"] = left.get("text", "") + "\n" + right.get("text", "")
    return result, detail


def preprocess_document(blocks: list[dict], ir: dict, native: dict | None, config: dict | None) -> dict:
    """Strip explicit non-body blocks and assemble accepted adjacent-page chains.

    All returned values are detached copies. No duplicate-text inference, OCR,
    missing-page interpolation or new continuation decision is performed.
    """
    document_id = ir.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("preprocess_requires_document_id")
    settings = (config or {}).get("preprocess", {})
    settings = settings if isinstance(settings, dict) else {}
    native = native if isinstance(native, dict) else {}
    copied = deepcopy(blocks)
    if len({block["block_id"] for block in copied}) != len(copied):
        raise ValueError("preprocess_duplicate_block_ids")
    body, excluded = [], []
    toc_units = _toc_units(ir)
    for block in copied:
        reason = _exclusion(block, toc_units, settings)
        if reason:
            excluded.append(dict(block, exclusion_reason=reason))
        else:
            body.append(block)
    by_id = {block["block_id"]: block for block in body}
    positions = {block["block_id"]: i for i, block in enumerate(body)}
    audits, viable, warnings = [], [], []
    outgoing, incoming = defaultdict(set), defaultdict(set)
    for index, relation in enumerate(ir.get("relations", []) or []):
        if not isinstance(relation, dict):
            warnings.append("invalid_relation_object_ignored")
            continue
        kind = relation.get("relation_type", relation.get("kind"))
        if kind not in CONTINUATIONS:
            continue
        source_id, target_id = relation.get("source_id", relation.get("source")), relation.get("target_id", relation.get("target"))
        audit = {"relation_id": relation.get("relation_id", relation.get("id", f"relation_{index + 1}")),
                 "relation_type": kind, "source_id": source_id, "target_id": target_id, "status": "rejected"}
        audits.append(audit)
        left = by_id.get(source_id) if isinstance(source_id, str) else None
        right = by_id.get(target_id) if isinstance(target_id, str) else None
        accepted = _accepted(relation)
        if accepted and left is not None and right is not None:
            outgoing[source_id].add(target_id)
            incoming[target_id].add(source_id)
        reason = None
        if not settings.get("merge_cross_page", True):
            reason = "cross_page_merge_disabled"
        elif not accepted:
            reason = "relation_not_explicitly_accepted"
        elif not isinstance(source_id, str) or not isinstance(target_id, str):
            reason = "invalid_relation_endpoint"
        elif left is None or right is None:
            reason = "missing_or_excluded_endpoint"
        elif any(_document_id(block, document_id) != document_id for block in (left, right)) or any(
            relation.get(key, document_id) != document_id for key in ("document_id", "source_document_id", "target_document_id")
        ):
            reason = "cross_document_relation"
        elif left.get("unit_type") != "physical_page" or right.get("unit_type") != "physical_page":
            reason = "physical_pages_required"
        elif type(left.get("page_number")) is not int or type(right.get("page_number")) is not int or left["page_number"] < 1 or right["page_number"] != left["page_number"] + 1:
            reason = "nonconsecutive_physical_pages"
        elif type(left.get("unit_index")) is not int or type(right.get("unit_index")) is not int or right["unit_index"] != left["unit_index"] + 1:
            reason = "nonconsecutive_units"
        elif positions[target_id] != positions[source_id] + 1:
            reason = "intervening_body_block"
        elif kind == "paragraph_continuation" and (left.get("kind") not in {"paragraph", "list_item"} or left.get("kind") != right.get("kind")):
            reason = "incompatible_block_types"
        elif kind != "paragraph_continuation" and (left.get("kind") != "table" or right.get("kind") != "table"):
            reason = "incompatible_block_types"
        elif kind != "paragraph_continuation" and _horizontal_table(relation):
            reason = "horizontal_table_merge_not_supported"
        elif kind != "paragraph_continuation" and _columns(left, native) is not None and _columns(right, native) is not None and _columns(left, native) != _columns(right, native):
            reason = "incompatible_table_columns"
        if reason:
            audit["reason"] = reason
        else:
            viable.append((relation, audit))
    successors = {}
    for relation, audit in viable:
        source_id, target_id = audit["source_id"], audit["target_id"]
        if len(outgoing[source_id]) > 1 or len(incoming[target_id]) > 1:
            audit["reason"] = "ambiguous_relation_branch"
        elif source_id in successors:
            audit["reason"] = "duplicate_relation_edge"
        else:
            successors[source_id] = (target_id, relation, audit)
    consumed, assembled = set(), []
    for block in body:
        ident = block["block_id"]
        if ident in consumed:
            continue
        result, cursor = block, ident
        fragments, relation_ids = [deepcopy(block)], []
        while cursor in successors:
            target_id, relation, audit = successors[cursor]
            if target_id in consumed:
                audit["reason"] = "endpoint_already_consumed"
                break
            fragments.append(deepcopy(by_id[target_id]))
            relation_ids.append(audit["relation_id"])
            result, detail = _join(result, by_id[target_id], relation, native, fragments, relation_ids)
            audit.update(status="merged", reason="accepted_adjacent_continuation", **detail)
            consumed.add(target_id)
            cursor = target_id
        assembled.append(result)
    return {"blocks": assembled, "excluded": excluded, "merges": audits, "warnings": warnings}
