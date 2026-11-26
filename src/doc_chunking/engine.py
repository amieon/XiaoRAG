"""章节树分组与两层父子块输出；普通文档不再整篇四选一。"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from chapter_tree.tree import ChapterTreeConfig, build_chapter_tree
from chapter_tree.groups import chapter_tree_to_groups

from .cdc import split_cdc
from .model import normalize_document, unique_refs
from .prepare import prepare_document


CLAUSE_RE = re.compile(r"^第\s*([0-9一二三四五六七八九十百千万零〇两]+)\s*条(?:\s|$|[：:，,。])")


def load_config(path=None):
    path = Path(path or Path(__file__).with_name("config.json"))
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("configuration_requires_object")
    for section in ("preprocess", "formats", "sizing"):
        if not isinstance(config.get(section), dict):
            raise ValueError("missing_configuration_section_" + section)
    sizing = config["sizing"]
    for name in ("min_chars", "target_chars", "max_chars", "window_chars"):
        if type(sizing.get(name)) is not int or sizing[name] <= 0:
            raise ValueError("invalid_" + name)
    if not sizing["min_chars"] <= sizing["target_chars"] <= sizing["max_chars"]:
        raise ValueError("cdc_requires_min_target_max_order")
    if type(sizing.get("boundary_snap_chars")) is not int or sizing["boundary_snap_chars"] < 0:
        raise ValueError("invalid_boundary_snap_chars")
    if sizing.get("algorithm") != "cdc-natural-text-v1":
        raise ValueError("unsupported_chunking_algorithm")
    if sizing.get("length_unit") != "unicode_codepoints" or sizing.get("overlap") != 0:
        raise ValueError("chapter_tree_requires_character_lengths_and_zero_overlap")
    if type(config.get("clause_boundaries", True)) is not bool:
        raise ValueError("invalid_clause_boundaries")


def _fragment(block, start, end, reason):
    item = deepcopy(block)
    item["text"] = block["text"][start:end]
    item["logical_text_span"] = {"block_id": block["block_id"], "start": start, "end": end}
    item["split_reason"] = reason
    if len(item.get("sources", [])) == 1:
        ref = item["sources"][0]
        if ref.get("block_id") == block["block_id"] or "char_start" in ref:
            offset = ref.get("char_start", 0)
            ref.update(char_start=offset + start, char_end=offset + end)
    return item


def _list_structures(block):
    metadata = block.get("metadata") or {}
    values = metadata.get("list_structures", [])
    return values if isinstance(values, list) else []


def _is_list_unit(block):
    metadata = block.get("metadata") or {}
    if metadata.get("chapter_tree", {}).get("is_section_heading") is True:
        return False  # Heading＋numPr 仍是章节标题，不降级成列表项。
    return block.get("kind") == "list_item" or bool(_list_structures(block))


def _is_explicit_list_continuation(block, previous_blocks):
    metadata = block.get("metadata") or {}
    if metadata.get("list_continuation") is True:
        return True
    parent = metadata.get("parent_id", metadata.get("list_parent_id"))
    return parent is not None and parent in {item["block_id"] for item in previous_blocks}


def _clause_label(block):
    first = next((line.strip() for line in block.get("text", "").splitlines() if line.strip()), "")
    match = CLAUSE_RE.match(first)
    return f"第{match.group(1)}条" if match else None


def _cdc_text_unit(unit):
    """章节标题可与正文一起打包；列表、公式及其他原子 IR 块不进入 CDC。"""
    return all(
        block["kind"] in ("paragraph", "text", "body", "heading", "section_heading", "document_title")
        and not _is_list_unit(block)
        and not (block.get("metadata") or {}).get("list_continuation")
        and not (block.get("metadata") or {}).get("math")
        for block in unit
    )


def _cdc_segment_parts(units, sizing, clause_path):
    """在连续正文上切分，再把全局区间映射回原始 IR 块，证据不丢字、不重叠。"""
    blocks = [block for unit in units for block in unit]
    layout, cursor = [], 0
    for block in blocks:
        end = cursor + len(block["text"])
        layout.append((block, cursor, end))
        cursor = end + 2  # children 渲染时仍使用 \n\n 连接原始块。
    text = "\n\n".join(block["text"] for block in blocks)
    block_starts = [item[1] for item in layout]
    parts = []
    for start, end, reason in split_cdc(text, sizing):
        fragments = []
        index = max(0, bisect_right(block_starts, start) - 1)
        while index < len(layout) and layout[index][1] < end:
            block, block_start, block_end = layout[index]
            left, right = max(start, block_start), min(end, block_end)
            if left < right:
                fragments.append(_fragment(block, left - block_start, right - block_start, reason))
            index += 1
        if not fragments:
            raise ValueError("cdc_range_has_no_source_text")
        parts.append({
            "blocks": fragments, "text": None,
            "split_reasons": sorted({block["split_reason"] for block in fragments}),
            "clause_path": clause_path,
        })
    return parts


def _child_parts(group, config):
    if group.get("text_override") is not None or group.get("atomic"):
        return [{"blocks": group["blocks"], "text": group.get("text_override"),
                 "split_reasons": ["atomic_structural_unit"], "clause_path": group.get("clause_path", [])}]
    units = []
    for block in group["blocks"]:
        if _is_list_unit(block):
            units.append([dict(block, split_reason="atomic_list_item")])
        elif units and _is_list_unit(units[-1][0]) and _is_explicit_list_continuation(block, units[-1]):
            units[-1].append(dict(block, split_reason="explicit_list_continuation"))
        elif _cdc_text_unit([block]):
            units.append([dict(block, split_reason="source_block")])
        else:
            units.append([dict(block, split_reason="atomic_ir_block")])

    # 先保留条款及原子单元边界；连续普通正文段内部不再按 target 贪心错位。
    segments, pending = [], []
    mode, current_clause = None, None
    clause_enabled = config.get("clause_boundaries", True)

    def flush_segment():
        nonlocal pending, mode, current_clause
        if pending:
            segments.append({
                "units": pending, "cdc": mode,
                "clause_path": [current_clause] if current_clause else [],
            })
        pending, mode, current_clause = [], None, None

    for unit in units:
        label = _clause_label(unit[0]) if clause_enabled else None
        if label and pending:
            only_headings = all(
                (block.get("metadata") or {}).get("chapter_tree", {}).get("is_section_heading") is True
                for pending_unit in pending for block in pending_unit
            )
            if not only_headings:
                flush_segment()
        unit_mode = _cdc_text_unit(unit)
        if pending and unit_mode != mode:
            flush_segment()
        if not pending:
            mode = unit_mode
        if label:
            current_clause = label
        pending.append(unit)
    flush_segment()

    parts, target = [], config["sizing"]["target_chars"]
    for segment in segments:
        if segment["cdc"]:
            parts.extend(_cdc_segment_parts(segment["units"], config["sizing"], segment["clause_path"]))
            continue
        pending, length = [], 0

        def flush_atomic_run():
            nonlocal pending, length
            if pending:
                parts.append({
                    "blocks": pending, "text": None,
                    "split_reasons": sorted({block["split_reason"] for block in pending}),
                    "clause_path": segment["clause_path"],
                })
                pending, length = [], 0

        for unit in segment["units"]:
            size = sum(len(block["text"]) for block in unit) + 2 * (len(unit) - 1)
            if pending and length + size + 2 > target:
                flush_atomic_run()
            pending.extend(unit)
            length += size + (2 if length else 0)
        flush_atomic_run()
    return parts


def _chunk_record(document_id, chunk_id, level, parent_id, group, blocks, text, channel, config, reasons=(), clause_path=None):
    refs = unique_refs(blocks)
    title_path = group.get("title_path", [])
    selected_clauses = group.get("clause_path", []) if clause_path is None else clause_path
    prefix = " > ".join(dict.fromkeys([*title_path, *selected_clauses]))
    body = text if text is not None else "\n\n".join(block["text"] for block in blocks)
    full = prefix + "\n\n" + body if prefix else body
    pages = sorted({ref["page_number"] for ref in refs if ref.get("page_number") is not None})
    body_parts = []
    for block in blocks:
        span = block.get("logical_text_span") or {
            "block_id": block["block_id"], "start": 0, "end": len(block["text"]),
        }
        body_parts.append({
            "logical_block_id": span["block_id"], "start": span["start"], "end": span["end"],
            "text": block["text"], "source_refs": unique_refs([block]),
        })
    keys = {
        "refs", "count", "security", "classification", "security_level", "caption_ids",
        "footnote_ids", "heading_level", "list_level", "list_structures", "list_continuation",
        "list_parent_id", "detection_origin", "chunk_preprocess", "fragments", "chapter_tree",
    }
    block_metadata = [
        {"block_id": block["block_id"], "metadata": {
            key: deepcopy(value) for key, value in block["metadata"].items() if key in keys
        }} for block in blocks if keys & block["metadata"].keys()
    ]
    return {
        "schema_version": "structural-chunk.v1", "document_id": document_id,
        "chunk_id": chunk_id, "level": level, "parent_id": parent_id, "child_ids": [],
        "strategy": group.get("strategy"), "format_channel": channel,
        "kind": group.get("kind", "body"), "title_path": title_path, "clause_path": selected_clauses,
        "context_prefix": prefix, "text": full, "body_text": body,
        "length": {"characters": len(full), "body_characters": len(body),
                   "unit": "unicode_codepoints", "token_count": None},
        "source_refs": refs, "source_block_ids": list(dict.fromkeys(ref["block_id"] for ref in refs)),
        "native_refs": deepcopy(group.get("native_refs", [])),
        "page_numbers": pages, "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "unit_ids": list(dict.fromkeys(ref["unit_id"] for ref in refs)),
        "metadata": {
            "context": deepcopy(group.get("context", {})),
            "atomic_structural_unit": bool(group.get("atomic")),
            "block_kinds": sorted({block["kind"] for block in blocks}),
            "split_reasons": list(reasons),
            "oversize_character_budget": len(full) > config["sizing"]["max_chars"],
            "token_limit_verified": False,
            "chunking": deepcopy(config["sizing"]),
            "body_rendering": "format_override" if group.get("text_override") is not None else "ir_blocks",
            "body_parts": body_parts if group.get("text_override") is None else [],
            "block_metadata": block_metadata,
            "document_metadata_ref": "preprocessing.json#/excluded",
            "logical_text_spans": [block["logical_text_span"] for block in blocks if "logical_text_span" in block],
        },
    }


def chunk_document(
    ir, native=None, source_manifest=None, config=None, *, tree_config=None,
    predictions=None,
):
    config = deepcopy(config) if config is not None else load_config()
    validate_config(config)
    native = {} if native is None else native
    source_manifest = {} if source_manifest is None else source_manifest
    if not isinstance(native, dict) or not isinstance(source_manifest, dict):
        raise ValueError("native_and_manifest_require_objects")
    if source_manifest.get("document_id", ir["document_id"]) != ir["document_id"]:
        raise ValueError("source_manifest_document_id_mismatch")
    origin = ir.get("source") if isinstance(ir.get("source"), dict) else {}
    hashes = [value for value in (
        ir.get("content_hash"), origin.get("sha256"), source_manifest.get("source_sha256"),
    ) if value is not None]
    if any(not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value) for value in hashes):
        raise ValueError("invalid_source_sha256")
    if len({value.lower() for value in hashes}) > 1:
        raise ValueError("original_source_hash_claim_mismatch")

    prepared = prepare_document(ir, native, source_manifest, config)
    formatted = prepared["format_result"]
    tree = None
    if formatted is not None:
        channel, strategy = formatted["channel"], formatted["strategy"]
        groups = formatted["groups"]
        rules = [{"rule": "format_channel", "matched": True}]
    else:
        channel, strategy = "document", "chapter_tree"
        tree = build_chapter_tree(
            prepared["blocks"], document_id=ir["document_id"],
            config=tree_config or ChapterTreeConfig(heading_mode="hybrid"),
            predictions=predictions,
        )
        groups = chapter_tree_to_groups(tree, prepared["blocks"])
        rules = [{"rule": "chapter_tree", "matched": True}]

    parents, children = [], []
    for index, group in enumerate(groups, 1):
        parent_id = f"{ir['document_id']}:p{index:05d}"
        parent = _chunk_record(
            ir["document_id"], parent_id, "parent", None, group,
            group["blocks"], group.get("text_override"), channel, config,
        )
        for child_index, part in enumerate(_child_parts(group, config), 1):
            child_id = f"{parent_id}:c{child_index:04d}"
            child = _chunk_record(
                ir["document_id"], child_id, "child", parent_id, group,
                part["blocks"], part["text"], channel, config,
                part["split_reasons"], part.get("clause_path"),
            )
            parent["child_ids"].append(child_id)
            children.append(child)
        parents.append(parent)

    blocks = normalize_document(ir)
    expected = {ref["block_id"] for ref in unique_refs(prepared["blocks"])}
    represented = {ref["block_id"] for parent in parents for ref in parent["source_refs"]}
    missing = sorted(expected - represented)
    if missing:
        raise ValueError("unrepresented_body_blocks")
    forbidden = {
        ref["block_id"] for block in prepared["excluded"]
        if block["kind"] not in ("speaker_notes", "speaker_note", "slide_notes")
        for ref in block.get("sources", [])
    }
    if represented & forbidden:
        raise ValueError("excluded_furniture_in_body")
    checks = {
        "input_block_count": len(blocks), "body_block_count": len(prepared["blocks"]),
        "excluded_block_count": len(prepared["excluded"]), "source_body_block_count": len(expected),
        "represented_source_body_block_count": len(expected & represented),
        "missing_source_block_count": len(missing), "parent_count": len(parents),
        "child_count": len(children),
        "merged_relation_count": sum(item.get("status") == "merged" for item in prepared["merges"]),
        "rejected_relation_count": sum(item.get("status") == "rejected" for item in prepared["merges"]),
        "oversize_child_count": sum(child["metadata"]["oversize_character_budget"] for child in children),
        "physical_positioned_child_count": sum(bool(child["page_numbers"]) for child in children),
        "source_locatable_child_count": sum(bool(child["source_refs"] or child["native_refs"]) for child in children),
        "clause_boundary_child_count": sum(bool(child["clause_path"]) for child in children),
        "list_atomic_child_count": sum("atomic_list_item" in child["metadata"]["split_reasons"] for child in children),
        "token_limit_verified": False,
    }
    warnings = list(prepared["warnings"]) + (tree.get("warnings", []) if tree else [])
    decision = {
        "schema_version": "chunk-decision.v1", "document_id": ir["document_id"],
        "format_channel": channel, "strategy": strategy,
        "feature_basis": "preprocessed_ir_chapter_tree" if not formatted else "explicit_format_and_native_structure",
        "measured_features": tree["stats"] if tree else formatted.get("metrics", {}),
        "rules": rules, "profile_audit": None, "counts": checks, "warnings": warnings,
        "length_policy": {**config["sizing"], "tokenizer": None, "token_limit_verified": False},
        "scope": "chapter_tree_groups_with_list_and_clause_child_boundaries",
    }
    return {
        "document_id": ir["document_id"], "parents": parents, "children": children,
        "chapter_tree": tree,
        "preprocessing": {
            "excluded": prepared["excluded"], "merges": prepared["merges"],
            "warnings": prepared["warnings"],
        },
        "decision": decision, "self_check": {"status": "passed", **checks},
    }
