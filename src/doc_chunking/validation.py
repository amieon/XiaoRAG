"""Independent integrity checks for emitted local chunk records."""
from collections import Counter
import json

from .model import normalize_document


def validate_result(result, ir, native=None):
    originals = {b.get("block_id", b.get("id")): b for p in ir["pages"] for b in p["blocks"]}
    canonical = {b["block_id"]: b["sources"][0] for b in normalize_document(ir)}
    parents, children = result["parents"], result["children"]
    all_chunks = parents + children
    ids = [c["chunk_id"] for c in all_chunks]
    issues = []
    issue_details = {}
    if len(ids) != len(set(ids)):
        issues.append("duplicate_chunk_id")
    by_parent = Counter(c["parent_id"] for c in children)
    parent_ids = {p["chunk_id"] for p in parents}
    if set(by_parent) != parent_ids:
        issues.append("parent_child_membership")
    for parent in parents:
        actual = [c["chunk_id"] for c in children if c["parent_id"] == parent["chunk_id"]]
        if parent["child_ids"] != actual:
            issues.append("child_id_list")
        child_sources = {r["block_id"] for c in children if c["parent_id"] == parent["chunk_id"] for r in c["source_refs"]}
        parent_sources = set(parent["source_block_ids"])
        if child_sources != parent_sources:
            issues.append("child_source_coverage")
            issue_details.setdefault("child_source_coverage", []).append({
                "parent_id": parent["chunk_id"],
                "missing_block_ids": sorted(parent_sources - child_sources),
                "unexpected_block_ids": sorted(child_sources - parent_sources),
                "parent_source_block_ids": parent["source_block_ids"],
                "child_source_block_ids": sorted(child_sources),
            })
        siblings = [c for c in children if c["parent_id"] == parent["chunk_id"]]
        if parent["metadata"].get("body_rendering") == "format_override":
            if len(siblings) != 1 or siblings[0]["body_text"] != parent["body_text"]:
                issues.append("format_body_content_changed")
        else:
            expected_parts = {p["logical_block_id"]: p for p in parent["metadata"].get("body_parts", [])}
            spans = {key: [] for key in expected_parts}
            for child in siblings:
                for part in child["metadata"].get("body_parts", []):
                    expected = expected_parts.get(part["logical_block_id"])
                    if expected is None:
                        issues.append("unknown_logical_body_part")
                        continue
                    start, end = part.get("start"), part.get("end")
                    if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(expected["text"])
                            or part.get("text") != expected["text"][start:end]):
                        issues.append("child_body_slice_changed")
                    else:
                        spans[part["logical_block_id"]].append((start, end))
            for key, ranges in spans.items():
                position = 0
                for start, end in sorted(ranges):
                    if start != position:
                        issues.append("logical_body_gap_or_overlap")
                    position = end
                expected_length = len(expected_parts[key]["text"])
                if position != expected_length or (not ranges and expected_length > 0):
                    issues.append("logical_body_gap_or_overlap")
    for chunk in all_chunks:
        if chunk["document_id"] != ir["document_id"]:
            issues.append("chunk_document_identity")
        if chunk["level"] == "child" and chunk["parent_id"] not in parent_ids:
            issues.append("dangling_parent")
        expected_text = (chunk["context_prefix"] + "\n\n" if chunk["context_prefix"] else "") + chunk["body_text"]
        if chunk["text"] != expected_text or chunk["length"]["characters"] != len(chunk["text"]):
            issues.append("chunk_text_or_length")
        if chunk["length"]["token_count"] is not None or chunk["metadata"]["token_limit_verified"]:
            issues.append("unverified_token_count")
        if set(chunk["source_block_ids"]) != {r["block_id"] for r in chunk["source_refs"]}:
            issues.append("source_id_inventory")
        if chunk["metadata"].get("body_rendering") == "ir_blocks":
            parts = chunk["metadata"].get("body_parts", [])
            if "\n\n".join(p["text"] for p in parts) != chunk["body_text"]:
                issues.append("body_rendering_changed")
            for part in parts:
                refs = part.get("source_refs", [])
                if len(refs) == 1 and refs[0].get("block_id") in originals:
                    ref = refs[0]
                    raw_text = originals[ref["block_id"]].get("text", "")
                    start, end = ref.get("char_start", 0), ref.get("char_end", len(raw_text))
                    if type(start) is int and type(end) is int and part["text"] != raw_text[start:end]:
                        issues.append("body_differs_from_original_ir")
        for ref in chunk["source_refs"]:
            original = originals.get(ref["block_id"])
            if original is None:
                issues.append("source_block_not_in_ir")
                continue
            expected = canonical[ref["block_id"]]
            if any(ref.get(k) != expected.get(k) for k in ("unit_id", "unit_index", "unit_type", "page_number", "bbox", "source_type", "confidence", "native_id")):
                issues.append("source_location_differs_from_ir")
            if "char_start" in ref or "char_end" in ref:
                start, end = ref.get("char_start"), ref.get("char_end")
                if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(original.get("text", "")):
                    issues.append("invalid_source_character_span")
        if native is not None:
            for ref in chunk["native_refs"]:
                pointer = ref.get("pointer")
                if not isinstance(pointer, str) or not pointer.startswith("/"):
                    issues.append("invalid_native_pointer")
                    continue
                node = native
                try:
                    for key in pointer[1:].split("/"):
                        key = key.replace("~1", "/").replace("~0", "~")
                        node = node[int(key)] if isinstance(node, list) else node[key]
                except (TypeError, ValueError, KeyError, IndexError):
                    issues.append("invalid_native_pointer")
    if issues:
        issue = sorted(set(issues))[0]
        message = "chunk_integrity_" + issue
        if issue in issue_details:
            message += ": " + json.dumps(issue_details[issue], ensure_ascii=False, separators=(",", ":"))
        raise ValueError(message)
    return {"status": "passed", "check_count": len(all_chunks) * 4 + len(parents) * 2 + 2,
            "error_count": 0}
