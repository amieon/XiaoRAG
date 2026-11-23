"""把任意深度章节树变成现有两层父子分块可消费的 flat groups。"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


def chapter_tree_to_groups(
    tree: dict[str, Any], blocks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if tree.get("schema_version") != "chapter-tree.v1":
        raise ValueError("groups_require_chapter-tree.v1")
    block_map = {block["block_id"]: block for block in blocks}
    nodes = {node["section_id"]: node for node in tree["nodes"]}
    root_id = tree["root_section_id"]
    if root_id not in nodes or len(block_map) != len(blocks):
        raise ValueError("invalid_tree_root_or_duplicate_blocks")
    groups: list[dict[str, Any]] = []
    heading_ids = {
        item["block_id"] for item in tree.get("block_decisions", [])
        if item.get("is_heading") is True
    }

    def emit(
        node: dict[str, Any], block_ids: list[str], *, kind: str,
        inherited: list[str],
    ) -> None:
        if not block_ids:
            return
        group_index = len(groups)
        heading_block_id = node.get("heading_block_id")
        selected = []
        for block_id in block_ids:
            block = dict(block_map[block_id])
            block["metadata"] = dict(block.get("metadata") or {})
            block["metadata"]["chapter_tree"] = {
                "section_id": node["section_id"],
                "title_path": list(node["title_path"]),
                "heading_source": node.get("heading_source"),
                "is_section_heading": block_id == heading_block_id,
            }
            selected.append(block)
        groups.append({
            "group_id": f"{tree['document_id']}:g{group_index:06d}",
            "group_index": group_index,
            "parent_chunk_index": group_index,
            "kind": kind,
            "clause_path": [],
            "title_path": list(node["title_path"]),
            "blocks": selected,
            "atomic": False,
            "strategy": "chapter_tree",
            "source_block_ids": list(block_ids),
            "context": {
                "section_id": node["section_id"],
                "section_kind": node["kind"],
                "section_level": node["level"],
                "section_depth": node["depth"],
                "boundary_rule": "first_contentful_descendant_inherits_empty_ancestors",
                "inherited_heading_block_ids": inherited,
                "heading_only_group": all(block_id in heading_ids for block_id in block_ids),
            },
        })

    def visit(node_id: str, inherited_ids: list[str]) -> list[str]:
        node = nodes[node_id]
        direct_ids = list(node["block_ids"])
        content_ids = list(node.get("content_block_ids", []))
        pending = inherited_ids + direct_ids
        if node_id == root_id:
            if content_ids:
                emit(node, direct_ids, kind="flat" if tree["stats"]["flat_document"] else "preamble", inherited=[])
                pending = []
            for child_id in node["child_section_ids"]:
                pending = visit(child_id, [])
            if pending:
                raise ValueError("root_preamble_coverage_failure")
            return []
        if content_ids:
            emit(node, pending, kind="section", inherited=inherited_ids)
            pending = []
        for child_id in node["child_section_ids"]:
            pending = visit(child_id, pending)
        if pending:
            # 标题后没有任何正文也不能静默丢失；明确保留 heading-only group。
            emit(node, pending, kind="section", inherited=inherited_ids)
            pending = []
        return pending

    visit(root_id, [])
    covered = [block_id for group in groups for block_id in group["source_block_ids"]]
    expected = [block["block_id"] for block in blocks]
    if covered != expected:
        counts = Counter(covered)
        missing = sorted(set(expected) - set(covered))
        duplicated = sorted(block_id for block_id, count in counts.items() if count > 1)
        raise ValueError(f"group_source_coverage:missing={missing[:3]},duplicated={duplicated[:3]}")
    return groups


def chapter_groups_document(
    tree: dict[str, Any], blocks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    groups = chapter_tree_to_groups(tree, blocks)
    return {
        "schema_version": "chapter-groups.v1",
        "document_id": tree["document_id"],
        "group_order": "physical_reading_order",
        "parent_child_depth": 2,
        "groups": groups,
        "self_check": {
            "status": "passed",
            "input_block_count": len(blocks),
            "represented_block_count": sum(len(group["source_block_ids"]) for group in groups),
            "group_count": len(groups),
            "heading_only_group_count": sum(group["context"]["heading_only_group"] for group in groups),
        },
    }
