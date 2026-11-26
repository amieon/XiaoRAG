"""标题识别后的文档级父子关系恢复。

模式：none / streaming / streaming-context / global。模型只选择已有父节点，
程序统一计算层级并验证引用，模型名称与规模由运行环境决定。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

from .heading_classifier import HEADING_LABEL, HeadingExample
from .tree import numbering_level
from .zero_shot_classifier import _content, _json_content


HIERARCHY_MODES = ("none", "streaming", "streaming-context", "global")


@dataclass
class _HeadingNode:
    local_id: str
    block_id: str
    text: str
    level: int
    parent_local_id: str | None


class HeadingHierarchyResolver:
    def __init__(self, client: Any, *, mode: str = "streaming") -> None:
        if mode not in HIERARCHY_MODES:
            raise ValueError(f"Unknown hierarchy mode: {mode}")
        self.client = client
        self.mode = mode

    @staticmethod
    def _anchor_level(example: HeadingExample, row: dict[str, Any]) -> int | None:
        if row.get("native_heading_count"):
            levels = [
                item.get("heading_level") for item in row.get("native_evidence", [])
                if type(item.get("heading_level")) is int
            ]
            if len(levels) == 1:
                return levels[0]
        return numbering_level(example.text.strip())

    @staticmethod
    def _attach_anchor(stack: list[_HeadingNode], level: int) -> tuple[_HeadingNode | None, list[_HeadingNode]]:
        retained = list(stack)
        while retained and retained[-1].level >= level:
            retained.pop()
        closed = stack[len(retained):]
        return (retained[-1] if retained else None), closed

    def _payload(self, system: str, user: Any) -> dict[str, Any]:
        payload = {
            "model": self.client.model,
            "temperature": 0,
            "max_tokens": self.client.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        }
        if self.client.use_response_format:
            payload["response_format"] = {"type": "json_object"}
        if self.client.disable_thinking:
            payload["enable_thinking"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _global_parents(
        self,
        nodes: Sequence[tuple[HeadingExample, dict[str, Any], int | None]],
        local_ids: Sequence[str],
    ) -> dict[str, str | None]:
        query_ids = [
            local_ids[index] for index, (_, _, anchor) in enumerate(nodes)
            if anchor is None and index > 0
        ]
        if not query_ids:
            return {}
        query_set = set(query_ids)
        headings = [{
            "heading_id": local_ids[index],
            "text": example.text,
            "anchor_level": anchor,
        } for index, (example, _, anchor) in enumerate(nodes)]
        system = """你是文档章节结构恢复器。根据完整标题序列判断标题父子关系。
只返回query_heading_ids中的标题；其他标题仅作上下文和父节点候选。
parent_id只能是当前标题之前的heading_id，顶级标题为null。不得修改、增加或遗漏ID。
同级章节选择同一个父节点；子章节选择最近的语义父标题。
只输出JSON：{"predictions":[{"heading_id":"h001","parent_id":"h000"}]}。"""
        response = self.client._post(self._payload(system, {
            "headings": headings, "query_heading_ids": query_ids,
        }))
        decoded = _json_content(_content(response))
        rows = decoded.get("predictions") if isinstance(decoded, dict) else None
        if not isinstance(rows, list):
            raise ValueError("Global hierarchy response must contain predictions")
        result: dict[str, str | None] = {}
        for row in rows:
            if not isinstance(row, dict) or "parent_id" not in row:
                raise ValueError("Global hierarchy row requires heading_id and parent_id")
            heading_id, parent_id = row.get("heading_id"), row.get("parent_id")
            if heading_id not in query_set or heading_id in result:
                raise ValueError(f"Unknown or duplicate global heading_id: {heading_id!r}")
            if parent_id is not None:
                if parent_id not in local_ids:
                    raise ValueError(f"Unknown global parent_id: {parent_id!r}")
                if local_ids.index(parent_id) >= local_ids.index(heading_id):
                    raise ValueError("Global parent must precede its child")
            result[heading_id] = parent_id
        if set(result) != query_set:
            raise ValueError(f"Global hierarchy missing headings: {sorted(query_set - set(result))[:3]}")
        return result

    def _streaming_parent(
        self,
        example: HeadingExample,
        local_id: str,
        stack: list[_HeadingNode],
        closed: list[_HeadingNode],
    ) -> str | None:
        candidates = [{"heading_id": "root", "text": "文档根节点"}] + [
            {"heading_id": node.local_id, "text": node.text} for node in stack
        ]
        user: dict[str, Any] = {
            "current": {"heading_id": local_id, "text": example.text},
            "active_path": candidates,
        }
        if self.mode == "streaming-context":
            user["previous_text"] = (example.context.get("previous") or {}).get("text")
            user["next_text"] = (example.context.get("next") or {}).get("text")
            user["recent_closed_siblings"] = [
                {"heading_id": node.local_id, "text": node.text} for node in closed[-3:]
            ]
        system = """你是文档章节树构建器。为当前标题从active_path中选择一个父节点。
active_path按根到上一标题排列。若当前标题与路径末尾同级，选择末尾的父节点；
若它是末尾的孩子，选择末尾；若开启更高层章节，选择相应祖先或root。
只能输出一个已有父节点ID，不得创造节点或输出解释。
只输出JSON：{"parent_id":"h003"}；顶级标题使用{"parent_id":"root"}。"""
        response = self.client._post(self._payload(system, user))
        decoded = _json_content(_content(response))
        if not isinstance(decoded, dict) or "parent_id" not in decoded:
            raise ValueError("Streaming hierarchy response requires parent_id")
        parent_id = decoded["parent_id"]
        if parent_id == "root":
            return None
        valid = {node.local_id for node in stack}
        if parent_id not in valid:
            raise ValueError(f"Streaming parent is not in active path: {parent_id!r}")
        return parent_id

    def resolve(
        self,
        examples: Sequence[HeadingExample],
        predictions: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(examples) != len(predictions):
            raise ValueError("Hierarchy examples and predictions count mismatch")
        rows = [dict(row) for row in predictions]
        headings = [index for index, row in enumerate(rows) if row.get("label") == HEADING_LABEL]
        if self.mode == "none":
            for index in headings:
                rows[index]["heading_level"] = 1
                rows[index]["hierarchy_parent_id"] = None
                rows[index]["hierarchy_source"] = "baseline_no_tree"
                rows[index]["hierarchy_mode"] = "none"
            return rows
        if not headings:
            return rows

        local_ids = [f"h{position:03d}" for position in range(len(headings))]
        nodes = [
            (examples[index], rows[index], self._anchor_level(examples[index], rows[index]))
            for index in headings
        ]
        global_parents = self._global_parents(nodes, local_ids) if self.mode == "global" else {}
        resolved: list[_HeadingNode] = []
        stack: list[_HeadingNode] = []
        closed: list[_HeadingNode] = []

        for position, (example, row, anchor) in enumerate(nodes):
            local_id = local_ids[position]
            parent: _HeadingNode | None = None
            if anchor is not None:
                parent, popped = self._attach_anchor(stack, anchor)
                stack = [node for node in stack if node.level < anchor]
                level = anchor
                source = "native" if row.get("native_heading_count") else "explicit_numbering"
            elif self.mode == "global":
                model_called = position > 0
                parent_local = global_parents[local_id] if model_called else None
                parent = next(
                    (node for node in reversed(resolved) if node.local_id == parent_local), None,
                )
                if parent_local is not None and parent is None:
                    raise ValueError(f"Unresolved global parent: {parent_local}")
                level = parent.level + 1 if parent else 1
                retained = [node for node in stack if node.level < level]
                popped = stack[len(retained):]
                stack = retained
                source = "global_model" if model_called else "document_root"
            else:
                model_called = bool(stack)
                if not model_called:
                    parent_local = None
                else:
                    parent_local = self._streaming_parent(example, local_id, stack, closed)
                parent = next((node for node in stack if node.local_id == parent_local), None)
                if parent_local is not None and parent is None:
                    raise ValueError(f"Unresolved streaming parent: {parent_local}")
                level = parent.level + 1 if parent else 1
                if parent is None:
                    popped = stack
                    stack = []
                else:
                    parent_index = stack.index(parent)
                    popped = stack[parent_index + 1:]
                    stack = stack[:parent_index + 1]
                source = (
                    "streaming_context_model" if self.mode == "streaming-context" else "streaming_model"
                ) if model_called else "document_root"
            closed.extend(popped)
            node = _HeadingNode(
                local_id=local_id, block_id=example.block_id, text=example.text,
                level=level, parent_local_id=parent.local_id if parent else None,
            )
            resolved.append(node)
            stack.append(node)
            row["heading_level"] = level
            row["hierarchy_parent_id"] = parent.block_id if parent else None
            row["hierarchy_source"] = source
            row["hierarchy_mode"] = self.mode
        return rows
