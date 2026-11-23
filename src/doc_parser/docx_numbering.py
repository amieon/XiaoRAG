"""DOCX OOXML 自动编号提取；只读 numPr/numbering.xml，不根据可见文本猜列表。"""
from __future__ import annotations

from typing import Any


def _roman(number: int) -> str:
    values = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
              (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
              (5, "V"), (4, "IV"), (1, "I"))
    result = []
    for value, label in values:
        while number >= value:
            result.append(label)
            number -= value
    return "".join(result) or str(number)


def _letter(number: int) -> str:
    result = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result or str(number)


def _format_number(number: int, fmt: str) -> str:
    if fmt == "decimalZero":
        return f"{number:02d}"
    if fmt == "lowerLetter":
        return _letter(number).lower()
    if fmt == "upperLetter":
        return _letter(number)
    if fmt == "lowerRoman":
        return _roman(number).lower()
    if fmt == "upperRoman":
        return _roman(number)
    # 中文计数等复杂格式先保留确定性的数值，格式名仍完整记录，不伪装还原效果。
    return str(number)


class DocxNumbering:
    """维护文档顺序上的列表计数器；numId 只在当前 DOCX 内有效。"""

    def __init__(self, document) -> None:
        from docx.oxml.ns import qn

        self.qn = qn
        self._counters: dict[str, dict[int, int]] = {}
        self._nums: dict[str, dict[str, Any]] = {}
        self._abstracts: dict[str, dict[int, dict[str, Any]]] = {}
        try:
            root = document.part.numbering_part.element
        except (AttributeError, NotImplementedError):
            root = None
        if root is None:
            return
        for num in root.findall(qn("w:num")):
            num_id = num.get(qn("w:numId"))
            abstract = num.find(qn("w:abstractNumId"))
            if num_id is None or abstract is None:
                continue
            overrides = {}
            for override in num.findall(qn("w:lvlOverride")):
                try:
                    level = int(override.get(qn("w:ilvl")))
                    start = override.find(qn("w:startOverride"))
                    if start is not None:
                        overrides[level] = int(start.get(qn("w:val")))
                except (TypeError, ValueError):
                    continue
            self._nums[num_id] = {
                "abstract_id": abstract.get(qn("w:val")), "start_overrides": overrides,
            }
        for abstract in root.findall(qn("w:abstractNum")):
            abstract_id = abstract.get(qn("w:abstractNumId"))
            levels = {}
            for node in abstract.findall(qn("w:lvl")):
                try:
                    level = int(node.get(qn("w:ilvl")))
                except (TypeError, ValueError):
                    continue
                levels[level] = {
                    "start": self._int_child(node, "w:start", 1),
                    "format": self._child_value(node, "w:numFmt", "decimal"),
                    "text": self._child_value(node, "w:lvlText", f"%{level + 1}."),
                }
            if abstract_id is not None:
                self._abstracts[abstract_id] = levels

    def _child_value(self, node, name: str, default: str) -> str:
        child = node.find(self.qn(name))
        return child.get(self.qn("w:val"), default) if child is not None else default

    def _int_child(self, node, name: str, default: int) -> int:
        try:
            return int(self._child_value(node, name, str(default)))
        except (TypeError, ValueError):
            return default

    def _num_pr(self, paragraph):
        direct = getattr(getattr(paragraph, "_p", None), "pPr", None)
        properties = direct.find(self.qn("w:numPr")) if direct is not None else None
        if properties is not None:
            return properties, False
        style = getattr(paragraph, "style", None)
        style_properties = getattr(getattr(style, "element", None), "pPr", None)
        properties = style_properties.find(self.qn("w:numPr")) if style_properties is not None else None
        return (properties, True) if properties is not None else (None, False)

    def list_structure(self, paragraph, native_id: str) -> dict[str, Any] | None:
        properties, style_based = self._num_pr(paragraph)
        if properties is None:
            return None
        num_node = properties.find(self.qn("w:numId"))
        level_node = properties.find(self.qn("w:ilvl"))
        if num_node is None:
            return None
        num_id = num_node.get(self.qn("w:val"))
        if num_id is None or num_id == "0":
            return None
        try:
            level = int(level_node.get(self.qn("w:val"))) if level_node is not None else 0
        except (TypeError, ValueError):
            return None
        if level < 0:
            return None
        definition = self._nums.get(num_id)
        if definition is None:
            # 明确有 numPr 但定义缺失时仍保留列表证据，编号值不伪造。
            return {
                "source": "docx_numbering", "native_id": native_id,
                "num_id": num_id, "abstract_num_id": None, "level": level,
                "number_format": None, "ordinal": None, "rendered_marker": None,
                "style_based": style_based, "numbering_status": "definition_missing",
            }
        abstract_id = definition["abstract_id"]
        level_definition = self._abstracts.get(abstract_id, {}).get(level, {})
        start = definition["start_overrides"].get(level, level_definition.get("start", 1))
        counters = self._counters.setdefault(num_id, {})
        counters[level] = counters.get(level, start - 1) + 1
        for deeper in [value for value in counters if value > level]:
            del counters[deeper]
        ordinal = counters[level]
        number_format = level_definition.get("format", "decimal")
        marker = level_definition.get("text", f"%{level + 1}.")
        for index in range(9):
            definition_at_index = self._abstracts.get(abstract_id, {}).get(index, {})
            value = counters.get(index, definition_at_index.get("start", 1))
            marker = marker.replace(f"%{index + 1}", _format_number(value, definition_at_index.get("format", "decimal")))
        return {
            "source": "docx_numbering", "native_id": native_id,
            "num_id": num_id, "abstract_num_id": abstract_id, "level": level,
            "number_format": number_format, "ordinal": ordinal,
            "rendered_marker": marker, "style_based": style_based,
            "numbering_status": "resolved",
        }
