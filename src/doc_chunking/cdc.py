"""普通正文的内容定义切分（CDC）与自然边界吸附；不保存历史切点。"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import re


_ALGORITHM = "cdc-natural-text-v1"
_MODULUS = (1 << 61) - 1
_BASE = 257
_STRONG_RE = re.compile(r"[。！？；;!?]+|\.(?:\s+|$)")
_PARAGRAPH_RE = re.compile(r"\n{2,}")
_NEWLINE_RE = re.compile(r"(?<!\n)\n(?!\n)")
_WEAK_RE = re.compile(r"[，、：:,](?:\s+)?")


def _rolling_hashes(text: str, window: int) -> list[int | None]:
    """有界 Unicode 字符窗口的多项式滚动哈希；修改不会影响窗口外的哈希状态。"""
    result: list[int | None] = [None] * (len(text) + 1)
    highest = pow(_BASE, window - 1, _MODULUS)
    value = 0
    for index, char in enumerate(text):
        if index >= window:
            outgoing = text[index - window]
            value = (value - ord(outgoing) * highest) % _MODULUS
        value = (value * _BASE + ord(char)) % _MODULUS
        if index + 1 >= window:
            result[index + 1] = value
    return result


def _natural_boundaries(text: str) -> list[tuple[int, int, str]]:
    """返回（切点，优先级，类型）；英文句点必须伴随空白或文本结尾。"""
    result = []
    for priority, pattern, name in (
        (0, _PARAGRAPH_RE, "paragraph"),
        (1, _STRONG_RE, "sentence"),
        (2, _NEWLINE_RE, "newline"),
        (3, _WEAK_RE, "comma"),
    ):
        result.extend((match.end(), priority, name) for match in pattern.finditer(text))
    return sorted(result)


def split_cdc(text: str, sizing: dict) -> list[tuple[int, int, str]]:
    """返回连续、无重叠的 Unicode 字符区间；长度上限只约束普通正文。"""
    if not isinstance(text, str):
        raise ValueError("cdc_text_requires_string")
    if not text:
        return []
    maximum = sizing["max_chars"]
    if len(text) <= maximum:
        return [(0, len(text), "whole_block")]
    minimum, target = sizing["min_chars"], sizing["target_chars"]
    window, snap = sizing["window_chars"], sizing["boundary_snap_chars"]
    hashes = _rolling_hashes(text, window)
    boundaries = _natural_boundaries(text)
    boundary_ends = [item[0] for item in boundaries]

    def boundaries_between(left: int, right: int):
        first = bisect_left(boundary_ends, left)
        last = bisect_right(boundary_ends, right)
        return boundaries[first:last]

    result = []
    start = 0
    while len(text) - start > maximum:
        earliest, hard_end = start + minimum, start + maximum
        candidate = next((
            end for end in range(earliest, hard_end + 1)
            if hashes[end] is not None and hashes[end] % target == 0
        ), None)
        chosen, reason = None, None
        if candidate is not None:
            nearby = boundaries_between(
                max(earliest, candidate - snap), min(hard_end, candidate + snap),
            )
            if nearby:
                # 先比较边界强度，再比较距离；距离相同时确定性选择候选点之后。
                end, _, name = min(nearby, key=lambda item: (
                    item[1], abs(item[0] - candidate), item[0] < candidate, item[0],
                ))
                chosen, reason = end, f"cdc_{name}_boundary"
            else:
                chosen, reason = candidate, "cdc_content_boundary"
        else:
            fallback = boundaries_between(start + target, hard_end)
            if fallback:
                # 兜底优先强边界；同级选择最后一个，避免制造不必要的短尾块。
                end, _, name = min(fallback, key=lambda item: (item[1], -item[0]))
                chosen, reason = end, f"cdc_{name}_fallback"
            else:
                chosen, reason = hard_end, "cdc_hard_character_boundary"
        result.append((start, chosen, reason))
        start = chosen
    if start < len(text):
        result.append((start, len(text), "cdc_tail"))
    return result
