"""确定性中英文分词：中文单字＋相邻二元词，英文数字保留完整 token。"""
from __future__ import annotations

import re


TOKENIZER_VERSION = "unicode-cjk-bigram-v1"
RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("tokenizer_requires_string")
    tokens: list[str] = []
    for match in RUN_RE.finditer(text.casefold()):
        value = match.group(0)
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", value):
            # 单字保证长词仍可召回，二元词提高短语和术语的区分度。
            tokens.extend(value)
            tokens.extend(value[index:index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens
