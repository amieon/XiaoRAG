"""小规模可复现 BM25；索引保存显式 postings，不依赖外部搜索引擎。"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .tokenizer import TOKENIZER_VERSION, tokenize


SCHEMA_VERSION = "bm25-index.v1"


def build_bm25(units: Iterable[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> dict[str, Any]:
    if not 0 < k1 or not 0 <= b <= 1:
        raise ValueError("invalid_bm25_parameters")
    unit_ids, lengths = [], []
    postings: dict[str, list[list[int]]] = defaultdict(list)
    for index, unit in enumerate(units):
        unit_id = unit.get("unit_id")
        text = unit.get("embedding_text")
        if not isinstance(unit_id, str) or not unit_id or not isinstance(text, str):
            raise ValueError("bm25_requires_unit_id_and_embedding_text")
        counts = Counter(tokenize(text))
        lengths.append(sum(counts.values()))
        unit_ids.append(unit_id)
        for token, frequency in counts.items():
            postings[token].append([index, frequency])
    if not unit_ids:
        raise ValueError("bm25_requires_nonempty_corpus")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("bm25_duplicate_unit_id")
    return {
        "schema_version": SCHEMA_VERSION,
        "tokenizer": TOKENIZER_VERSION,
        "parameters": {"k1": k1, "b": b},
        "unit_ids": unit_ids,
        "document_lengths": lengths,
        "average_document_length": sum(lengths) / len(lengths),
        "document_count": len(unit_ids),
        "postings": dict(postings),
    }


def search_bm25(index: dict[str, Any], query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k_must_be_positive")
    if index.get("schema_version") != SCHEMA_VERSION or index.get("tokenizer") != TOKENIZER_VERSION:
        raise ValueError("bm25_index_schema_or_tokenizer_mismatch")
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    k1 = index["parameters"]["k1"]
    b = index["parameters"]["b"]
    count = index["document_count"]
    lengths = index["document_lengths"]
    average = index["average_document_length"]
    scores: dict[int, float] = defaultdict(float)
    matched: dict[int, set[str]] = defaultdict(set)
    for token in set(query_tokens):
        entries = index["postings"].get(token, [])
        if not entries:
            continue
        document_frequency = len(entries)
        inverse_document_frequency = math.log(1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
        for document_index, frequency in entries:
            length = lengths[document_index]
            denominator = frequency + k1 * (1.0 - b + b * length / average) if average else frequency + k1
            scores[document_index] += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
            matched[document_index].add(token)
    ranked = sorted(scores, key=lambda index_: (-scores[index_], index["unit_ids"][index_]))[:top_k]
    return [
        {"unit_id": index["unit_ids"][index_], "score": scores[index_], "rank": rank,
         "matched_tokens": sorted(matched[index_])}
        for rank, index_ in enumerate(ranked, 1)
    ]


def save_bm25(path: str | Path, index: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(index, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_bm25(path: str | Path) -> dict[str, Any]:
    index = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported_bm25_index")
    return index
