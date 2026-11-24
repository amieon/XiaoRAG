"""统一知识单元的本地稠密／BM25 索引构建与查询。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bm25 import build_bm25, load_bm25, save_bm25, search_bm25
from .embedding import search_dense


MANIFEST_SCHEMA = "hybrid-retrieval-index.v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records
    ), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl_line:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_record_requires_object:{line_no}")
            records.append(value)
    return records


def build_index(
    units: Sequence[dict[str, Any]], parents: Sequence[dict[str, Any]], embedder,
    output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("index_output_not_empty_choose_new_directory")
    if not units:
        raise ValueError("retrieval_units_required")
    unit_ids = [unit.get("unit_id") for unit in units]
    if any(not isinstance(value, str) or not value for value in unit_ids) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("retrieval_unit_ids_must_be_unique")
    if any(unit.get("schema_version") != "retrieval-unit.v1" for unit in units):
        raise ValueError("index_requires_retrieval-unit.v1")
    texts = [unit.get("embedding_text") for unit in units]
    if any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("embedding_text_required")
    vectors = embedder.encode(texts)
    if vectors.shape[0] != len(units):
        raise ValueError("embedding_count_mismatch")
    bm25 = build_bm25(units)
    model_id = getattr(embedder, "model_id", None)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("embedder_model_id_required")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", vectors, allow_pickle=False)
    _write_json(output / "embedding-unit-ids.json", unit_ids)
    _write_jsonl(output / "units.jsonl", units)
    _write_jsonl(output / "parents.jsonl", parents)
    save_bm25(output / "bm25-index.json", bm25)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "embedding_model": model_id,
        "embedding_dimension": int(vectors.shape[1]),
        "vector_dtype": "float32",
        "vector_similarity": "cosine_normalized_embeddings",
        "unit_count": len(units), "parent_count": len(parents),
        "sparse_algorithm": "bm25", "tokenizer": bm25["tokenizer"],
        "index_complete": True,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def load_manifest(index_dir: str | Path) -> dict[str, Any]:
    manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("index_complete") is not True:
        raise ValueError("unsupported_or_incomplete_retrieval_index")
    return manifest


def query_index(
    index_dir: str | Path, query: str, *, mode: str, top_k: int = 10, embedder=None,
) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query_must_be_nonempty")
    index_dir = Path(index_dir)
    load_manifest(index_dir)
    units = read_jsonl(index_dir / "units.jsonl")
    by_id = {unit["unit_id"]: unit for unit in units}
    if mode == "bm25":
        hits = search_bm25(load_bm25(index_dir / "bm25-index.json"), query, top_k=top_k)
    elif mode == "dense":
        if embedder is None:
            raise ValueError("dense_query_requires_embedder")
        embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
        unit_ids = json.loads((index_dir / "embedding-unit-ids.json").read_text(encoding="utf-8"))
        query_vector = embedder.encode([query.strip()])
        hits = search_dense(embeddings, unit_ids, query_vector, top_k=top_k)
    else:
        raise ValueError("query_mode_must_be_dense_or_bm25")
    result = []
    for hit in hits:
        unit = by_id.get(hit["unit_id"])
        if unit is None:
            raise ValueError("index_hit_references_missing_unit")
        result.append({
            "mode": mode, **hit, "document_id": unit["document_id"],
            "parent_id": unit["parent_id"], "text": unit["embedding_text"],
            "body_text": unit["body_text"], "title_path": unit["title_path"],
            "clause_path": unit["clause_path"], "source_refs": unit["source_refs"],
            "page_numbers": unit["page_numbers"], "security": unit["security"],
        })
    return result
