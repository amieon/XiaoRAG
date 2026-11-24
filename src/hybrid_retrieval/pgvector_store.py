"""可选 pgvector dense 后端；JSONL 与 BM25 仍是独立产物，不整体搬入向量库。"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .index import load_manifest, read_jsonl


SCHEMA_RE = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")


def validate_schema_name(value: str) -> str:
    """Schema 同时是隔离边界和 SQL 标识符，禁止动态拼入任意输入。"""
    if not isinstance(value, str) or not SCHEMA_RE.fullmatch(value):
        raise ValueError("pgvector_schema_requires_lowercase_sql_identifier")
    return value


def _connect(dsn: str):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("pgvector backend requires psycopg; install requirements-pgvector.txt") from exc
    return psycopg.connect(dsn)


def _vector_literal(vector) -> str:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("pgvector_requires_finite_1d_embedding")
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def store_dense_index(
    index_dir: str | Path, *, dsn: str, schema: str,
) -> dict[str, Any]:
    """把已完成的本地 dense 索引复制到一个全新的 PostgreSQL schema。"""
    schema = validate_schema_name(schema)
    if not isinstance(dsn, str) or not dsn.strip():
        raise ValueError("PGVECTOR_DSN_required")
    index_dir = Path(index_dir)
    manifest = load_manifest(index_dir)
    units = read_jsonl(index_dir / "units.jsonl")
    unit_ids = json.loads((index_dir / "embedding-unit-ids.json").read_text(encoding="utf-8-sig"))
    embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(units) or unit_ids != [unit.get("unit_id") for unit in units]:
        raise ValueError("dense_index_vector_unit_mapping_mismatch")
    dimension = int(manifest["embedding_dimension"])
    if embeddings.shape[1] != dimension:
        raise ValueError("dense_index_dimension_mismatch")

    rows = [(
        unit["unit_id"], unit["document_id"], unit["parent_id"], _vector_literal(embeddings[index]),
        unit["embedding_text"], unit["body_text"],
        json.dumps(unit["title_path"], ensure_ascii=False),
        json.dumps(unit["clause_path"], ensure_ascii=False),
        json.dumps(unit["source_refs"], ensure_ascii=False),
        json.dumps(unit["page_numbers"], ensure_ascii=False),
        json.dumps(unit["security"], ensure_ascii=False),
    ) for index, unit in enumerate(units)]
    stored_manifest = {**manifest, "storage_backend": "pgvector", "schema": schema}
    with _connect(dsn) as connection:
        with connection.cursor() as cursor:
            # 不覆盖已有实验；建 schema、建表和写入在同一事务内提交或回滚。
            cursor.execute(f'CREATE SCHEMA "{schema}"')
            cursor.execute(f'''
                CREATE TABLE "{schema}".manifest (
                    key text PRIMARY KEY,
                    value jsonb NOT NULL
                )
            ''')
            cursor.execute(f'''
                CREATE TABLE "{schema}".units (
                    unit_id text PRIMARY KEY,
                    document_id text NOT NULL,
                    parent_id text NOT NULL,
                    embedding vector({dimension}) NOT NULL,
                    embedding_text text NOT NULL,
                    body_text text NOT NULL,
                    title_path jsonb NOT NULL,
                    clause_path jsonb NOT NULL,
                    source_refs jsonb NOT NULL,
                    page_numbers jsonb NOT NULL,
                    security jsonb NOT NULL
                )
            ''')
            cursor.execute(
                f'INSERT INTO "{schema}".manifest (key, value) VALUES (%s, %s::jsonb)',
                ("index", json.dumps(stored_manifest, ensure_ascii=False)),
            )
            cursor.executemany(f'''
                INSERT INTO "{schema}".units (
                    unit_id, document_id, parent_id, embedding, embedding_text, body_text,
                    title_path, clause_path, source_refs, page_numbers, security
                ) VALUES (
                    %s, %s, %s, %s::vector, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
                )
            ''', rows)
    # 返回值绝不带 DSN，避免凭证进入日志或清单。
    return {
        "status": "success", "storage_backend": "pgvector", "schema": schema,
        "unit_count": len(rows), "embedding_dimension": dimension,
        "embedding_model": manifest["embedding_model"], "exact_cosine_search": True,
    }


def load_stored_manifest(connection, schema: str) -> dict[str, Any]:
    schema = validate_schema_name(schema)
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT value FROM "{schema}".manifest WHERE key = %s', ("index",))
        row = cursor.fetchone()
    if row is None:
        raise ValueError("pgvector_index_manifest_missing")
    manifest = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    if manifest.get("storage_backend") != "pgvector" or manifest.get("schema") != schema:
        raise ValueError("pgvector_manifest_backend_mismatch")
    return manifest


def query_dense_pgvector(
    dsn: str, schema: str, query: str, *, embedder, top_k: int = 10,
) -> list[dict[str, Any]]:
    """精确余弦 Top-K；当前不过滤 security，也不融合 BM25。"""
    schema = validate_schema_name(schema)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k_must_be_positive")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query_must_be_nonempty")
    with _connect(dsn) as connection:
        manifest = load_stored_manifest(connection, schema)
        model_id = getattr(embedder, "model_id", None)
        if model_id != manifest.get("embedding_model"):
            raise ValueError("pgvector_query_embedding_model_mismatch")
        query_vector = embedder.encode([query.strip()])
        if query_vector.shape != (1, int(manifest["embedding_dimension"])):
            raise ValueError("pgvector_query_embedding_dimension_mismatch")
        with connection.cursor() as cursor:
            cursor.execute(f'''
                SELECT unit_id, document_id, parent_id, embedding_text, body_text,
                       title_path, clause_path, source_refs, page_numbers, security,
                       embedding <=> %s::vector AS cosine_distance
                FROM "{schema}".units
                ORDER BY embedding <=> %s::vector ASC, unit_id ASC
                LIMIT %s
            ''', (_vector_literal(query_vector[0]), _vector_literal(query_vector[0]), top_k))
            rows = cursor.fetchall()
    return [{
        "mode": "dense_pgvector", "unit_id": row[0], "document_id": row[1], "parent_id": row[2],
        "text": row[3], "body_text": row[4], "title_path": row[5], "clause_path": row[6],
        "source_refs": row[7], "page_numbers": row[8], "security": row[9],
        "score": 1.0 - float(row[10]), "cosine_distance": float(row[10]), "rank": rank,
    } for rank, row in enumerate(rows, 1)]
