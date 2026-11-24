from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.index import build_index
from hybrid_retrieval.pgvector_store import (
    query_dense_pgvector, store_dense_index, validate_schema_name,
)


class TinyEmbedder:
    model_id = "tiny-pgvector-test-model"

    def encode(self, texts):
        values = {
            "增量更新": [1.0, 0.0], "版本快照": [0.0, 1.0], "增量": [1.0, 0.0],
        }
        return np.asarray([values[text] for text in texts], dtype=np.float32)


def unit(uid, text):
    return {
        "schema_version": "retrieval-unit.v1", "unit_id": uid, "document_id": "doc1",
        "parent_id": "chapter_tree:doc1:p1", "embedding_text": text, "body_text": text,
        "title_path": ["第一章"], "clause_path": [],
        "source_refs": [{"block_id": "b1", "page_number": 1}], "page_numbers": [1],
        "security": {"classification": "internal", "allowed_principals": ["group:dev"]},
    }


class FakeCursor:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.owner.calls.append((sql, params))
        if sql.lstrip().startswith("SELECT value"):
            self.owner.one = (self.owner.manifest,)
        elif sql.lstrip().startswith("SELECT unit_id"):
            self.owner.rows = self.owner.result_rows

    def executemany(self, sql, rows):
        self.owner.batch_sql = sql
        self.owner.batches.append(list(rows))

    def fetchone(self):
        return self.owner.one

    def fetchall(self):
        return self.owner.rows


class FakeConnection:
    def __init__(self, manifest=None, result_rows=None):
        self.calls, self.batches = [], []
        self.manifest, self.result_rows = manifest, result_rows or []
        self.one, self.rows = None, []
        self.batch_sql = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return FakeCursor(self)


def make_index(tmp_path):
    output = tmp_path / "index"
    build_index([
        unit("chapter_tree:doc1:c1", "增量更新"),
        unit("chapter_tree:doc1:c2", "版本快照"),
    ], [], TinyEmbedder(), output)
    return output


def test_store_copies_existing_dense_without_model_or_dsn_leak(tmp_path):
    connection = FakeConnection()
    with patch("hybrid_retrieval.pgvector_store._connect", return_value=connection) as connect:
        result = store_dense_index(make_index(tmp_path), dsn="postgresql://secret", schema="retrieval_exp01")
    connect.assert_called_once_with("postgresql://secret")
    assert result["unit_count"] == 2
    assert result["exact_cosine_search"] is True
    assert "secret" not in json.dumps(result)
    assert any(call[0].startswith('CREATE SCHEMA "retrieval_exp01"') for call in connection.calls)
    assert len(connection.batches) == 1 and len(connection.batches[0]) == 2
    assert "embedding vector(2)" in connection.calls[2][0]


def test_query_uses_cosine_distance_and_restores_unit_fields():
    manifest = {
        "storage_backend": "pgvector", "schema": "retrieval_exp01",
        "embedding_model": TinyEmbedder.model_id, "embedding_dimension": 2,
    }
    source = unit("chapter_tree:doc1:c1", "增量更新")
    row = (
        source["unit_id"], source["document_id"], source["parent_id"], source["embedding_text"],
        source["body_text"], source["title_path"], source["clause_path"], source["source_refs"],
        source["page_numbers"], source["security"], 0.0,
    )
    connection = FakeConnection(manifest, [row])
    with patch("hybrid_retrieval.pgvector_store._connect", return_value=connection):
        hits = query_dense_pgvector("dsn", "retrieval_exp01", "增量", embedder=TinyEmbedder(), top_k=1)
    assert hits[0]["unit_id"] == source["unit_id"]
    assert hits[0]["score"] == 1.0 and hits[0]["cosine_distance"] == 0.0
    assert hits[0]["parent_id"] == source["parent_id"]
    query_sql = [call for call in connection.calls if call[0].lstrip().startswith("SELECT unit_id")][0]
    assert "ORDER BY embedding <=> %s::vector ASC, unit_id ASC" in query_sql[0]
    assert query_sql[1][-1] == 1


def test_schema_identifier_and_model_mismatch_fail_closed():
    with pytest.raises(ValueError, match="schema"):
        validate_schema_name('bad"; DROP TABLE public.units; --')
    connection = FakeConnection({
        "storage_backend": "pgvector", "schema": "retrieval_exp01",
        "embedding_model": "another-model", "embedding_dimension": 2,
    })
    with patch("hybrid_retrieval.pgvector_store._connect", return_value=connection):
        with pytest.raises(ValueError, match="model_mismatch"):
            query_dense_pgvector("dsn", "retrieval_exp01", "增量", embedder=TinyEmbedder())
