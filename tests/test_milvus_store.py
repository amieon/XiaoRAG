from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.index import build_index
from hybrid_retrieval.milvus_store import query_dense_milvus, store_dense_milvus


class TinyEmbedder:
    model_id = "tiny-milvus-test-model"

    def encode(self, texts):
        values = {"增量更新": [1.0, 0.0], "版本快照": [0.0, 1.0], "增量": [1.0, 0.0]}
        return np.asarray([values[text] for text in texts], dtype=np.float32)


class FakeDataType:
    VARCHAR, FLOAT_VECTOR, JSON = "VARCHAR", "FLOAT_VECTOR", "JSON"


class FakeSchema:
    def __init__(self, **options):
        self.options, self.fields = options, []

    def add_field(self, **field):
        self.fields.append(field)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **index):
        self.indexes.append(index)


class FakeClient:
    def __init__(self, uri, token=""):
        self.uri, self.token = uri, token
        self.collections, self.rows, self.closed = set(), [], False
        self.created_schema, self.index_params = None, None

    def create_schema(self, **options):
        return FakeSchema(**options)

    def prepare_index_params(self):
        self.index_params = FakeIndexParams()
        return self.index_params

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, schema, index_params, consistency_level):
        self.collections.add(collection_name)
        self.created_schema = schema
        assert consistency_level == "Strong"

    def insert(self, collection_name, data):
        self.rows = data

    def flush(self, collection_name):
        pass

    def load_collection(self, collection_name):
        pass

    def drop_collection(self, collection_name):
        self.collections.remove(collection_name)

    def close(self):
        self.closed = True

    def query(self, collection_name, filter, output_fields, limit):
        return [{"embedding_model": self.rows[0]["embedding_model"]}] if self.rows else []

    def search(self, collection_name, data, anns_field, search_params, limit, output_fields):
        row = self.rows[0]
        entity = {name: row[name] for name in output_fields}
        return [[{"id": row["unit_id"], "distance": 0.98, "entity": entity}]]


def unit(uid, text):
    return {
        "schema_version": "retrieval-unit.v1", "unit_id": uid, "document_id": "doc1",
        "parent_id": "chapter_tree:doc1:p1", "embedding_text": text, "body_text": text,
        "title_path": ["第一章"], "clause_path": [],
        "source_refs": [{"block_id": "b1", "page_number": 1}], "page_numbers": [1],
        "security": {"classification": "internal", "allowed_principals": ["group:dev"]},
    }


def make_index(tmp_path):
    output = tmp_path / "index"
    build_index([
        unit("chapter_tree:doc1:c1", "增量更新"),
        unit("chapter_tree:doc1:c2", "版本快照"),
    ], [], TinyEmbedder(), output)
    return output


def test_store_builds_flat_cosine_collection_and_reuses_vectors(tmp_path):
    client = FakeClient("fake")
    with patch("hybrid_retrieval.milvus_store._client", return_value=client), \
            patch("hybrid_retrieval.milvus_store._data_type", return_value=FakeDataType):
        result = store_dense_milvus(
            make_index(tmp_path), uri="fake", collection="retrieval_exp01", token="secret",
        )
    assert result["unit_count"] == 2 and result["index_type"] == "FLAT"
    assert client.index_params.indexes[0]["metric_type"] == "COSINE"
    assert client.rows[0]["embedding"] == [1.0, 0.0]
    assert client.rows[0]["embedding_model"] == TinyEmbedder.model_id
    assert client.closed
    assert "secret" not in str(result) and "fake" not in str(result)


def test_query_restores_entity_and_cosine_score(tmp_path):
    client = FakeClient("fake")
    with patch("hybrid_retrieval.milvus_store._client", return_value=client), \
            patch("hybrid_retrieval.milvus_store._data_type", return_value=FakeDataType):
        store_dense_milvus(make_index(tmp_path), uri="fake", collection="retrieval_exp01")
        client.closed = False
        hits = query_dense_milvus(
            "fake", "retrieval_exp01", "增量", embedder=TinyEmbedder(), top_k=1,
        )
    assert hits[0]["unit_id"] == "chapter_tree:doc1:c1"
    assert hits[0]["score"] == 0.98 and hits[0]["rank"] == 1
    assert hits[0]["security"]["classification"] == "internal"
    assert client.closed


def test_existing_collection_is_not_overwritten(tmp_path):
    client = FakeClient("fake")
    client.collections.add("retrieval_exp01")
    with patch("hybrid_retrieval.milvus_store._client", return_value=client):
        with pytest.raises(ValueError, match="already_exists"):
            store_dense_milvus(make_index(tmp_path), uri="fake", collection="retrieval_exp01")
    assert client.collections == {"retrieval_exp01"}
    assert client.closed
