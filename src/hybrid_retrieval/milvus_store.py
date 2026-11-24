"""可选 Milvus dense 后端；复用本地向量，不搬入 parent、BM25 或全部 JSON。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .index import load_manifest, read_jsonl
from .pgvector_store import validate_schema_name


OUTPUT_FIELDS = [
    "unit_id", "document_id", "parent_id", "embedding_text", "body_text",
    "title_path", "clause_path", "source_refs", "page_numbers", "security",
]


def _client(uri: str, token: str = ""):
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("Milvus backend requires pymilvus; install requirements-milvus.txt") from exc
    kwargs = {"uri": uri}
    if token:
        kwargs["token"] = token
    return MilvusClient(**kwargs)


def _data_type():
    from pymilvus import DataType
    return DataType


def _make_schema(client, dimension: int):
    data_type = _data_type()
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="unit_id", datatype=data_type.VARCHAR, is_primary=True, max_length=1024)
    schema.add_field(field_name="document_id", datatype=data_type.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_id", datatype=data_type.VARCHAR, max_length=1024)
    schema.add_field(field_name="embedding", datatype=data_type.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="embedding_model", datatype=data_type.VARCHAR, max_length=1024)
    schema.add_field(field_name="embedding_text", datatype=data_type.VARCHAR, max_length=65535)
    schema.add_field(field_name="body_text", datatype=data_type.VARCHAR, max_length=65535)
    for name in ("title_path", "clause_path", "source_refs", "page_numbers", "security"):
        schema.add_field(field_name=name, datatype=data_type.JSON)
    return schema


def _read_dense(index_dir: str | Path):
    index_dir = Path(index_dir)
    manifest = load_manifest(index_dir)
    units = read_jsonl(index_dir / "units.jsonl")
    unit_ids = json.loads((index_dir / "embedding-unit-ids.json").read_text(encoding="utf-8-sig"))
    embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(units) or unit_ids != [unit.get("unit_id") for unit in units]:
        raise ValueError("dense_index_vector_unit_mapping_mismatch")
    dimension = int(manifest["embedding_dimension"])
    if embeddings.shape[1] != dimension or not np.isfinite(embeddings).all():
        raise ValueError("dense_index_dimension_or_value_mismatch")
    return manifest, units, embeddings


def store_dense_milvus(
    index_dir: str | Path, *, uri: str, collection: str, token: str = "",
) -> dict[str, Any]:
    collection = validate_schema_name(collection)
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("milvus_uri_required")
    manifest, units, embeddings = _read_dense(index_dir)
    dimension = int(manifest["embedding_dimension"])
    # 本地 Milvus Lite URI 是普通文件路径；服务端 URI 不在本地建目录。
    if "://" not in uri:
        Path(uri).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    client = _client(uri, token)
    created = False
    try:
        if client.has_collection(collection_name=collection):
            raise ValueError("milvus_collection_already_exists_choose_new_name")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding", index_name="embedding_flat_cosine",
            index_type="FLAT", metric_type="COSINE",
        )
        client.create_collection(
            collection_name=collection, schema=_make_schema(client, dimension),
            index_params=index_params, consistency_level="Strong",
        )
        created = True
        rows = []
        for index, unit in enumerate(units):
            rows.append({
                "unit_id": unit["unit_id"], "document_id": unit["document_id"],
                "parent_id": unit["parent_id"], "embedding": embeddings[index].astype(float).tolist(),
                "embedding_model": manifest["embedding_model"],
                "embedding_text": unit["embedding_text"], "body_text": unit["body_text"],
                "title_path": unit["title_path"], "clause_path": unit["clause_path"],
                "source_refs": unit["source_refs"], "page_numbers": unit["page_numbers"],
                "security": unit["security"],
            })
        client.insert(collection_name=collection, data=rows)
        client.flush(collection_name=collection)
        client.load_collection(collection_name=collection)
    except Exception:
        # 只清理由本次调用新建的半成品，绝不覆盖或删除已有 collection。
        if created:
            client.drop_collection(collection_name=collection)
        raise
    finally:
        client.close()
    return {
        "status": "success", "storage_backend": "milvus", "collection": collection,
        "unit_count": len(units), "embedding_dimension": dimension,
        "embedding_model": manifest["embedding_model"], "index_type": "FLAT",
        "metric_type": "COSINE", "exact_cosine_search": True,
    }


def load_milvus_embedding_model(uri: str, collection: str, *, token: str = "") -> str:
    """读取建库时随每条记录保存的模型 ID，不加载向量或正文。"""
    collection = validate_schema_name(collection)
    client = _client(uri, token)
    try:
        if not client.has_collection(collection_name=collection):
            raise ValueError("milvus_collection_missing")
        rows = client.query(collection_name=collection, filter="", output_fields=["embedding_model"], limit=1)
    finally:
        client.close()
    if not rows or not isinstance(rows[0].get("embedding_model"), str):
        raise ValueError("milvus_embedding_model_metadata_missing")
    return rows[0]["embedding_model"]


def query_dense_milvus(
    uri: str, collection: str, query: str, *, embedder, top_k: int = 10, token: str = "",
) -> list[dict[str, Any]]:
    collection = validate_schema_name(collection)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k_must_be_positive")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query_must_be_nonempty")
    client = _client(uri, token)
    try:
        if not client.has_collection(collection_name=collection):
            raise ValueError("milvus_collection_missing")
        model_rows = client.query(
            collection_name=collection, filter="", output_fields=["embedding_model"], limit=1,
        )
        if not model_rows:
            raise ValueError("milvus_collection_is_empty")
        model_id = model_rows[0].get("embedding_model")
        if model_id != getattr(embedder, "model_id", None):
            raise ValueError("milvus_query_embedding_model_mismatch")
        query_vector = embedder.encode([query.strip()])
        if query_vector.ndim != 2 or query_vector.shape[0] != 1 or not np.isfinite(query_vector).all():
            raise ValueError("milvus_query_embedding_invalid")
        result = client.search(
            collection_name=collection, data=[query_vector[0].astype(float).tolist()],
            anns_field="embedding", search_params={"metric_type": "COSINE", "params": {}},
            limit=top_k, output_fields=OUTPUT_FIELDS,
        )
        hits = result[0] if result else []
    finally:
        client.close()
    output = []
    for rank, hit in enumerate(hits, 1):
        entity = hit.get("entity", {})
        output.append({
            "mode": "dense_milvus", "unit_id": entity.get("unit_id", hit.get("id")),
            "document_id": entity["document_id"], "parent_id": entity["parent_id"],
            "text": entity["embedding_text"], "body_text": entity["body_text"],
            "title_path": entity["title_path"], "clause_path": entity["clause_path"],
            "source_refs": entity["source_refs"], "page_numbers": entity["page_numbers"],
            "security": entity["security"], "score": float(hit["distance"]), "rank": rank,
        })
    return output
