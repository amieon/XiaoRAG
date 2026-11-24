from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.bm25 import build_bm25, load_bm25, save_bm25, search_bm25
from hybrid_retrieval.embedding import search_dense
from hybrid_retrieval.index import build_index, load_manifest, query_index
from hybrid_retrieval.tokenizer import tokenize


class TinyEmbedder:
    """只用于验证矩阵与索引数据流，不代表真实模型或检索精度。"""

    model_id = "tiny-explicit-test-vectors"

    def encode(self, texts):
        # 正交单位向量便于精确断言；顺序与输入 texts 一一对应。
        vectors = {
            "第一章\n\n系统支持增量更新": [1.0, 0.0],
            "第二章\n\n系统支持版本管理": [0.0, 1.0],
            "增量更新": [1.0, 0.0],
            "版本管理": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def unit(unit_id, text):
    return {
        "schema_version": "retrieval-unit.v1",
        "unit_id": unit_id,
        "document_id": "doc1",
        "parent_id": "chapter_tree:doc1:p1",
        "embedding_text": text,
        "body_text": text.split("\n\n")[-1],
        "title_path": [text.split("\n")[0]],
        "clause_path": [],
        "source_refs": [{"block_id": "b1", "page_number": 1}],
        "page_numbers": [1],
        "security": {"classification": "internal", "allowed_principals": ["group:dev"]},
    }


def units():
    return [
        unit("chapter_tree:doc1:c1", "第一章\n\n系统支持增量更新"),
        unit("chapter_tree:doc1:c2", "第二章\n\n系统支持版本管理"),
    ]


def test_tokenizer_preserves_terms_numbers_versions_and_cjk_bigrams():
    tokens = tokenize("增量更新 API v2.1 HTTP-2 Alpha_beta 3.14")
    assert tokens[:6] == ["增", "量", "更", "新", "增量", "量更"]
    assert "更新" in tokens
    assert "api" in tokens
    assert "v2.1" in tokens
    assert "http-2" in tokens
    assert "alpha_beta" in tokens
    assert "3.14" in tokens


def test_bm25_exact_term_hit_scores_and_serialization(tmp_path):
    index = build_bm25([
        {"unit_id": "b", "embedding_text": "普通说明与版本管理"},
        {"unit_id": "a", "embedding_text": "增量更新与增量同步"},
    ])
    hits = search_bm25(index, "增量更新", top_k=2)
    assert hits[0]["unit_id"] == "a"
    assert hits[0]["rank"] == 1
    assert hits[0]["score"] > 0 and math.isfinite(hits[0]["score"])
    assert "增量" in hits[0]["matched_tokens"] and "更新" in hits[0]["matched_tokens"]
    assert search_bm25(index, "不存在的术语") == []
    path = tmp_path / "bm25.json"
    save_bm25(path, index)
    assert search_bm25(load_bm25(path), "增量更新", top_k=1) == search_bm25(index, "增量更新", top_k=1)


def test_dense_top_k_dimension_validation_and_deterministic_ties():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    ids = ["b", "c", "a"]
    hits = search_dense(embeddings, ids, np.asarray([1.0, 0.0]), top_k=3)
    assert [hit["unit_id"] for hit in hits] == ["a", "b", "c"]
    assert [hit["rank"] for hit in hits] == [1, 2, 3]
    with pytest.raises(ValueError, match="dimension"):
        search_dense(embeddings, ids, np.asarray([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="top_k"):
        search_dense(embeddings, ids, np.asarray([1.0, 0.0]), top_k=0)


def test_build_and_query_both_independent_modes(tmp_path):
    output = tmp_path / "index"
    manifest = build_index(units(), [], TinyEmbedder(), output)
    assert manifest["embedding_dimension"] == 2
    assert manifest["unit_count"] == 2
    assert np.load(output / "embeddings.npy", allow_pickle=False).dtype == np.float32
    assert load_manifest(output)["embedding_model"] == TinyEmbedder.model_id

    dense = query_index(output, "增量更新", mode="dense", top_k=1, embedder=TinyEmbedder())
    sparse = query_index(output, "增量更新", mode="bm25", top_k=1)
    assert dense[0]["unit_id"] == units()[0]["unit_id"]
    assert sparse[0]["unit_id"] == units()[0]["unit_id"]
    assert dense[0]["mode"] == "dense" and sparse[0]["mode"] == "bm25"
    assert dense[0]["parent_id"] == units()[0]["parent_id"]
    assert dense[0]["security"] == units()[0]["security"]
    with pytest.raises(ValueError, match="requires_embedder"):
        query_index(output, "增量更新", mode="dense", top_k=1)


def test_query_bm25_cli_does_not_load_embedding_dependency(tmp_path):
    output = tmp_path / "index"
    build_index(units(), [], TinyEmbedder(), output)
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "query_retrieval_index.py"),
        "--index", str(output), "--query", "增量更新", "--mode", "bm25", "--top-k", "1",
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result["mode"] == "bm25"
    assert result["hits"][0]["unit_id"] == units()[0]["unit_id"]
    assert result["hits"][0]["text"] == units()[0]["embedding_text"]
