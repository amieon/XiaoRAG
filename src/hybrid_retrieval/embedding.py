"""稠密向量编码与精确 Top-K；真实模型通过可选 sentence-transformers 加载。"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    """生产嵌入器；依赖或模型缺失时明确失败，不使用随机向量冒充。"""

    def __init__(self, model_id: str, *, device: str = "cpu", batch_size: int = 16) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("embedding_model_id_required")
        if batch_size < 1:
            raise ValueError("embedding_batch_size_must_be_positive")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "dense retrieval requires sentence-transformers; install requirements-retrieval.txt"
            ) from exc
        self._model = SentenceTransformer(model_id, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return _validated_vectors(vectors)


def _validated_vectors(vectors) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("embedding_vectors_require_nonempty_2d_matrix")
    if not np.isfinite(result).all():
        raise ValueError("embedding_vectors_contain_nonfinite_values")
    return result


def search_dense(
    embeddings: np.ndarray, unit_ids: Sequence[str], query_vector: np.ndarray, *, top_k: int = 10,
) -> list[dict]:
    matrix = _validated_vectors(embeddings)
    if len(unit_ids) != matrix.shape[0] or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("dense_unit_ids_do_not_match_embeddings")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k_must_be_positive")
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1 or query.shape[0] != matrix.shape[1] or not np.isfinite(query).all():
        raise ValueError("query_embedding_dimension_mismatch")
    scores = matrix @ query
    ranked = sorted(range(len(unit_ids)), key=lambda index: (-float(scores[index]), unit_ids[index]))[:top_k]
    return [
        {"unit_id": unit_ids[index], "score": float(scores[index]), "rank": rank}
        for rank, index in enumerate(ranked, 1)
    ]
