#!/usr/bin/env python3
"""统一 dense 查询入口：--backend pgvector|milvus；两后端均不融合 BM25。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.embedding import SentenceTransformerEmbedder
from hybrid_retrieval.milvus_store import load_milvus_embedding_model, query_dense_milvus
from hybrid_retrieval.pgvector_store import _connect, load_stored_manifest, query_dense_pgvector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["pgvector", "milvus"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dsn-env", default="PGVECTOR_DSN")
    parser.add_argument("--uri", default=os.environ.get("MILVUS_URI", ""))
    parser.add_argument("--token-env", default="MILVUS_TOKEN")
    args = parser.parse_args()
    dsn = os.environ.get(args.dsn_env, "")
    token = os.environ.get(args.token_env, "")
    try:
        if args.backend == "pgvector":
            if not dsn:
                raise ValueError(f"{args.dsn_env}_required")
            with _connect(dsn) as connection:
                model_id = load_stored_manifest(connection, args.name)["embedding_model"]
        else:
            model_id = load_milvus_embedding_model(args.uri, args.name, token=token)
        embedder = SentenceTransformerEmbedder(
            model_id, device=args.device, batch_size=args.batch_size,
        )
        if args.backend == "pgvector":
            hits = query_dense_pgvector(dsn, args.name, args.query, embedder=embedder, top_k=args.top_k)
        else:
            hits = query_dense_milvus(
                args.uri, args.name, args.query, embedder=embedder, top_k=args.top_k, token=token,
            )
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    except Exception as exc:
        parser.error(f"{args.backend} query failed: {exc}")
        return
    print(json.dumps({
        "backend": args.backend, "name": args.name, "query": args.query,
        "top_k": args.top_k, "hits": hits,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
