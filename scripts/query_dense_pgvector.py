#!/usr/bin/env python3
"""使用 pgvector 做独立 dense 精确余弦 Top-K，不执行 BM25 或权限过滤。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.embedding import SentenceTransformerEmbedder
from hybrid_retrieval.pgvector_store import _connect, load_stored_manifest, query_dense_pgvector, validate_schema_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dsn-env", default="PGVECTOR_DSN", help="保存 DSN 的环境变量名")
    args = parser.parse_args()
    dsn = os.environ.get(args.dsn_env, "")
    if not dsn:
        parser.error(f"{args.dsn_env}_required")
    try:
        schema = validate_schema_name(args.schema)
        with _connect(dsn) as connection:
            manifest = load_stored_manifest(connection, schema)
        embedder = SentenceTransformerEmbedder(
            manifest["embedding_model"], device=args.device, batch_size=args.batch_size,
        )
        hits = query_dense_pgvector(dsn, schema, args.query, embedder=embedder, top_k=args.top_k)
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    except Exception as exc:
        parser.error(f"pgvector query failed: {exc}")
        return
    print(json.dumps({
        "mode": "dense_pgvector", "schema": schema, "query": args.query,
        "top_k": args.top_k, "hits": hits,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
