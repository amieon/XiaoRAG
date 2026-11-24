#!/usr/bin/env python3
"""统一向量入库入口：--backend pgvector|milvus；复用已有 dense 索引。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.milvus_store import store_dense_milvus
from hybrid_retrieval.pgvector_store import store_dense_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=["pgvector", "milvus"])
    parser.add_argument("--name", required=True, help="pgvector schema 或 Milvus collection；必须不存在")
    parser.add_argument("--dsn-env", default="PGVECTOR_DSN")
    parser.add_argument("--uri", default=os.environ.get("MILVUS_URI", ""), help="Milvus URI 或 Milvus Lite 文件路径")
    parser.add_argument("--token-env", default="MILVUS_TOKEN", help="Milvus token 的环境变量名")
    args = parser.parse_args()
    try:
        if args.backend == "pgvector":
            result = store_dense_index(
                args.index, dsn=os.environ.get(args.dsn_env, ""), schema=args.name,
            )
        else:
            result = store_dense_milvus(
                args.index, uri=args.uri, collection=args.name,
                token=os.environ.get(args.token_env, ""),
            )
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    except Exception as exc:
        parser.error(f"{args.backend} import failed: {exc}")
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
