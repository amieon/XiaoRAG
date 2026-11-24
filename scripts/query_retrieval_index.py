#!/usr/bin/env python3
"""查询本地稠密向量或 BM25 索引；两路独立执行，不做融合。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.embedding import SentenceTransformerEmbedder
from hybrid_retrieval.index import load_manifest, query_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path, help="build_retrieval_index.py 生成的索引目录")
    parser.add_argument("--query", required=True, help="非空查询文本")
    parser.add_argument("--mode", required=True, choices=["dense", "bm25"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu", help="仅 dense 模式使用")
    parser.add_argument("--batch-size", type=int, default=16, help="仅 dense 模式使用")
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.index)
        embedder = None
        if args.mode == "dense":
            # 查询必须使用建库清单中的同一模型，不能用另一个模型混算向量。
            embedder = SentenceTransformerEmbedder(
                manifest["embedding_model"], device=args.device, batch_size=args.batch_size,
            )
        hits = query_index(
            args.index, args.query, mode=args.mode, top_k=args.top_k, embedder=embedder,
        )
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    print(json.dumps({
        "mode": args.mode, "query": args.query, "top_k": args.top_k, "hits": hits,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
