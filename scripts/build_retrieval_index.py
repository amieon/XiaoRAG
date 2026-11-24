#!/usr/bin/env python3
"""用统一检索单元同时构建稠密向量与 BM25 索引。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.embedding import SentenceTransformerEmbedder
from hybrid_retrieval.index import build_index, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", required=True, type=Path, help="retrieval-units.jsonl")
    parser.add_argument("--parents", type=Path, help="默认同目录 retrieval-parents.jsonl")
    parser.add_argument("--model", required=True, help="本地嵌入模型路径或 sentence-transformers 模型 ID")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path, help="新的空索引目录")
    args = parser.parse_args()
    if not args.units.is_file():
        parser.error("retrieval-units.jsonl 不存在")
    parent_path = args.parents or args.units.with_name("retrieval-parents.jsonl")
    if not parent_path.is_file():
        parser.error("retrieval-parents.jsonl 不存在")
    try:
        units = read_jsonl(args.units)
        parents = read_jsonl(parent_path)
        embedder = SentenceTransformerEmbedder(
            args.model, device=args.device, batch_size=args.batch_size,
        )
        manifest = build_index(units, parents, embedder, args.output)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
