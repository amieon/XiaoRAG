#!/usr/bin/env python3
"""发布单文档知识版本：路径定身份、完整快照、按内容哈希复用向量。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.embedding import SentenceTransformerEmbedder
from versioning import publish_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="受管原始文件；其规范化绝对路径决定文档身份")
    parser.add_argument("--input", required=True, type=Path, help="单文档 retrieval export 目录")
    parser.add_argument("--store", required=True, type=Path, help="版本库目录；向量池由全部文档共享")
    parser.add_argument("--model", required=True, help="本地嵌入模型路径或 sentence-transformers 模型 ID")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error("source 文件不存在")
    if not args.input.is_dir():
        parser.error("input 导出目录不存在")
    try:
        embedder = SentenceTransformerEmbedder(
            args.model, device=args.device, batch_size=args.batch_size,
        )
        manifest = publish_version(
            args.source, args.input, args.store, embedder=embedder,
        )
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError, KeyError) as exc:
        parser.error(str(exc))
        return
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
