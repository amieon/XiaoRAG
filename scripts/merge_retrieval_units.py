#!/usr/bin/env python3
"""合并多个单文档 units/parents 导出目录，生成可直接建库的统一语料。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.merge import merge_retrieval_exports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", required=True, nargs="+", type=Path,
        help="一个或多个 export_retrieval_units.py 输出目录，顺序即合并顺序",
    )
    parser.add_argument("--output", required=True, type=Path, help="新的空合并目录")
    args = parser.parse_args()
    try:
        manifest = merge_retrieval_exports(args.inputs, args.output)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    print(json.dumps({
        "status": "passed", "document_count": manifest["document_count"],
        "producer_document_count": manifest["producer_document_count"],
        "unit_count": manifest["unit_count"], "parent_count": manifest["parent_count"],
        "retrieval_units": str(args.output / "retrieval-units.jsonl"),
        "parents": str(args.output / "retrieval-parents.jsonl"),
        "manifest": str(args.output / "manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
