#!/usr/bin/env python3
"""备用独立入口：调用 parser 的原生文本无损分段与旧 IR 迁移。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.native_text import DEFAULT_NATIVE_BLOCK_CHARS, reflow_native_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="递归扫描 document.json")
    parser.add_argument("--predictions", type=Path, help="可选：同步移除受影响文档的旧预测行")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_NATIVE_BLOCK_CHARS)
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error("--input 必须是目录")
    if args.predictions is not None and not args.predictions.is_file():
        parser.error("--predictions 必须是已存在的 JSONL")
    result = reflow_native_documents(
        args.input, predictions=args.predictions, max_chars=args.max_chars,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
