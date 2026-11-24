#!/usr/bin/env python3
"""将现有 embeddings.npy 与 units.jsonl 导入 pgvector，不重新运行嵌入模型。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.pgvector_store import store_dense_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path, help="已有 retrieval-index 目录")
    parser.add_argument("--schema", required=True, help="新的 lowercase schema，例如 retrieval_exp01")
    parser.add_argument("--dsn-env", default="PGVECTOR_DSN", help="保存 DSN 的环境变量名，不在命令行传密码")
    args = parser.parse_args()
    dsn = os.environ.get(args.dsn_env, "")
    try:
        result = store_dense_index(args.index, dsn=dsn, schema=args.schema)
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    except Exception as exc:
        # 数据库驱动异常只显示消息；DSN 不写入日志。
        parser.error(f"pgvector import failed: {exc}")
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
