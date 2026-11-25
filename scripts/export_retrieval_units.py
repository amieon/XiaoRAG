#!/usr/bin/env python3
"""把任一切块生产者的 chunks.jsonl 导出为 M2 统一检索单元。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_retrieval.adapter import PRODUCERS, adapt_structural_chunks


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records) -> None:
    path.write_text("".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records
    ), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="单文档 chunks.jsonl")
    parser.add_argument("--security", required=True, type=Path, help="显式权限策略 JSON")
    parser.add_argument("--producer", required=True, choices=sorted(PRODUCERS))
    parser.add_argument("--output", required=True, type=Path, help="新的空输出目录")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("输入 chunks.jsonl 不存在")
    if not args.security.is_file():
        parser.error("权限策略 JSON 不存在")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录必须为空，避免混合两次实验产物")
    try:
        raw = args.input.read_bytes()
        chunks = []
        for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"chunks JSONL 第 {line_no} 行无效：{exc}") from exc
        security = json.loads(args.security.read_text(encoding="utf-8-sig"))
        result = adapt_structural_chunks(chunks, security=security, producer=args.producer)
        chunking_values = {
            json.dumps(chunk.get("metadata", {}).get("chunking"), ensure_ascii=False, sort_keys=True)
            for chunk in chunks
        }
        if len(chunking_values) != 1:
            raise ValueError("chunks_inconsistent_chunking_configuration")
        chunking = json.loads(chunking_values.pop())
    except (ValueError, UnicodeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "retrieval-units.jsonl", result["retrieval_units"])
    _write_jsonl(args.output / "retrieval-parents.jsonl", result["parents"])
    _write_json(args.output / "manifest.json", {
        "schema_version": "retrieval-preparation.v1",
        "document_id": result["document_id"], "producer": result["producer"],
        "source_chunks_sha256": hashlib.sha256(raw).hexdigest(),
        "unit_count": len(result["retrieval_units"]), "parent_count": len(result["parents"]),
        "embedding_field": "embedding_text", "vectorization_performed": False,
        "chunking": chunking,
        "retrieval_weight_training_performed": False,
        "self_check": result["self_check"],
    })
    print(json.dumps({
        "document_id": result["document_id"], "producer": result["producer"],
        "retrieval_units": str(args.output / "retrieval-units.jsonl"),
        "parents": str(args.output / "retrieval-parents.jsonl"),
        "unit_count": len(result["retrieval_units"]),
        "parent_count": len(result["parents"]), "status": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
