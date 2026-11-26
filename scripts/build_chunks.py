#!/usr/bin/env python3
"""从单份 document.json 生成章节树和 structural-chunk.v1 两层父子块。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chapter_tree.tree import ChapterTreeConfig
from doc_chunking.engine import chunk_document, load_config
from doc_chunking.validation import validate_result


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def _save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="单份 DocumentIR document.json")
    parser.add_argument("--output", required=True, type=Path, help="新的空输出目录")
    parser.add_argument("--config", type=Path, help="默认 src/doc_chunking/config.json")
    parser.add_argument("--mode", choices=("native", "upstream", "hybrid"), default="hybrid")
    parser.add_argument("--predictions", type=Path, help="可选：zero-shot/transformer 标题预测 JSONL")
    parser.add_argument(
        "--flat-headings", action="store_true",
        help="重放 hierarchy-mode=none：所有标题（含原生标题）平铺为一级",
    )
    parser.add_argument("--no-numbered-promotion", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file() or args.input.suffix.lower() != ".json":
        parser.error("--input 必须是存在的 document.json")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录必须为空，避免混合两次实验产物")
    try:
        ir = _read_json(args.input)
        native = _read_json(args.input.parent / "native" / "objects.json")
        manifest = _read_json(args.input.parent / "source_manifest.json")
        config = load_config(args.config)
        tree_config = ChapterTreeConfig(
            heading_mode=args.mode,
            promote_numbered_headings=not args.no_numbered_promotion,
            flat_headings=args.flat_headings,
        )
        predictions = None
        if args.predictions:
            if not args.predictions.is_file():
                raise ValueError(f"标题预测文件不存在：{args.predictions}")
            predictions = [
                json.loads(line) for line in args.predictions.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        if args.mode != "hybrid" and predictions is not None:
            raise ValueError("--predictions 只用于 --mode hybrid")
        result = chunk_document(
            ir, native, manifest, config, tree_config=tree_config, predictions=predictions,
        )
        integrity = validate_result(result, ir, native)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    chunks = result["parents"] + result["children"]
    (args.output / "chunks.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in chunks),
        encoding="utf-8",
    )
    _save(args.output / "chapter-tree.json", result["chapter_tree"])
    _save(args.output / "preprocessing.json", result["preprocessing"])
    _save(args.output / "decision.json", result["decision"])
    _save(args.output / "self_check.json", {**result["self_check"], "integrity": integrity})
    print(json.dumps({
        "document_id": result["document_id"], "strategy": result["decision"]["strategy"],
        "parent_count": len(result["parents"]), "child_count": len(result["children"]),
        "chunks": str(args.output / "chunks.jsonl"),
        "integrity": integrity["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
