#!/usr/bin/env python3
"""从 DocumentIR 输出独立 chapter-tree.v1 与 chapter-groups.v1，不覆盖 IR。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chapter_tree.groups import chapter_groups_document
from chapter_tree.tree import ChapterTreeConfig, build_chapter_tree_from_ir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="DocumentIR document.json")
    parser.add_argument("--output", required=True, type=Path, help="新的输出目录")
    parser.add_argument("--mode", choices=("native", "upstream", "hybrid"), default="hybrid")
    parser.add_argument("--predictions", type=Path, help="可选：hybrid 模式的标题分类 JSONL")
    parser.add_argument("--no-numbered-promotion", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"输入不存在：{args.input}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录必须为空，避免混合两次实验产物")
    if args.mode != "hybrid" and args.predictions:
        parser.error("--predictions 只用于 --mode hybrid")
    predictions = None
    if args.predictions:
        if not args.predictions.is_file():
            parser.error(f"预测文件不存在：{args.predictions}")
        predictions = []
        with args.predictions.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        predictions.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        parser.error(f"预测 JSONL 第 {line_no} 行无效：{exc}")
    try:
        ir = json.loads(args.input.read_text(encoding="utf-8"))
        config = ChapterTreeConfig(
            heading_mode=args.mode,
            promote_numbered_headings=not args.no_numbered_promotion,
        )
        tree, blocks = build_chapter_tree_from_ir(ir, config=config, predictions=predictions)
        grouped = chapter_groups_document(tree, blocks)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chapter-tree.json").write_text(
        json.dumps(tree, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.output / "chapter-groups.json").write_text(
        json.dumps(grouped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "document_id": tree["document_id"],
        "tree": str(args.output / "chapter-tree.json"),
        "groups": str(args.output / "chapter-groups.json"),
        "stats": tree["stats"],
        "group_count": len(grouped["groups"]),
        "warning_count": len(tree["warnings"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
