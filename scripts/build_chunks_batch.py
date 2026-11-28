#!/usr/bin/env python3
"""批量生成章节树与 CDC 父子块；单文档失败不阻断其他文档。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chapter_tree.tree import ChapterTreeConfig
from doc_chunking.engine import chunk_document, load_config
from doc_chunking.validation import validate_result

OUTPUT_FILES = (
    "chunks.jsonl", "chapter-tree.json", "preprocessing.json",
    "decision.json", "self_check.json",
)


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def _save(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _document_output(document_path: Path, input_root: Path, output_root: Path) -> tuple[str, Path]:
    document_dir = document_path.parent
    if document_dir.name == "results":
        document_dir = document_dir.parent
    relative = document_dir.relative_to(input_root)
    name = relative.as_posix() if relative.parts else document_path.parent.name
    return name, output_root / relative


def _completed(output: Path) -> bool:
    if not all((output / name).is_file() for name in OUTPUT_FILES):
        return False
    try:
        check = _read_json(output / "self_check.json")
        return check.get("integrity", {}).get("status") == "passed"
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="递归查找 DocumentIR document.json 的根目录")
    parser.add_argument("--output-root", required=True, type=Path, help="批量切块输出根目录")
    parser.add_argument("--predictions", required=True, type=Path, help="全量标题／层级预测 JSONL")
    parser.add_argument("--config", type=Path, help="默认 src/doc_chunking/config.json")
    parser.add_argument("--mode", choices=("native", "upstream", "hybrid"), default="hybrid")
    parser.add_argument("--flat-headings", action="store_true")
    parser.add_argument("--no-numbered-promotion", action="store_true")
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        parser.error("--input 必须是存在的目录")
    if not args.predictions.is_file():
        parser.error("--predictions 必须是存在的 JSONL 文件")
    if args.mode != "hybrid":
        parser.error("当前批量入口要求 --mode hybrid，避免提供预测却静默降级")
    prediction_bytes = args.predictions.read_bytes()
    predictions_by_document = defaultdict(list)
    try:
        for line_number, line in enumerate(prediction_bytes.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("document_id"), str):
                raise ValueError(f"预测 JSONL 第 {line_number} 行缺少 document_id")
            predictions_by_document[row["document_id"]].append(row)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return

    # 任何合法 DocumentIR 都可切块，不按目录名称或伴随文件排除来源。
    # 仅排除本次输出根目录，防止批量任务读取自己正在生成的产物。
    documents = sorted(
        path for path in input_root.rglob("document.json")
        if path.is_file() and output_root != path.parent and output_root not in path.parents
    )
    if not documents:
        parser.error("输入目录下没有 document.json")
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    tree_config = ChapterTreeConfig(
        heading_mode=args.mode,
        promote_numbered_headings=not args.no_numbered_promotion,
        flat_headings=args.flat_headings,
    )
    manifest_path = output_root / "batch-manifest.json"
    records = []

    def save_manifest(status: str) -> None:
        _save(manifest_path, {
            "schema_version": "chunk-batch-manifest.v1",
            "status": status,
            "input_root": str(input_root),
            "output_root": str(output_root),
            "predictions": str(args.predictions.resolve()),
            "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            "mode": args.mode,
            "document_count": len(documents),
            "success_count": sum(row["status"] == "success" for row in records),
            "skipped_count": sum(row["status"] == "skipped" for row in records),
            "failure_count": sum(row["status"] == "failed" for row in records),
            "documents": records,
        })

    for index, document_path in enumerate(documents, 1):
        name, output = _document_output(document_path, input_root, output_root)
        print(f"[{index}/{len(documents)}] {name}", flush=True)
        if _completed(output):
            records.append({"document": name, "status": "skipped", "output": str(output)})
            save_manifest("running")
            print("  已完成，跳过", flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            message = "输出目录存在但没有完整通过的产物；请换用新输出根目录或手动清理该失败目录"
            records.append({"document": name, "status": "failed", "error": message})
            save_manifest("running")
            print(f"  失败：{message}", flush=True)
            continue
        try:
            ir = _read_json(document_path)
            document_id = ir["document_id"]
            predictions = predictions_by_document.get(document_id, [])
            if not predictions:
                raise ValueError(f"missing_heading_predictions: document_id={document_id!r}")
            native = _read_json(document_path.parent / "native" / "objects.json")
            source_manifest = _read_json(document_path.parent / "source_manifest.json")
            result = chunk_document(
                ir, native, source_manifest, config,
                tree_config=tree_config, predictions=predictions,
            )
            integrity = validate_result(result, ir, native)
            output.mkdir(parents=True, exist_ok=True)
            chunks = result["parents"] + result["children"]
            (output / "chunks.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in chunks),
                encoding="utf-8",
            )
            _save(output / "chapter-tree.json", result["chapter_tree"])
            _save(output / "preprocessing.json", result["preprocessing"])
            _save(output / "decision.json", result["decision"])
            _save(output / "self_check.json", {**result["self_check"], "integrity": integrity})
            records.append({
                "document": name, "document_id": document_id, "status": "success",
                "parent_count": len(result["parents"]), "child_count": len(result["children"]),
                "output": str(output),
            })
            print(f"  成功：parents={len(result['parents'])}, children={len(result['children'])}", flush=True)
        except Exception as exc:
            # KeyboardInterrupt/SystemExit 不在这里吞掉；普通单文档异常记录后继续。
            records.append({
                "document": name, "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc),
            })
            print(f"  失败：{type(exc).__name__}: {exc}", flush=True)
        save_manifest("running")

    failures = [row for row in records if row["status"] == "failed"]
    save_manifest("failed" if failures else "success")
    print(json.dumps({
        "status": "failed" if failures else "success",
        "document_count": len(documents),
        "success_count": sum(row["status"] == "success" for row in records),
        "skipped_count": sum(row["status"] == "skipped" for row in records),
        "failure_count": len(failures),
        "failed_documents": [row["document"] for row in failures],
        "failures": [
            {"document": row["document"], "error_type": row.get("error_type"), "error": row["error"]}
            for row in failures
        ],
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
