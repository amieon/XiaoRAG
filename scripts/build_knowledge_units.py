r"""从已有 DocumentIR 离线构建知识单元，不加载任何识别模型。

单个 document.json 或结果目录：
  python scripts/build_knowledge_units.py --input var/parsed/results/document.json --output var/knowledge-units --config config/knowledge_units.json
  python scripts/build_knowledge_units.py --input var/parsed/results --output var/knowledge-units

递归批处理（每份 IR 独立输出，失败不阻断其他文档）：
  python scripts/build_knowledge_units.py --input var/parsed-batch --output var/knowledge-units-batch --workers 4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doc_parser.knowledge_units import KnowledgeUnitConfig, build_knowledge_units_from_json

IGNORED_DIRECTORIES = {".git", ".svn", ".venv", ".idea", "__pycache__", "node_modules"}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _single_document(path: Path) -> Path | None:
    """识别 document.json、results/ 或包含 results/document.json 的单文档输出目录。"""
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    for candidate in (path / "document.json", path / "results" / "document.json"):
        if candidate.is_file():
            return candidate
    return None


def _discover_documents(root: Path, output: Path) -> list[Path]:
    """递归发现 DocumentIR；跳过输出目录和符号链接，避免自读产物。"""
    documents: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current).resolve()
        kept = []
        for name in sorted(directories):
            candidate = current_path / name
            resolved = candidate.resolve()
            if name in IGNORED_DIRECTORIES or candidate.is_symlink():
                continue
            if resolved == output or output.is_relative_to(resolved):
                continue
            kept.append(name)
        directories[:] = kept
        if "document.json" not in names:
            continue
        candidate = current_path / "document.json"
        if candidate.is_symlink() or not candidate.is_file() or candidate.is_relative_to(output):
            continue
        documents.append(candidate)
    return sorted(documents, key=lambda path: path.relative_to(root).as_posix().casefold())


def _worker_count(requested: int, document_count: int) -> int:
    if requested < 0:
        raise ValueError("workers must be non-negative")
    if requested:
        return min(requested, document_count)
    return min(4, os.cpu_count() or 1, document_count)


def _run_single(source: Path, output: Path, config: KnowledgeUnitConfig) -> dict:
    result = build_knowledge_units_from_json(source, output, config=config)
    return {
        "status": "success",
        "input": str(source),
        "output": str(output),
        "document_id": result.manifest["document_id"],
        "counts": result.manifest["counts"],
        "coverage_ratio": result.manifest["coverage_ratio"],
    }


def _run_batch(root: Path, output: Path, config: KnowledgeUnitConfig, workers: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Knowledge unit batch output directory must be empty; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    documents = _discover_documents(root, output)
    if not documents:
        raise ValueError(f"No document.json found below: {root}")
    worker_count = _worker_count(workers, len(documents))

    def process(source: Path) -> dict:
        relative = source.relative_to(root)
        target = output / relative.parent / "knowledge-units"
        summary = {
            "source": relative.as_posix(),
            "output": target.relative_to(output).as_posix(),
        }
        try:
            result = build_knowledge_units_from_json(source, target, config=config)
            summary.update({
                "status": "success",
                "document_id": result.manifest["document_id"],
                "counts": result.manifest["counts"],
                "coverage_ratio": result.manifest["coverage_ratio"],
            })
        except Exception as exc:
            summary.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            _write_json(target.parent / "knowledge-units-error.json", summary)
        return summary

    if worker_count == 1:
        summaries = [process(source) for source in documents]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="knowledge-unit-cpu") as executor:
            summaries = list(executor.map(process, documents))
    success = sum(item["status"] == "success" for item in summaries)
    failed = len(summaries) - success
    status = "success" if failed == 0 else "partial" if success else "failed"
    manifest = {
        "schema_version": "knowledge-units-batch.v1",
        "status": status,
        "input_root": str(root),
        "document_count": len(documents),
        "counts": {"success": success, "failed": failed},
        "cpu_worker_count": worker_count,
        "config": config.to_dict(),
        "files": summaries,
    }
    _write_json(output / "batch-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="document.json、结果目录或批量根目录")
    parser.add_argument("--output", type=Path, required=True, help="不存在或为空的输出目录")
    parser.add_argument("--config", type=Path, help="知识单元 JSON 配置；缺省使用内置参数")
    parser.add_argument("--workers", type=int, default=0, help="批量 CPU 线程数；0 为自动，最多 4")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers 不能为负数")
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.exists():
        parser.error(f"输入不存在：{source}")
    config = KnowledgeUnitConfig.from_json(args.config) if args.config else KnowledgeUnitConfig()

    single = _single_document(source)
    if single is not None:
        result = _run_single(single, output, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not source.is_dir():
        parser.error("输入必须是 document.json、结果目录或目录")
    result = _run_batch(source, output, config, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"success": 0, "partial": 2}.get(result["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
