"""新管线基准：预热、重复取中位数、输入/源码哈希和逐轮落盘。

示例：
  python scripts/benchmark_parser.py --input var/incoming --output var/bench-001 --rounds 3

计时包含每轮的预处理、模型推理和结果写出；所有轮次共享同一组已加载模型，
因此 warm 轮不包含模型加载。首轮 warm-up 不进入汇总。该脚本不声称测量精度。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doc_parser.knowledge_units import KnowledgeUnitConfig
from doc_parser.orchestration import run_path
from doc_parser.pipeline import ParserConfig
from doc_parser.runtime_pool import PageRuntimePool, gpu_memory_snapshot


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(path: Path) -> dict:
    path = path.resolve()
    if path.is_file():
        return {"type": "file", "sha256": file_sha256(path), "bytes": path.stat().st_size}
    rows = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append({"path": item.relative_to(path).as_posix(), "sha256": file_sha256(item)})
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"type": "folder", "file_count": len(rows), "tree_sha256": hashlib.sha256(canonical).hexdigest()}


def snapshot_sources(output: Path) -> dict[str, str]:
    hashes = {}
    sources = [
        ROOT / "scripts" / "run_parser.py", ROOT / "scripts" / "light_ocr_worker.py",
        ROOT / "scripts" / "benchmark_parser.py", ROOT / "scripts" / "build_knowledge_units.py",
    ]
    sources += sorted((ROOT / "src" / "doc_parser").glob("*.py"))
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / "snapshots" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = file_sha256(source)
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="必须是不存在或为空的实验目录")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build-knowledge-units", action="store_true", help="每轮包含知识单元构建与四个文件写出")
    parser.add_argument("--knowledge-unit-config", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int)
    parser.add_argument("--gpu-workers", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.warmups < 0:
        parser.error("--rounds 必须大于 0，--warmups 不能为负数")
    output = args.output.resolve()
    source = args.input.resolve()
    if source.is_dir() and output.is_relative_to(source):
        parser.error("基准输出不能位于输入目录内，否则后续轮次会把前轮产物当成输入")
    if output.exists() and any(output.iterdir()):
        parser.error("为避免混合两次实验，--output 必须为空")
    output.mkdir(parents=True, exist_ok=True)

    if args.knowledge_unit_config and not args.build_knowledge_units:
        parser.error("--knowledge-unit-config 需要与 --build-knowledge-units 一起使用")
    knowledge_unit_config = KnowledgeUnitConfig.from_json(args.knowledge_unit_config) if args.knowledge_unit_config else None
    config = ParserConfig.from_json(args.config) if args.config else ParserConfig()
    overrides = {key: value for key, value in {
        "cpu_workers": args.cpu_workers, "gpu_workers": args.gpu_workers,
    }.items() if value is not None}
    if overrides:
        config = replace(config, **overrides)

    report = {
        "schema_version": "parser-benchmark.v1",
        "input": str(args.input.resolve()),
        "input_fingerprint": input_fingerprint(args.input),
        "config": config.to_dict(),
        "build_knowledge_units": args.build_knowledge_units,
        "knowledge_unit_config": (
            knowledge_unit_config.to_dict() if knowledge_unit_config is not None else KnowledgeUnitConfig().to_dict()
        ) if args.build_knowledge_units else None,
        "rounds": args.rounds,
        "warmups": args.warmups,
        "timing_scope": (
            "每轮完整预处理、推理、知识单元构建与写出；warm 轮共享已加载模型，不含模型加载"
            if args.build_knowledge_units else
            "每轮完整预处理、推理与写出；warm 轮共享已加载模型，不含模型加载"
        ),
        "source_sha256": snapshot_sources(output),
        "runs": [],
    }
    pool = None
    try:
        if not args.prepare_only:
            pool = PageRuntimePool(ROOT, config)
            report["runtime_pool"] = pool.parallelism_report()
        write_json(output / "report.json", report)
        total_rounds = args.warmups + args.rounds
        for index in range(total_rounds):
            phase = "warmup" if index < args.warmups else "warm"
            measured_round = index - args.warmups + 1 if phase == "warm" else index + 1
            before = None if args.prepare_only else gpu_memory_snapshot()
            started = time.perf_counter()
            result = run_path(
                args.input,
                output / f"{phase}-{measured_round:03d}",
                config=config,
                knowledge_unit_config=knowledge_unit_config,
                prepare_only=args.prepare_only,
                build_knowledge_units=args.build_knowledge_units,
                runtime_pool=pool,
            )
            elapsed = time.perf_counter() - started
            after = None if args.prepare_only else gpu_memory_snapshot()
            row = {
                "phase": phase,
                "round": measured_round,
                "seconds": elapsed,
                "status": result.get("status"),
                "memory_before": asdict(before) if before is not None else None,
                "memory_after": asdict(after) if after is not None else None,
                "result_manifest": result,
            }
            report["runs"].append(row)
            # 每轮立即保存，长实验中断后仍能分析已完成部分。
            write_json(output / "report.json", report)
            with (output / "runs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(json.dumps({key: row[key] for key in ("phase", "round", "seconds", "status")}), flush=True)

        warm = [row["seconds"] for row in report["runs"] if row["phase"] == "warm"]
        report["warm_summary_seconds"] = {
            "count": len(warm), "median": statistics.median(warm),
            "mean": statistics.mean(warm), "min": min(warm), "max": max(warm),
        }
        report["status"] = "success"
        write_json(output / "report.json", report)
        print(json.dumps(report["warm_summary_seconds"], ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        write_json(output / "report.json", report)
        raise
    finally:
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
