r"""把单个文件、压缩包或整个文件夹解析为 DocumentIR。

Windows PowerShell 示例：
  .venv\Scripts\python.exe scripts\run_parser.py --input var\incoming --output var\prepared --prepare-only
  .venv\Scripts\python.exe scripts\run_parser.py --input var\incoming --output var\parsed --cpu-workers 4 --gpu-workers 0
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doc_parser.knowledge_units import KnowledgeUnitConfig
from doc_parser.orchestration import run_path
from doc_parser.pipeline import ParserConfig
from doc_parser.routing import GateConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="文件、压缩包、图片或目录")
    parser.add_argument("--output", type=Path, required=True, help="新的空输出目录")
    parser.add_argument("--config", type=Path, help="解析 JSON 配置；默认读取内置审计参数")
    parser.add_argument("--build-knowledge-units", action="store_true", help="解析后生成知识单元和元素位置表")
    parser.add_argument("--knowledge-unit-config", type=Path, help="知识单元 JSON 配置，例如 config/knowledge_units.json")
    parser.add_argument("--risk-threshold", type=float, help="只覆盖 gate.risk_threshold")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--office-converter", help="LibreOffice/soffice 可执行文件")
    parser.add_argument("--max-pages-per-document", type=int)
    parser.add_argument("--cpu-workers", type=int, help="文档级纯 CPU 并发；0 为自动")
    parser.add_argument("--gpu-workers", type=int, help="视觉运行时数；0 为按空闲显存自动准入")
    parser.add_argument("--max-gpu-workers", type=int, help="自动模式的运行时上限")
    parser.add_argument("--gpu-free-margin-mb", type=int, help="每个新增运行时必须预留的 MiB 数")
    parser.add_argument("--prepare-only", action="store_true", help="不加载模型，仅预处理并生成 IR 页骨架")
    args = parser.parse_args()

    if args.knowledge_unit_config and not args.build_knowledge_units:
        parser.error("--knowledge-unit-config 需要与 --build-knowledge-units 一起使用")
    knowledge_unit_config = KnowledgeUnitConfig.from_json(args.knowledge_unit_config) if args.knowledge_unit_config else None
    config = ParserConfig.from_json(args.config) if args.config else ParserConfig()
    if args.risk_threshold is not None:
        config = replace(config, gate=GateConfig(**{
            **asdict(config.gate), "risk_threshold": args.risk_threshold,
        }))
    overrides = {
        name: value for name, value in {
            "cpu_workers": args.cpu_workers,
            "gpu_workers": args.gpu_workers,
            "max_gpu_workers": args.max_gpu_workers,
            "gpu_free_margin_mb": args.gpu_free_margin_mb,
        }.items() if value is not None
    }
    if overrides:
        config = replace(config, **overrides)
    result = run_path(
        args.input,
        args.output,
        config=config,
        knowledge_unit_config=knowledge_unit_config,
        dpi=args.dpi,
        office_converter=args.office_converter,
        prepare_only=args.prepare_only,
        build_knowledge_units=args.build_knowledge_units,
        max_pages_per_document=args.max_pages_per_document,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return {"success": 0, "partial": 2}.get(result.get("status"), 1)


if __name__ == "__main__":
    raise SystemExit(main())
