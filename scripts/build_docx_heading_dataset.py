#!/usr/bin/env python3
r"""DOCX 原生直读 → 零样本标题补充 → 章节树 → 带 Heading 样式的新 DOCX。

全程不运行 OCR，不覆盖原文件，输出目录必须为空。
PowerShell 示例：
  $env:HEADING_MODEL_API_KEY = '你的密钥'
  .venv\Scripts\python.exe scripts\build_docx_heading_dataset.py --input datasets\raw-docx --output datasets\styled-exp01 --base-url http://127.0.0.1:8000/v1 --model Qwen
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chapter_tree.groups import chapter_groups_document
from chapter_tree.heading_hierarchy import HIERARCHY_MODES, HeadingHierarchyResolver
from chapter_tree.local_hf_classifier import LocalHFHeadingClassifier
from chapter_tree.native_classifier import NativeHeadingClassifier
from chapter_tree.tree import ChapterTreeConfig, build_chapter_tree_from_ir
from chapter_tree.zero_shot_classifier import OpenAICompatibleHeadingClassifier
from doc_parser.docx_native_ir import (
    heading_examples_from_native_ir, native_docx_to_ir, write_styled_docx,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sources(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.casefold() != ".docx":
            raise ValueError("Input file must be DOCX")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(
        path for path in source.rglob("*.docx")
        if path.is_file() and not path.name.startswith("~$") and not path.is_symlink()
    )
    if not files:
        raise ValueError(f"Input folder contains no DOCX files: {source}")
    return files


def build_docx_heading_dataset(
    source: str | Path,
    output: str | Path,
    *,
    classifier: Any = None,
    batch_size: int = 8,
    hierarchy_mode: str = "streaming",
    hierarchy_resolver: Any = None,
    native_only: bool = False,
) -> dict[str, Any]:
    if not native_only and classifier is None:
        raise ValueError("Hybrid DOCX dataset generation requires a classifier")
    if native_only and hierarchy_resolver is not None:
        raise ValueError("native_only does not accept a hierarchy resolver")
    effective_hierarchy_mode = "native" if native_only else hierarchy_mode
    source_path = Path(source).resolve()
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Dataset output directory must be empty; use a new experiment directory")
    if source_path.is_dir() and output_root.is_relative_to(source_path):
        raise ValueError("Dataset output must not be inside the input folder")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in _sources(source_path):
        relative = path.name if source_path.is_file() else path.relative_to(source_path).as_posix()
        stem_path = Path(relative).with_suffix("")
        document_output = output_root / "documents" / stem_path
        ir = native_docx_to_ir(path)
        examples = heading_examples_from_native_ir(ir)
        if native_only:
            # 纯原生 baseline：无模型、无编号补标题；resolver 只给原生标题写审计字段。
            predictions = NativeHeadingClassifier().predict(examples, batch_size=batch_size)
            predictions = HeadingHierarchyResolver(None, mode="streaming").resolve(examples, predictions)
        else:
            predictions = classifier.predict(examples, batch_size=batch_size)
            resolver = hierarchy_resolver or HeadingHierarchyResolver(
                classifier, mode=hierarchy_mode,
            )
            predictions = resolver.resolve(examples, predictions)
        tree, blocks = build_chapter_tree_from_ir(
            ir.to_dict(), config=ChapterTreeConfig(
                heading_mode="native" if native_only else "hybrid",
                flat_headings=not native_only and hierarchy_mode == "none",
            ),
            predictions=predictions,
        )
        groups = chapter_groups_document(tree, blocks)
        styled_path = document_output / "styled.docx"
        style_stats = write_styled_docx(path, styled_path, ir, predictions)

        document_output.mkdir(parents=True, exist_ok=True)
        ir.to_json(document_output / "document.json")
        _write_json(document_output / "chapter-tree.json", tree)
        _write_json(document_output / "chapter-groups.json", groups)
        with (document_output / "heading-predictions.jsonl").open("x", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        rows.append({
            "source": relative,
            "document_id": ir.document_id,
            "output": document_output.relative_to(output_root).as_posix(),
            "styled_docx": styled_path.relative_to(output_root).as_posix(),
            "native_block_count": sum(len(page.blocks) for page in ir.pages),
            "classified_paragraph_count": len(examples),
            "native_heading_count": sum(row.get("native_heading_count", 0) > 0 for row in predictions),
            "model_heading_count": style_stats["model_heading_count"],
            "heading_mode": "native" if native_only else "hybrid",
            "hierarchy_mode": effective_hierarchy_mode,
            "hierarchy_model_decision_count": sum(
                row.get("hierarchy_source") in {
                    "streaming_model", "streaming_context_model", "global_model",
                } for row in predictions
            ),
            "tree_max_depth": tree["stats"]["max_depth"],
            "tree_section_count": tree["stats"]["section_count"],
            "flat_document": tree["stats"]["flat_document"],
            "warning_count": len(tree["warnings"]),
            **style_stats,
        })
    manifest = {
        "schema_version": "docx-heading-dataset.v1",
        "status": "success",
        "source": str(source_path),
        "document_count": len(rows),
        "ocr_required": False,
        "source_files_modified": False,
        "model": getattr(classifier, "model", None),
        "heading_mode": "native" if native_only else "hybrid",
        "hierarchy_mode": effective_hierarchy_mode,
        "documents": rows,
    }
    _write_json(output_root / "dataset-manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--native-only", action="store_true",
        help="纯原生 baseline：只信 Heading／OOXML 大纲，不调用模型，也不用编号补标题",
    )
    parser.add_argument(
        "--local-model", type=Path,
        help="HF 本地模型目录（含 config.json）；Transformers 直接加载，无需 Ollama／vLLM",
    )
    parser.add_argument("--device", default="auto", help="本地模型 device_map，默认 auto；CPU 可设 cpu")
    parser.add_argument("--base-url", help="OpenAI 兼容 /v1 地址；也可用 HEADING_MODEL_BASE_URL")
    parser.add_argument("--model", help="远程模型名；也可用 HEADING_MODEL_NAME")
    parser.add_argument("--batch-size", type=int, default=8, help="每次请求判别的段落数")
    parser.add_argument(
        "--hierarchy-mode", choices=HIERARCHY_MODES, default="streaming",
        help="none=平铺 baseline；streaming=活动路径；streaming-context=路径加上下文；global=完整标题序列",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=4096, help="包含思考内容在内的最大输出 token 数")
    parser.add_argument("--disable-thinking", action="store_true", help="Qwen 兼容服务显式关闭思考")
    parser.add_argument("--without-response-format", action="store_true", help="服务不支持 response_format 时关闭 JSON 模式")
    args = parser.parse_args()
    if args.native_only and args.local_model:
        parser.error("--native-only 与 --local-model 不能同时使用")
    if args.local_model:
        classifier = LocalHFHeadingClassifier(
            args.local_model,
            device=args.device,
            max_tokens=args.max_tokens,
            disable_thinking=args.disable_thinking,
            use_response_format=not args.without_response_format,
        )
    elif args.native_only:
        classifier = None
    else:
        classifier = OpenAICompatibleHeadingClassifier(
            base_url=args.base_url or os.environ.get("HEADING_MODEL_BASE_URL", ""),
            model=args.model or os.environ.get("HEADING_MODEL_NAME", ""),
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            disable_thinking=args.disable_thinking,
            use_response_format=not args.without_response_format,
        )
    result = build_docx_heading_dataset(
        args.input, args.output, classifier=classifier, batch_size=args.batch_size,
        hierarchy_mode=args.hierarchy_mode, native_only=args.native_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
