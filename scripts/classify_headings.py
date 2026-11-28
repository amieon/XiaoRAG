r"""预测 heading/body，不覆盖原始 page_result 或 DocumentIR。

原生模式（默认，无需训练）：
python scripts/classify_headings.py --input var/parsed --mode native --output var/heading-native.jsonl
零样本通用模型模式（密钥只从 HEADING_MODEL_API_KEY 环境变量读取）：
python scripts/classify_headings.py --input var/parsed --mode zero-shot --base-url http://127.0.0.1:8000/v1 --model Qwen --output var/heading-zero-shot.jsonl
微调模型模式：
python scripts/classify_headings.py --input var/parsed --mode transformer --model var/heading-model-exp01 --output var/heading-model.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chapter_tree.heading_classifier import (
    TransformerHeadingClassifier, load_heading_examples,
)
from chapter_tree.heading_hierarchy import HIERARCHY_MODES, predict_document_structure
from chapter_tree.local_hf_classifier import LocalHFHeadingClassifier
from chapter_tree.native_classifier import NativeHeadingClassifier
from chapter_tree.zero_shot_classifier import OpenAICompatibleHeadingClassifier
from doc_parser.native_text import reflow_native_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict binary heading/body labels")
    parser.add_argument("--input", required=True, help="page_result.json or a directory containing them")
    parser.add_argument("--mode", choices=("native", "transformer", "zero-shot"), default="native")
    parser.add_argument("--model", help="transformer 的模型目录，或远程 zero-shot 的模型名；也可用 HEADING_MODEL_NAME")
    parser.add_argument("--local-model", help="zero-shot 的 HF 本地模型目录，直接由 Transformers 加载")
    parser.add_argument("--base-url", help="远程 zero-shot 的 OpenAI 兼容 /v1 地址；也可用 HEADING_MODEL_BASE_URL")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--without-response-format", action="store_true")
    parser.add_argument("--output", required=True, help="New prediction JSONL path")
    parser.add_argument(
        "--resume", action="store_true",
        help="从同一输入、模型和层级模式的按文档 JSONL 检查点续跑",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--hierarchy-mode", choices=HIERARCHY_MODES,
        help="zero-shot 分类后继续用同一模型恢复 parent_id；按文档隔离标题栈",
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = Path(args.output)
    if args.local_model and args.mode != "zero-shot":
        raise ValueError("--local-model 只用于 --mode zero-shot")
    if args.hierarchy_mode and args.mode != "zero-shot":
        raise ValueError("--hierarchy-mode 需要与 --mode zero-shot 一起使用")
    if args.resume and not args.hierarchy_mode:
        raise ValueError("--resume 目前需要与 --hierarchy-mode 一起使用")
    if output.exists() and not args.resume:
        raise ValueError("Prediction output already exists; use a new file or --resume")
    # 新 IR 已由 parser 分段；这里仅自动迁移旧版超长的 native_only IR。
    # 必须在读取续跑检查点之前完成，以便移除受影响文档的陈旧块预测。
    reflow = reflow_native_documents(
        args.input,
        predictions=output if args.resume and output.exists() else None,
    )
    if reflow["changed_document_count"]:
        print(
            f"[native-reflow] migrated={reflow['changed_document_count']}; "
            f"stale_predictions_removed={reflow['removed_stale_prediction_count']}",
            file=sys.stderr, flush=True,
        )
    resumed_predictions = []
    if args.resume and output.exists():
        with output.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("document_id"), str):
                    raise ValueError(f"Invalid resume prediction at line {line_number}")
                resumed_predictions.append(row)
    completed_document_ids = {row["document_id"] for row in resumed_predictions}
    examples = load_heading_examples(args.input)
    if args.mode == "native":
        classifier = NativeHeadingClassifier()
    elif args.mode == "zero-shot":
        if args.local_model:
            classifier = LocalHFHeadingClassifier(
                args.local_model,
                device=args.device,
                max_tokens=args.max_tokens,
                disable_thinking=args.disable_thinking,
                use_response_format=not args.without_response_format,
            )
        else:
            classifier = OpenAICompatibleHeadingClassifier(
                base_url=args.base_url or os.environ.get("HEADING_MODEL_BASE_URL", ""),
                model=args.model or os.environ.get("HEADING_MODEL_NAME", ""),
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                disable_thinking=args.disable_thinking,
                use_response_format=not args.without_response_format,
            )
    else:
        if not args.model:
            raise ValueError("--model is required in transformer mode")
        classifier = TransformerHeadingClassifier(
            args.model, device=args.device, max_length=args.max_length,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = list(resumed_predictions)
    if args.hierarchy_mode:
        grouped = {}
        for example in examples:
            grouped.setdefault(example.document_id, []).append(example)
        with output.open("a" if args.resume else "x", encoding="utf-8") as stream:
            for document_id, document_examples in grouped.items():
                if document_id in completed_document_ids:
                    print(f"[structure] resume skip completed document: {document_id}", flush=True)
                    continue
                document_predictions = predict_document_structure(
                    document_examples, classifier,
                    batch_size=args.batch_size, hierarchy_mode=args.hierarchy_mode,
                )
                for prediction in document_predictions:
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                stream.flush()
                predictions.extend(document_predictions)
                print(
                    f"[structure] checkpoint saved: document={document_id}; "
                    f"blocks={len(document_predictions)}",
                    flush=True,
                )
    else:
        predictions = classifier.predict(examples, batch_size=args.batch_size)
        with output.open("x", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "success", "mode": args.mode, "count": len(predictions),
        "heading_count": sum(item["label"] == "heading" for item in predictions),
        "body_count": sum(item["label"] == "body" for item in predictions),
        "document_count": len({item.document_id for item in examples}),
        "resumed_document_count": len(completed_document_ids),
        "hierarchy_mode": args.hierarchy_mode,
        "hierarchy_model_decision_count": sum(
            item.get("hierarchy_source") in {
                "streaming_model", "streaming_context_model", "global_model",
            } for item in predictions
        ),
        "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
