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
from chapter_tree.local_hf_classifier import LocalHFHeadingClassifier
from chapter_tree.native_classifier import NativeHeadingClassifier
from chapter_tree.zero_shot_classifier import OpenAICompatibleHeadingClassifier


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = Path(args.output)
    if args.local_model and args.mode != "zero-shot":
        raise ValueError("--local-model 只用于 --mode zero-shot")
    if output.exists():
        raise ValueError("Prediction output already exists; use a new file")
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
    predictions = classifier.predict(examples, batch_size=args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "success", "mode": args.mode, "count": len(predictions),
        "heading_count": sum(item["label"] == "heading" for item in predictions),
        "body_count": sum(item["label"] == "body" for item in predictions),
        "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
