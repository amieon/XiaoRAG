r"""预测 heading/body，不覆盖原始 page_result 或 DocumentIR。

原生模式（默认，无需训练）：
python scripts/classify_headings.py --input var/parsed --mode native --output var/heading-native.jsonl
模型模式：
python scripts/classify_headings.py --input var/parsed --mode transformer --model var/heading-model-exp01 --output var/heading-model.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chapter_tree.heading_classifier import (
    TransformerHeadingClassifier, load_heading_examples,
)
from chapter_tree.native_classifier import NativeHeadingClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict binary heading/body labels")
    parser.add_argument("--input", required=True, help="page_result.json or a directory containing them")
    parser.add_argument("--mode", choices=("native", "transformer"), default="native")
    parser.add_argument("--model", help="Required in transformer mode; unused in native mode")
    parser.add_argument("--output", required=True, help="New prediction JSONL path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise ValueError("Prediction output already exists; use a new file")
    examples = load_heading_examples(args.input)
    if args.mode == "native":
        classifier = NativeHeadingClassifier()
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
