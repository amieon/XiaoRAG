r"""用已微调模型预测 heading/body，不覆盖原始 page_result 或 DocumentIR。

python scripts/classify_headings.py --input var/parsed --model var/heading-model-exp01 --output var/heading-predictions-exp01.jsonl
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict binary heading/body labels")
    parser.add_argument("--input", required=True, help="page_result.json or a directory containing them")
    parser.add_argument("--model", required=True, help="Directory saved by train_heading_classifier.py")
    parser.add_argument("--output", required=True, help="New prediction JSONL path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise ValueError("Prediction output already exists; use a new file")
    examples = load_heading_examples(args.input)
    classifier = TransformerHeadingClassifier(
        args.model, device=args.device, max_length=args.max_length,
    )
    predictions = classifier.predict(examples, batch_size=args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "success", "count": len(predictions), "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
