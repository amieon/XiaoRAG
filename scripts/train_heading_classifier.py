r"""微调 heading/body 二分类器。

python scripts/train_heading_classifier.py --input var/parsed --labels datasets/heading-labels.jsonl --output var/heading-model-exp01
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
    load_heading_examples, split_examples_by_document, train_transformer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a binary heading/body Transformer")
    parser.add_argument("--input", required=True, help="page_result.json or a directory containing them")
    parser.add_argument("--labels", required=True, help="JSONL: block_id + heading/body label")
    parser.add_argument("--output", required=True, help="New, empty experiment/model directory")
    parser.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    examples = load_heading_examples(args.input, args.labels, require_labels=True)
    train, validation = split_examples_by_document(
        examples, val_ratio=args.val_ratio, seed=args.seed,
    )
    summary = train_transformer(
        train, validation, base_model=args.base_model, output=args.output,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, max_length=args.max_length,
        seed=args.seed, device=args.device,
    )
    print(json.dumps({
        "status": "success", "output": str(Path(args.output).resolve()),
        "train_count": len(train), "validation_count": len(validation),
        "last_epoch": summary["history"][-1],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
