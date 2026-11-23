"""可训练的 heading/body 二分类器。

模型训练依赖 PyTorch 与 Transformers，但数据读取、输入构造和测试不要求安装它们。
二分类只决定“标题或正文”，标题层级由后续章节树恢复，不在这里假装完成。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Sequence


BODY_LABEL = "body"
HEADING_LABEL = "heading"
LABELS = (BODY_LABEL, HEADING_LABEL)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
LABEL_ALIASES = {
    "body": BODY_LABEL, "paragraph": BODY_LABEL, "text": BODY_LABEL,
    "heading": HEADING_LABEL, "title": HEADING_LABEL,
    "section_heading": HEADING_LABEL, "document_title": HEADING_LABEL,
    "doc_title": HEADING_LABEL, "paragraph_title": HEADING_LABEL,
}


@dataclass(frozen=True)
class HeadingExample:
    block_id: str
    page_id: str
    document_id: str
    text: str
    context: dict[str, Any]
    raw_label: str
    reading_order: int | None
    source_page_result: str
    label: str | None = None
    native_structures: list[dict[str, Any]] = field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(value)
    return rows


def _read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in _read_jsonl(path):
        block_id = str(row.get("block_id") or "").strip()
        raw_label = str(row.get("label") or "").strip().lower()
        if not block_id or raw_label not in LABEL_ALIASES:
            raise ValueError(
                f"{path}: expected block_id and label in {sorted(LABEL_ALIASES)}"
            )
        if block_id in labels:
            raise ValueError(f"Duplicate label for block_id={block_id}")
        labels[block_id] = LABEL_ALIASES[raw_label]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def _page_result_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.name != "page_result.json":
            raise ValueError("Input file must be named page_result.json")
        return [source]
    if not source.is_dir():
        raise ValueError(f"Input does not exist: {source}")
    paths = sorted(source.rglob("page_result.json"))
    if not paths:
        raise ValueError(f"No page_result.json found below: {source}")
    return paths


def _document_id(page_id: str) -> str:
    match = re.match(r"^(.*)_p\d+$", page_id)
    return match.group(1) if match else page_id


def load_heading_examples(
    source: str | Path,
    labels: str | Path | None = None,
    *,
    require_labels: bool = False,
) -> list[HeadingExample]:
    """读取 page_result；有标签文件时仅返回已标注块，并检查悬空标签。"""
    source_path = Path(source)
    label_map = _read_labels(Path(labels)) if labels is not None else {}
    if require_labels and not label_map:
        raise ValueError("Training requires a non-empty labels JSONL file")
    examples: list[HeadingExample] = []
    used_labels: set[str] = set()
    seen_block_ids: set[str] = set()
    for path in _page_result_paths(source_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        page = payload.get("page") or {}
        page_id = str(page.get("page_id") or "")
        routes = payload.get("routes") or []
        route_by_region = {}
        for route in routes:
            route_id = route.get("block_id")
            if isinstance(route_id, list) and len(route_id) == 2:
                route_by_region[int(route_id[1])] = route
        for position, block in enumerate(page.get("blocks") or []):
            order = block.get("reading_order", position)
            route = route_by_region.get(int(order), {})
            metadata = block.get("metadata") or {}
            context = metadata.get("context") or route.get("context") or {}
            native_structures = metadata.get("native_structures", metadata.get("native_structure", []))
            if isinstance(native_structures, dict):
                native_structures = [native_structures]
            if not isinstance(native_structures, list):
                native_structures = []
            block_id = str(block.get("block_id") or "")
            if not block_id:
                raise ValueError(f"{path}: block without block_id")
            if block_id in seen_block_ids:
                raise ValueError(f"Duplicate block_id across input page results: {block_id}")
            seen_block_ids.add(block_id)
            label = label_map.get(block_id)
            if label_map and label is None:
                continue
            if label is not None:
                used_labels.add(block_id)
            examples.append(HeadingExample(
                block_id=block_id,
                page_id=str(block.get("page_id") or page_id),
                document_id=_document_id(str(block.get("page_id") or page_id)),
                text=str(block.get("text") or route.get("result") or "").strip(),
                context=context,
                raw_label=str(metadata.get("raw_label") or block.get("block_type") or ""),
                reading_order=int(order) if order is not None else None,
                source_page_result=str(path),
                label=label,
                native_structures=[item for item in native_structures if isinstance(item, dict)],
            ))
    missing = sorted(set(label_map) - used_labels)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Labels reference unknown block_id values: {preview}")
    if require_labels and not examples:
        raise ValueError("No labeled heading examples found")
    return examples


def format_model_input(example: HeadingExample) -> str:
    """当前块置首，避免 tokenizer 右侧截断把被分类文本吃掉。"""
    previous = (example.context.get("previous") or {}).get("text") or ""
    following = (example.context.get("next") or {}).get("text") or ""
    return (
        f"[当前块]\n{example.text}\n"
        f"[前一块]\n{previous}\n"
        f"[后一块]\n{following}"
    )


def split_examples_by_document(
    examples: Sequence[HeadingExample], *, val_ratio: float = 0.2, seed: int = 42,
) -> tuple[list[HeadingExample], list[HeadingExample]]:
    """文档级隔离，防止同篇文档的相邻 crop 同时泄漏到训练集和验证集。"""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    documents = sorted({example.document_id for example in examples})
    if not documents:
        raise ValueError("No examples to split")
    if val_ratio == 0 or len(documents) == 1:
        return list(examples), []
    rng = random.Random(seed)
    rng.shuffle(documents)
    val_count = min(len(documents) - 1, max(1, round(len(documents) * val_ratio)))
    val_documents = set(documents[:val_count])
    train = [item for item in examples if item.document_id not in val_documents]
    val = [item for item in examples if item.document_id in val_documents]
    if not train or not val:
        raise ValueError("Document split produced an empty train or validation set")
    return train, val


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Metrics require equally sized, non-empty labels and predictions")
    tp = sum(int(y == 1 and p == 1) for y, p in zip(labels, predictions))
    tn = sum(int(y == 0 and p == 0) for y, p in zip(labels, predictions))
    fp = sum(int(y == 0 and p == 1) for y, p in zip(labels, predictions))
    fn = sum(int(y == 1 and p == 0) for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "count": len(labels), "accuracy": (tp + tn) / len(labels),
        "heading_precision": precision, "heading_recall": recall,
        "heading_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _ml_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer training/inference requires torch and transformers; "
            "install the deep-learning dependencies in a separate environment"
        ) from exc
    return torch, AutoTokenizer, AutoModelForSequenceClassification


def _resolve_device(torch, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _predict_probabilities(
    model, tokenizer, examples: Sequence[HeadingExample], *,
    batch_size: int, max_length: int, device: str, torch,
) -> list[list[float]]:
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            encoded = tokenizer(
                [format_model_input(item) for item in batch],
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    return probabilities


def train_transformer(
    train_examples: Sequence[HeadingExample],
    val_examples: Sequence[HeadingExample],
    *,
    base_model: str,
    output: str | Path,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
    max_grad_norm: float = 1.0,
    device: str | None = None,
) -> dict[str, Any]:
    """微调预训练编码器；每次训练必须使用新目录，避免混合实验产物。"""
    if not train_examples:
        raise ValueError("Training set is empty")
    if epochs < 1 or batch_size < 1 or max_length < 16:
        raise ValueError("epochs/batch_size must be positive and max_length at least 16")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a positive finite number")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be a positive finite number")
    if any(item.label not in LABEL_TO_ID for item in [*train_examples, *val_examples]):
        raise ValueError("Every training and validation example must have a binary label")
    train_label_counts = Counter(item.label for item in train_examples)
    if set(train_label_counts) != set(LABELS):
        raise ValueError("Training set must contain both body and heading examples")
    output_path = Path(output)
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise ValueError("Training output directory must be empty; use a new experiment directory")

    torch, AutoTokenizer, AutoModelForSequenceClassification = _ml_dependencies()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = _resolve_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
    )
    max_positions = getattr(model.config, "max_position_embeddings", max_length)
    if max_length > max_positions:
        raise ValueError(f"max_length={max_length} exceeds model limit {max_positions}")
    model.to(resolved_device)

    def collate(batch: Sequence[HeadingExample]):
        encoded = tokenizer(
            [format_model_input(item) for item in batch],
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [LABEL_TO_ID[item.label] for item in batch], dtype=torch.long,
        )
        return encoded

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        list(train_examples), batch_size=batch_size, shuffle=True,
        collate_fn=collate, generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total = sum(train_label_counts.values())
    class_weights = torch.tensor(
        [total / (2 * train_label_counts[label]) for label in LABELS],
        dtype=torch.float32, device=resolved_device,
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in loader:
            batch = {key: value.to(resolved_device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            seen += len(labels)
        train_probabilities = _predict_probabilities(
            model, tokenizer, train_examples,
            batch_size=batch_size, max_length=max_length,
            device=resolved_device, torch=torch,
        )
        train_predictions = [max(range(2), key=lambda index: row[index]) for row in train_probabilities]
        epoch_result = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train": binary_metrics(
                [LABEL_TO_ID[item.label] for item in train_examples], train_predictions,
            ),
        }
        if val_examples:
            val_probabilities = _predict_probabilities(
                model, tokenizer, val_examples,
                batch_size=batch_size, max_length=max_length,
                device=resolved_device, torch=torch,
            )
            val_predictions = [max(range(2), key=lambda index: row[index]) for row in val_probabilities]
            epoch_result["validation"] = binary_metrics(
                [LABEL_TO_ID[item.label] for item in val_examples], val_predictions,
            )
        history.append(epoch_result)

    output_path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    summary = {
        "schema_version": "heading-classifier-training.v1",
        "task": "binary_heading_body",
        "labels": list(LABELS),
        "base_model": base_model,
        "device": resolved_device,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "max_length": max_length,
            "seed": seed, "max_grad_norm": max_grad_norm,
        },
        "train_document_ids": sorted({item.document_id for item in train_examples}),
        "validation_document_ids": sorted({item.document_id for item in val_examples}),
        "train_label_counts": dict(train_label_counts),
        "validation_label_counts": dict(Counter(item.label for item in val_examples)),
        "class_weights": {label: float(class_weights[index].cpu()) for index, label in enumerate(LABELS)},
        "history": history,
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary


class TransformerHeadingClassifier:
    """加载已微调目录；没有训练产物时不提供伪造的默认预测。"""

    def __init__(
        self, model_dir: str | Path, *, device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise ValueError(f"Trained model directory does not exist: {self.model_dir}")
        torch, AutoTokenizer, AutoModelForSequenceClassification = _ml_dependencies()
        self._torch = torch
        self.device = _resolve_device(torch, device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device).eval()
        summary_path = self.model_dir / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        self.max_length = max_length or int(summary.get("hyperparameters", {}).get("max_length", 256))

    def predict(
        self, examples: Sequence[HeadingExample], *, batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        probabilities = _predict_probabilities(
            self.model, self.tokenizer, examples,
            batch_size=batch_size, max_length=self.max_length,
            device=self.device, torch=self._torch,
        )
        results = []
        for example, scores in zip(examples, probabilities):
            label_id = max(range(2), key=lambda index: scores[index])
            results.append({
                "block_id": example.block_id,
                "page_id": example.page_id,
                "document_id": example.document_id,
                "reading_order": example.reading_order,
                "label": ID_TO_LABEL[label_id],
                "confidence": scores[label_id],
                "scores": {label: scores[index] for index, label in ID_TO_LABEL.items()},
                "raw_label": example.raw_label,
                "source_page_result": example.source_page_result,
                "task": "binary_heading_body",
            })
        return results
