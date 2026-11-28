"""Lossless segmentation of native-only text blocks for bounded model inputs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

from .document_ir import Block, DocumentIR, Page

DEFAULT_NATIVE_BLOCK_CHARS = 6000


def split_native_text(text: str, max_chars: int = DEFAULT_NATIVE_BLOCK_CHARS) -> list[tuple[int, int]]:
    """Pack whole lines up to max_chars; hard-split only an individually long line."""
    if max_chars < 256:
        raise ValueError("native block max_chars must be at least 256")
    if not text:
        return []
    pieces = []
    for line in text.splitlines(keepends=True):
        # Pretty-printed JSON and ordinary text normally never enter this loop.
        # A minified JSON string or an exceptional long line still needs a hard cap.
        while len(line) > max_chars:
            pieces.append(line[:max_chars])
            line = line[max_chars:]
        if line:
            pieces.append(line)
    spans = []
    start = cursor = length = 0
    for piece in pieces:
        if length and length + len(piece) > max_chars:
            spans.append((start, cursor))
            start, length = cursor, 0
        cursor += len(piece)
        length += len(piece)
    if cursor > start:
        spans.append((start, cursor))
    return spans


def segment_native_page(page: Page, max_chars: int = DEFAULT_NATIVE_BLOCK_CHARS) -> tuple[Page, bool]:
    """Split only oversized blocks; concatenating segment texts is byte-identical."""
    changed = any(len(block.text) > max_chars for block in page.blocks)
    if not changed:
        return page, False
    blocks: list[Block] = []
    for block in page.blocks:
        spans = split_native_text(block.text, max_chars)
        if len(spans) <= 1:
            blocks.append(block)
            continue
        original_id = block.block_id
        for index, (start, end) in enumerate(spans):
            segment = deepcopy(block)
            segment.block_id = original_id if index == 0 else f"{original_id}_s{index:04d}"
            segment.text = block.text[start:end]
            segment.metadata.update({
                "native_source_block_id": original_id,
                "native_segment_strategy": "newline_pack_v2",
                "native_segment_index": index,
                "native_segment_count": len(spans),
                "native_char_start": start,
                "native_char_end": end,
            })
            if segment.source:
                segment.source.raw_block_id = f"{block.source.raw_block_id or original_id}:{index}"
            blocks.append(segment)
    for order, block in enumerate(blocks):
        block.reading_order = order
    page.blocks = blocks
    page.validate()
    return page, True


def _backup(path: Path, raw: bytes) -> None:
    backup = path.with_name(path.name.replace(".json", ".pre-reflow.json"))
    if not backup.exists():
        backup.write_bytes(raw)


def reflow_native_documents(
    source: str | Path,
    *,
    predictions: str | Path | None = None,
    max_chars: int = DEFAULT_NATIVE_BLOCK_CHARS,
) -> dict:
    """Migrate old native IR in place with backups; visual OCR documents are untouched."""
    source = Path(source)
    paths = [source] if source.is_file() else sorted(source.rglob("document.json"))
    changed_ids = set()
    changed_files = []
    for path in paths:
        if not path.is_file() or path.name != "document.json":
            continue
        raw = path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
            pages = payload.get("pages") or []
            if not pages or not all((page.get("metadata") or {}).get("native_only") is True for page in pages):
                continue
            if not any(
                len(block.get("text", "")) > max_chars
                for page in pages for block in page.get("blocks", [])
            ):
                continue
            document = DocumentIR.from_dict(payload)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"[native-reflow] 跳过无效 IR：{path}: {exc}", file=sys.stderr)
            continue
        changed = False
        for page in document.pages:
            _, page_changed = segment_native_page(page, max_chars)
            changed = changed or page_changed
        if not changed:
            continue
        document.validate()
        _backup(path, raw)
        temporary = path.with_name("document.reflow-tmp.json")
        temporary.write_text(
            json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(path)
        changed_ids.add(document.document_id)
        changed_files.append(str(path))
        print(f"[native-reflow] {document.document_id}: {path}", file=sys.stderr, flush=True)

    removed_prediction_count = 0
    prediction_path = Path(predictions) if predictions is not None else None
    if prediction_path is not None and prediction_path.is_file() and changed_ids:
        raw = prediction_path.read_bytes()
        rows = []
        for line in raw.decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("document_id") in changed_ids:
                removed_prediction_count += 1
            else:
                rows.append(row)
        _backup(prediction_path, raw)
        prediction_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
        )
    return {
        "status": "success", "changed_document_count": len(changed_ids),
        "changed_document_ids": sorted(changed_ids), "changed_files": changed_files,
        "removed_stale_prediction_count": removed_prediction_count,
        "max_chars": max_chars,
    }
