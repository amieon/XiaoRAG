from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.document_ir import Block, Page
from doc_parser.native_text import segment_native_page, split_native_text


def test_native_text_segments_are_contiguous_and_lossless():
    text = ("第一段。\n" * 1000) + "末尾"
    spans = split_native_text(text, 1000)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert "".join(text[start:end] for start, end in spans) == text
    assert all(end - start <= 1000 for start, end in spans)
    assert all(text[end - 1] == "\n" for _, end in spans[:-1])
    assert all(text[start:end].splitlines(keepends=True)[-1].endswith("\n") for start, end in spans[:-1])


def test_native_page_reflow_preserves_text_and_source_spans():
    text = "甲" * 7000
    page = Page(
        page_id="doc_p0001", page_no=1, width=1, height=1,
        blocks=[Block(
            block_id="doc_p0001_b0000", page_id="doc_p0001", block_type="paragraph",
            bbox=[0, 0, 1, 1], text=text, reading_order=0,
        )],
        metadata={"native_only": True},
    )
    page, changed = segment_native_page(page)
    assert changed and len(page.blocks) == 2
    assert "".join(block.text for block in page.blocks) == text
    assert page.blocks[0].block_id == "doc_p0001_b0000"
    assert page.blocks[1].block_id == "doc_p0001_b0000_s0001"
    assert page.blocks[1].metadata["native_char_start"] == 6000
    assert [block.reading_order for block in page.blocks] == [0, 1]


def test_existing_native_ir_and_stale_predictions_are_migrated_with_backups(tmp_path):
    results = tmp_path / "doc/results"
    results.mkdir(parents=True)
    document = {
        "document_id": "doc_long", "pages": [{
            "page_id": "doc_long_p0001", "page_no": 1, "width": 1, "height": 1,
            "metadata": {"native_only": True},
            "blocks": [{
                "block_id": "doc_long_p0001_b0000", "page_id": "doc_long_p0001",
                "block_type": "paragraph", "bbox": [0, 0, 1, 1],
                "text": "乙" * 6500, "reading_order": 0, "metadata": {},
            }],
        }],
    }
    path = results / "document.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"document_id": "doc_long", "block_id": "doc_long_p0001_b0000", "label": "body"},
        {"document_id": "doc_other", "block_id": "other", "label": "body"},
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "reflow_native_ir.py"),
        "--input", str(tmp_path), "--predictions", str(predictions),
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    summary = json.loads(completed.stdout)
    assert summary["changed_document_count"] == 1
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert len(migrated["pages"][0]["blocks"]) == 2
    assert "".join(block["text"] for block in migrated["pages"][0]["blocks"]) == "乙" * 6500
    assert (results / "document.pre-reflow.json").is_file()
    remaining = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert [row["document_id"] for row in remaining] == ["doc_other"]
    assert (tmp_path / "predictions.pre-reflow.jsonl").is_file()
