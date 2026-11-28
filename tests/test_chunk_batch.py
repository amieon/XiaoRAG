from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_document(root: Path, name: str, text: str) -> str:
    document_id = f"doc-{name}"
    directory = root / name / "results"
    directory.mkdir(parents=True)
    (directory / "page_result.json").write_text("[]", encoding="utf-8")
    (directory / "document.json").write_text(json.dumps({
        "document_id": document_id, "schema_version": "0.1",
        "pages": [{
            "page_id": "p0", "page_no": 1, "width": 100, "height": 100,
            "blocks": [{
                "block_id": "b0", "page_id": "p0", "block_type": "text", "text": text,
                "reading_order": 0, "bbox": [0, 0, 10, 10], "metadata": {},
            }],
        }],
    }, ensure_ascii=False), encoding="utf-8")
    return document_id


def test_python_batch_continues_after_failure_and_resumes_completed_documents(tmp_path):
    source = tmp_path / "输入"
    first = write_document(source, "甲", "第一份正文。")
    write_document(source, "乙", "缺少预测。")
    third = write_document(source, "丙", "")
    derived = source / "派生DocumentIR"
    derived.mkdir(parents=True)
    derived_ir = json.loads((source / "甲" / "results" / "document.json").read_text(encoding="utf-8"))
    derived_ir["document_id"] = "doc-丁"
    (derived / "document.json").write_text(json.dumps(derived_ir, ensure_ascii=False), encoding="utf-8")
    predictions = tmp_path / "预测.jsonl"
    rows = [
        {"document_id": first, "block_id": "b0", "label": "body"},
        {"document_id": third, "block_id": "b0", "label": "body"},
        {"document_id": "doc-丁", "block_id": "b0", "label": "body"},
    ]
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    output = tmp_path / "输出"
    command = [
        sys.executable, str(ROOT / "scripts" / "build_chunks_batch.py"),
        "--input", str(source), "--predictions", str(predictions),
        "--output-root", str(output), "--mode", "hybrid",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    manifest = json.loads((output / "batch-manifest.json").read_text(encoding="utf-8"))
    assert manifest["document_count"] == 4
    assert manifest["success_count"] == 3 and manifest["failure_count"] == 1
    assert manifest["status"] == "failed"
    assert [row["document"] for row in manifest["documents"] if row["status"] == "failed"] == ["乙"]
    assert (output / "甲" / "chunks.jsonl").is_file()
    assert (output / "丙" / "chunks.jsonl").is_file()
    assert (output / "派生DocumentIR" / "chunks.jsonl").is_file()
    assert not (output / "乙" / "chunks.jsonl").exists()

    rows.append({"document_id": "doc-乙", "block_id": "b0", "label": "body"})
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    resumed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    manifest = json.loads((output / "batch-manifest.json").read_text(encoding="utf-8"))
    assert manifest["success_count"] == 1 and manifest["skipped_count"] == 3
    assert manifest["failure_count"] == 0 and manifest["status"] == "success"
