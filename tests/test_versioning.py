from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from versioning import publish_version


class RecordingEmbedder:
    """显式小向量只验证版本数据流，不代表真实模型效果。"""

    def __init__(self, model_id="tiny-version-vectors"):
        self.model_id = model_id
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            digest = hashlib.sha256((self.model_id + "\0" + text).encode("utf-8")).digest()
            rows.append([digest[0] / 255, digest[1] / 255, digest[2] / 255, 1.0])
        return np.asarray(rows, dtype=np.float32)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_export(root: Path, name: str, parent_specs, document_id="source-doc") -> Path:
    """parent_specs: [(title_path, [child_text, ...]), ...]"""
    output = root / name
    output.mkdir(parents=True)
    parents, units = [], []
    for parent_index, (title_path, texts) in enumerate(parent_specs):
        parent_id = f"chapter_tree:{document_id}:p{parent_index}"
        parents.append({
            "schema_version": "retrieval-parent.v1", "parent_id": parent_id,
            "document_id": document_id, "text": "完整父块", "body_text": "完整父块",
            "title_path": title_path, "provenance": {"producer": "chapter_tree"},
        })
        for child_index, text in enumerate(texts):
            units.append({
                "schema_version": "retrieval-unit.v1",
                "unit_id": f"chapter_tree:{document_id}:p{parent_index}:c{child_index}",
                "document_id": document_id, "parent_id": parent_id,
                "embedding_text": text, "body_text": text, "title_path": title_path,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "provenance": {"producer": "chapter_tree"},
            })
    (output / "retrieval-units.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in units), encoding="utf-8",
    )
    (output / "retrieval-parents.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in parents), encoding="utf-8",
    )
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": "retrieval-preparation.v1", "document_id": document_id,
        "producer": "chapter_tree", "unit_count": len(units), "parent_count": len(parents),
    }), encoding="utf-8")
    return output


def by_text(units):
    return {unit["embedding_text"]: unit for unit in units}


def version_dir(store: Path, revision: int) -> Path:
    current = json.loads((store / "documents").glob("doc_*/current.json").__next__().read_text(encoding="utf-8"))
    return store / "documents" / current["document_key"] / "versions" / f"r{revision:06d}"


def test_first_publish_unchanged_and_changed_revision_with_pointer_reuse(tmp_path):
    source = tmp_path / "受管文档.txt"
    source.write_text("原始文件版本零", encoding="utf-8")
    store = tmp_path / "store"
    export0 = write_export(tmp_path, "export0", [(["第一章"], ["保持不变", "即将修改"])])
    embedder = RecordingEmbedder()

    created = publish_version(source, export0, store, embedder=embedder)
    assert created["status"] == "created" and created["revision"] == 0
    assert created["newly_encoded_text_count"] == 2
    assert created["reused_from_previous_count"] == 0
    assert created["snapshot_complete"] is True and created["diff_persisted"] is False
    root0 = version_dir(store, 0)
    assert (root0 / "source" / "original.txt").read_bytes() == source.read_bytes()
    old = by_text(read_jsonl(root0 / "retrieval-units.jsonl"))
    current = json.loads((root0.parent.parent / "current.json").read_text(encoding="utf-8"))
    assert current == {
        "document_key": created["document_key"], "revision": 0, "version_id": created["version_id"],
    }
    calls_after_first = len(embedder.calls)

    unchanged = publish_version(source, export0, store, embedder=embedder)
    assert unchanged["status"] == "unchanged" and unchanged["revision"] == 0
    assert len(embedder.calls) == calls_after_first
    assert not (root0.parent / "r000001").exists()

    source.write_text("原始文件版本一", encoding="utf-8")
    export1 = write_export(tmp_path, "export1", [(["第一章"], ["保持不变", "已经修改", "新增正文"])])
    changed = publish_version(source, export1, store, embedder=embedder)
    assert changed["revision"] == 1 and changed["previous_version_id"] == created["version_id"]
    assert changed["reused_from_previous_count"] == 1
    assert changed["newly_encoded_text_count"] == 2
    root1 = version_dir(store, 1)
    new = by_text(read_jsonl(root1 / "retrieval-units.jsonl"))
    assert new["保持不变"]["vector_ref"] == old["保持不变"]["vector_ref"]
    assert new["已经修改"]["vector_ref"]["vector_id"] != old["即将修改"]["vector_ref"]["vector_id"]
    assert new["新增正文"]["vector_ref"]["vector_id"] not in {
        old["保持不变"]["vector_ref"]["vector_id"], old["即将修改"]["vector_ref"]["vector_id"],
    }
    assert (root1 / "source" / "original.txt").read_text(encoding="utf-8") == "原始文件版本一"
    for unit in new.values():
        vector = np.load(store / unit["vector_ref"]["path"], allow_pickle=False)
        assert vector.shape == (4,) and vector.dtype == np.float32


def test_same_text_reuses_vector_after_parent_move(tmp_path):
    source = tmp_path / "move.txt"
    source.write_text("v0", encoding="utf-8")
    store = tmp_path / "store"
    export0 = write_export(tmp_path, "move0", [(["旧章节"], ["相同正文"]), (["其他"], ["旧值"])])
    embedder = RecordingEmbedder()
    publish_version(source, export0, store, embedder=embedder)
    old = by_text(read_jsonl(version_dir(store, 0) / "retrieval-units.jsonl"))

    source.write_text("v1", encoding="utf-8")
    export1 = write_export(tmp_path, "move1", [(["旧章节"], ["新值"]), (["新章节"], ["相同正文"])])
    result = publish_version(source, export1, store, embedder=embedder)
    new = by_text(read_jsonl(version_dir(store, 1) / "retrieval-units.jsonl"))
    assert result["reused_from_previous_count"] == 1
    assert new["相同正文"]["vector_ref"] == old["相同正文"]["vector_ref"]
    assert new["相同正文"]["parent_key"] != old["相同正文"]["parent_key"]
    assert new["相同正文"]["unit_key"] != old["相同正文"]["unit_key"]


def test_same_text_with_different_model_does_not_reuse(tmp_path):
    source = tmp_path / "model.txt"
    source.write_text("v0", encoding="utf-8")
    store = tmp_path / "store"
    export = write_export(tmp_path, "model0", [(["章节"], ["正文不变"])])
    publish_version(source, export, store, embedder=RecordingEmbedder("model-a"))
    old = read_jsonl(version_dir(store, 0) / "retrieval-units.jsonl")[0]
    source.write_text("v1", encoding="utf-8")
    changed_embedder = RecordingEmbedder("model-b")
    result = publish_version(source, export, store, embedder=changed_embedder)
    new = read_jsonl(version_dir(store, 1) / "retrieval-units.jsonl")[0]
    assert result["reused_from_previous_count"] == 0 and result["newly_encoded_text_count"] == 1
    assert new["vector_ref"]["vector_id"] != old["vector_ref"]["vector_id"]
    assert new["vector_ref"]["embedding_model"] == "model-b"
    assert len(changed_embedder.calls) == 1


def test_duplicate_text_multiple_parents_and_empty_title_path(tmp_path):
    source = tmp_path / "duplicates.txt"
    source.write_text("v0", encoding="utf-8")
    export = write_export(tmp_path, "duplicates", [
        ([], ["相同正文", "相同正文"]), (["重复标题"], ["尾"]), (["重复标题"], ["相同正文"]),
    ])
    result = publish_version(source, export, tmp_path / "store", embedder=RecordingEmbedder())
    assert result["newly_encoded_text_count"] == 2
    units = read_jsonl(version_dir(tmp_path / "store", 0) / "retrieval-units.jsonl")
    parents = read_jsonl(version_dir(tmp_path / "store", 0) / "retrieval-parents.jsonl")
    assert len({unit["vector_ref"]["vector_id"] for unit in units}) == 2
    assert units[0]["unit_key"].endswith("child:000000")
    assert units[1]["unit_key"].endswith("child:000001")
    assert parents[0]["parent_key"] == "chapter_tree:<root>#occurrence:0"
    assert parents[1]["parent_key"].endswith("#occurrence:0")
    assert parents[2]["parent_key"].endswith("#occurrence:1")


def test_global_vector_pool_reuses_across_documents(tmp_path):
    store = tmp_path / "store"
    source0, source1 = tmp_path / "a.txt", tmp_path / "b.txt"
    source0.write_text("a", encoding="utf-8")
    source1.write_text("b", encoding="utf-8")
    export0 = write_export(tmp_path, "global0", [(["章节"], ["共同", "甲"])])
    export1 = write_export(tmp_path, "global1", [(["章节"], ["共同", "乙"])])
    publish_version(source0, export0, store, embedder=RecordingEmbedder())
    second = RecordingEmbedder()
    result = publish_version(source1, export1, store, embedder=second)
    assert result["reused_from_global_count"] == 1
    assert result["newly_encoded_text_count"] == 1
    assert second.calls == [["乙"]]
    assert len(list((store / "vectors").glob("*.npy"))) == 3


def test_dangling_parent_and_hash_mismatch_fail_before_version_commit(tmp_path):
    source = tmp_path / "bad.txt"
    source.write_text("bad", encoding="utf-8")
    export = write_export(tmp_path, "dangling", [(["章节"], ["正文"])])
    unit_path = export / "retrieval-units.jsonl"
    unit = json.loads(unit_path.read_text(encoding="utf-8"))
    unit["parent_id"] = "missing"
    unit_path.write_text(json.dumps(unit), encoding="utf-8")
    with pytest.raises(ValueError, match="dangling_parent"):
        publish_version(source, export, tmp_path / "store", embedder=RecordingEmbedder())
    assert not (tmp_path / "store" / "documents").exists() or not list((tmp_path / "store" / "documents").glob("doc_*/current.json"))

    export2 = write_export(tmp_path, "bad-hash", [(["章节"], ["正文"])])
    unit_path = export2 / "retrieval-units.jsonl"
    unit = json.loads(unit_path.read_text(encoding="utf-8"))
    unit["content_hash"] = "0" * 64
    unit_path.write_text(json.dumps(unit), encoding="utf-8")
    embedder = RecordingEmbedder()
    with pytest.raises(ValueError, match="content_hash_mismatch"):
        publish_version(source, export2, tmp_path / "store2", embedder=embedder)
    assert embedder.calls == []
    assert not list((tmp_path / "store2" / "vectors").glob("*.npy"))


@pytest.mark.parametrize("matrix", [
    np.asarray([1.0, 2.0], dtype=np.float32),
    np.asarray([[1.0, np.nan]], dtype=np.float32),
    np.empty((0, 2), dtype=np.float32),
])
def test_invalid_embedding_matrix_fails_without_version(tmp_path, matrix):
    class BadEmbedder(RecordingEmbedder):
        def encode(self, texts):
            return matrix

    source = tmp_path / "matrix.txt"
    source.write_text("bad", encoding="utf-8")
    export = write_export(tmp_path, "matrix", [(["章节"], ["正文"])])
    store = tmp_path / "store"
    with pytest.raises(ValueError, match="invalid_embedding_matrix"):
        publish_version(source, export, store, embedder=BadEmbedder())
    assert not list(store.glob("documents/doc_*/current.json"))
    assert not list(store.glob("documents/doc_*/versions/r*"))


def test_publish_cli_with_explicit_test_embedding_module(tmp_path):
    source = tmp_path / "cli.txt"
    source.write_text("CLI 原文", encoding="utf-8")
    export = write_export(tmp_path, "cli-export", [(["章节"], ["CLI 正文"])])
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    (fake_modules / "sentence_transformers.py").write_text(
        "import numpy as np\n"
        "class SentenceTransformer:\n"
        "    def __init__(self, model_id, device='cpu'): self.model_id = model_id\n"
        "    def encode(self, texts, **kwargs):\n"
        "        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(fake_modules) + os.pathsep + str(ROOT / "src")
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "publish_document_version.py"),
        "--source", str(source), "--input", str(export), "--store", str(tmp_path / "store"),
        "--model", "cli-explicit-test-model", "--device", "cpu", "--batch-size", "2",
    ], cwd=ROOT, env=environment, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result["status"] == "created" and result["revision"] == 0
    assert result["newly_encoded_text_count"] == 1
    assert result["embedding_model"] == "cli-explicit-test-model"
