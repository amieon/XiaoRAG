"""最简线性版本管理：路径定文档、章节路径定 parent、内容哈希复用向量。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from filelock import FileLock
import numpy as np

from hybrid_retrieval.index import read_jsonl


VERSION_SCHEMA = "document-knowledge-version.v1"
VECTOR_STORE_VERSION = "local-content-addressed-npy-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records) -> None:
    path.write_text("".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records
    ), encoding="utf-8")


def _document_key(source: Path) -> str:
    # 默认路径即逻辑文档身份；移动／改名需要后续显式换绑，不能靠内容猜测。
    normalized = os.path.normcase(str(source.resolve()))
    return "doc_" + _sha256_bytes(normalized.encode("utf-8"))[:16]


def _current(document_root: Path) -> dict[str, Any] | None:
    path = document_root / "current.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None


def _version_dir(document_root: Path, revision: int) -> Path:
    return document_root / "versions" / f"r{revision:06d}"


def _read_export(export_dir: Path):
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != "retrieval-preparation.v1":
        raise ValueError("versioning_requires_retrieval-preparation.v1")
    units = read_jsonl(export_dir / "retrieval-units.jsonl")
    parents = read_jsonl(export_dir / "retrieval-parents.jsonl")
    if not units or not parents or len(units) != manifest.get("unit_count") or len(parents) != manifest.get("parent_count"):
        raise ValueError("versioning_export_count_or_content_mismatch")
    if any(unit.get("schema_version") != "retrieval-unit.v1" for unit in units):
        raise ValueError("versioning_requires_retrieval-unit.v1")
    if any(parent.get("schema_version") != "retrieval-parent.v1" for parent in parents):
        raise ValueError("versioning_requires_retrieval-parent.v1")
    document_id = manifest.get("document_id")
    if any(unit.get("document_id") != document_id for unit in units) or any(
        parent.get("document_id") != document_id for parent in parents
    ):
        raise ValueError("versioning_export_document_mismatch")
    unit_ids = [unit.get("unit_id") for unit in units]
    parent_ids = [parent.get("parent_id") for parent in parents]
    if any(not isinstance(value, str) or not value for value in unit_ids + parent_ids):
        raise ValueError("versioning_export_ids_require_nonempty_strings")
    if len(set(unit_ids)) != len(unit_ids) or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("versioning_export_duplicate_id")
    for unit in units:
        text = unit.get("embedding_text")
        if not isinstance(text, str) or not text:
            raise ValueError("versioning_unit_embedding_text_required")
        if unit.get("content_hash") != _sha256_bytes(text.encode("utf-8")):
            raise ValueError("versioning_unit_content_hash_mismatch")
    return manifest, units, parents


def _vector_id(model_id: str, content_hash: str) -> str:
    return "vec_" + _sha256_bytes((model_id + "\0" + content_hash).encode("utf-8"))


def _vector_ref(store: Path, model_id: str, content_hash: str) -> dict[str, str]:
    vector_id = _vector_id(model_id, content_hash)
    return {
        "backend": "local_npy", "store_version": VECTOR_STORE_VERSION,
        "embedding_model": model_id, "vector_id": vector_id,
        "path": f"vectors/{vector_id}.npy",
    }


def _previous_reuse(previous_units: list[dict], store: Path, model_id: str):
    """只按内容哈希复用；不要求新旧 child 的位置或 unit_key 相同。"""
    result = {}
    for unit in previous_units:
        content_hash = unit.get("content_hash")
        ref = unit.get("vector_ref") or {}
        if not isinstance(content_hash, str) or ref.get("embedding_model") != model_id:
            continue
        expected = _vector_ref(store, model_id, content_hash)
        if all(ref.get(key) == value for key, value in expected.items()) and (
            store / expected["path"]
        ).is_file():
            # 保留旧引用对象的全部字段，确保未变 child 的 vector_ref 完全不变。
            result[content_hash] = ref
    return result


def _versioned_records(document_key, revision, units, parents, vector_refs):
    """parent_key 使用章节路径＋同路径出现序号；child 仅保留 parent 内普通序号。"""
    parent_versions = {}
    parent_keys = {}
    occurrences = defaultdict(int)
    versioned_parents = []
    for index, parent in enumerate(parents):
        producer = (parent.get("provenance") or {}).get("producer") or "unknown"
        title_path = parent.get("title_path") or []
        if not isinstance(title_path, list) or any(not isinstance(item, str) for item in title_path):
            raise ValueError("versioning_parent_title_path_requires_strings")
        structural_path = " > ".join(title_path) if title_path else "<root>"
        occurrence_key = (producer, structural_path)
        occurrence = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        parent_key = f"{producer}:{structural_path}#occurrence:{occurrence}"
        parent_version_id = f"{document_key}:r{revision:06d}:p{index:06d}"
        parent_versions[parent["parent_id"]] = parent_version_id
        parent_keys[parent["parent_id"]] = parent_key
        versioned_parents.append({
            **parent, "source_document_id": parent["document_id"], "document_id": document_key,
            "source_parent_id": parent["parent_id"], "parent_id": parent_version_id,
            "parent_key": parent_key, "revision": revision,
        })

    child_ordinals = defaultdict(int)
    versioned_units = []
    for index, unit in enumerate(units):
        source_parent = unit.get("parent_id")
        if source_parent not in parent_versions:
            raise ValueError("versioning_dangling_parent_id")
        expected_hash = _sha256_bytes(unit["embedding_text"].encode("utf-8"))
        if unit.get("content_hash") != expected_hash:
            raise ValueError("versioning_unit_content_hash_mismatch")
        parent_key = parent_keys[source_parent]
        ordinal = child_ordinals[source_parent]
        child_ordinals[source_parent] += 1
        unit_version_id = f"{document_key}:r{revision:06d}:c{index:06d}"
        versioned_units.append({
            **unit, "source_document_id": unit["document_id"], "document_id": document_key,
            "source_unit_id": unit["unit_id"], "unit_id": unit_version_id,
            "source_parent_id": source_parent, "parent_id": parent_versions[source_parent],
            "parent_key": parent_key, "unit_key": f"{parent_key}/child:{ordinal:06d}",
            "revision": revision, "vector_ref": vector_refs[unit["content_hash"]],
        })
    if set(child_ordinals) != set(parent_versions):
        raise ValueError("versioning_parent_without_children")
    return versioned_units, versioned_parents


def publish_version(
    source: str | Path, export_dir: str | Path, store: str | Path, *, embedder,
) -> dict[str, Any]:
    """发布线性新版本；完整保存快照，不生成或保存 diff。"""
    source, export_dir, store = Path(source).resolve(), Path(export_dir).resolve(), Path(store).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    model_id = getattr(embedder, "model_id", None)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("versioning_embedder_model_id_required")
    source_sha256 = _sha256_file(source)
    export_manifest, units, parents = _read_export(export_dir)
    document_key = _document_key(source)
    document_root = store / "documents" / document_key
    document_root.mkdir(parents=True, exist_ok=True)
    (store / "vectors").mkdir(parents=True, exist_ok=True)

    with FileLock(str(document_root / "version.lock")):
        current = _current(document_root)
        versions_root = document_root / "versions"
        existing_versions = list(versions_root.glob("r[0-9][0-9][0-9][0-9][0-9][0-9]")) if versions_root.is_dir() else []
        if current is None and existing_versions:
            raise ValueError("versioning_unpublished_versions_require_repair")
        if current is not None:
            current_manifest = json.loads((_version_dir(document_root, current["revision"]) / "manifest.json").read_text(encoding="utf-8-sig"))
            if current_manifest["source_sha256"] == source_sha256:
                return {**current_manifest, "status": "unchanged"}
            revision = int(current["revision"]) + 1
            previous_units = read_jsonl(_version_dir(document_root, current["revision"]) / "retrieval-units.jsonl")
        else:
            revision = 0
            previous_units = []
        if revision >= 1_000_000:
            raise ValueError("versioning_revision_limit_reached")

        reusable = _previous_reuse(previous_units, store, model_id)
        vector_refs = {}
        reused_previous = reused_global = 0
        texts_to_encode, hashes_to_encode = [], []
        global_hashes = set()
        pending_hashes = set()
        for unit in units:
            content_hash = unit["content_hash"]
            if content_hash in reusable:
                vector_refs[content_hash] = reusable[content_hash]
                reused_previous += 1
                continue
            if content_hash in global_hashes:
                # 同版重复正文共享一个向量，但复用计数按 child 数统计。
                reused_global += 1
                continue
            if content_hash in pending_hashes:
                continue
            ref = _vector_ref(store, model_id, content_hash)
            if (store / ref["path"]).is_file():
                vector_refs[content_hash] = ref
                global_hashes.add(content_hash)
                reused_global += 1
            else:
                texts_to_encode.append(unit["embedding_text"])
                hashes_to_encode.append(content_hash)
                pending_hashes.add(content_hash)

        if texts_to_encode:
            encoded = np.asarray(embedder.encode(texts_to_encode), dtype=np.float32)
            if encoded.ndim != 2 or encoded.shape[0] != len(texts_to_encode) or encoded.shape[1] < 1 or not np.isfinite(encoded).all():
                raise ValueError("versioning_invalid_embedding_matrix")
            for content_hash, text, vector in zip(hashes_to_encode, texts_to_encode, encoded):
                ref = _vector_ref(store, model_id, content_hash)
                target = store / ref["path"]
                # 内容寻址记录不可变；已存在则复用，不写第二份。
                if not target.exists():
                    temporary = target.with_name(f".{target.stem}-{uuid4().hex}.tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, vector, allow_pickle=False)
                    os.replace(temporary, target)
                vector_refs[content_hash] = ref

        versioned_units, versioned_parents = _versioned_records(
            document_key, revision, units, parents, vector_refs,
        )
        version_id = f"{document_key}:r{revision:06d}:{source_sha256[:16]}"
        manifest = {
            "schema_version": VERSION_SCHEMA, "status": "created",
            "document_key": document_key, "source_path": str(source), "source_sha256": source_sha256,
            "revision": revision, "version_id": version_id,
            "previous_version_id": current["version_id"] if current else None,
            "snapshot_complete": True, "diff_persisted": False,
            "chunking": export_manifest.get("chunking"),
            "unit_count": len(versioned_units), "parent_count": len(versioned_parents),
            "embedding_model": model_id, "vector_store": VECTOR_STORE_VERSION,
            "reused_from_previous_count": reused_previous,
            "reused_from_global_count": reused_global,
            "newly_encoded_text_count": len(texts_to_encode),
        }
        staging = versions_root / f".staging-{uuid4().hex}"
        final = _version_dir(document_root, revision)
        try:
            staging.mkdir(parents=True, exist_ok=False)
            source_target = staging / "source" / ("original" + source.suffix.lower())
            source_target.parent.mkdir(parents=True)
            shutil.copyfile(source, source_target)
            _write_jsonl(staging / "retrieval-units.jsonl", versioned_units)
            _write_jsonl(staging / "retrieval-parents.jsonl", versioned_parents)
            _write_json(staging / "manifest.json", manifest)
            os.replace(staging, final)
            current_tmp = document_root / f".current-{uuid4().hex}.tmp"
            _write_json(current_tmp, {
                "document_key": document_key, "revision": revision, "version_id": version_id,
            })
            os.replace(current_tmp, document_root / "current.json")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return manifest
