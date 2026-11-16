"""File/folder orchestration for the explicit public-model DocumentIR pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from .document_ir import Block, DocumentIR, Page, SourceRef
from .document_orchestration import prepare_document
from .explicit_pipeline import (
    ExplicitParserConfig,
    PIPELINE_VERSION,
    PageParseResult,
    render_markdown,
)


BATCH_SCHEMA_VERSION = "document-folder-batch.v1"
IGNORED_DIRECTORIES = {".git", ".svn", ".venv", "__pycache__", "node_modules"}
IGNORED_FILES = {".ds_store", "thumbs.db", "desktop.ini"}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _collect_pages(root: Path) -> list[tuple[Path, Path, dict]]:
    items = []
    for package_path in sorted(root.rglob("page.json")):
        package = _read(package_path)
        package_root = package_path.parents[2]
        image_path = package_root / package["image"]["path"]
        items.append((package_path, image_path, package))
    return items


def _native_text(native: dict) -> str:
    kind = native.get("format")
    if kind == "text":
        return str(native.get("text", ""))
    if kind == "json":
        return "```json\n" + json.dumps(native.get("value"), ensure_ascii=False, indent=2, default=str) + "\n```"
    if kind == "docx":
        chunks = [item.get("text", "") for item in native.get("paragraphs", [])]
        for table in native.get("tables", []):
            chunks.extend("\t".join("\n".join(cell) for cell in row) for row in table)
        return "\n\n".join(item for item in chunks if item)
    if kind == "pptx":
        chunks = []
        for slide in native.get("slides", []):
            for shape in slide.get("shapes", []):
                if shape.get("text"):
                    chunks.append(shape["text"])
                for row in shape.get("table", []) or []:
                    chunks.append("\t".join(row))
        return "\n\n".join(chunks)
    if kind == "xlsx":
        chunks = []
        for sheet in native.get("sheets", []):
            rows = ["\t".join("" if value is None else str(value) for value in row)
                    for row in sheet.get("rows", [])]
            chunks.append(f"## {sheet.get('name', 'Sheet')}\n" + "\n".join(rows))
        return "\n\n".join(chunks)
    return ""


def _native_pages(
    prepared_root: Path,
    *,
    document_id: str,
    source_path: str,
    starting_page_no: int,
) -> list[Page]:
    """Turn native-only children into real IR pages instead of an empty shell."""
    pages = []
    for objects_path in sorted(prepared_root.rglob("native/objects.json")):
        package_root = objects_path.parent.parent
        manifest_path = package_root / "manifest.json"
        if manifest_path.is_file() and _read(manifest_path).get("page_count", 0):
            continue
        native = _read(objects_path)
        text = _native_text(native).strip()
        if not text:
            continue
        page_no = starting_page_no + len(pages)
        page_id = f"{document_id}_p{page_no:04d}"
        block_id = f"{page_id}_b0000"
        relative_raw = objects_path.relative_to(prepared_root).as_posix()
        pages.append(Page(
            page_id=page_id,
            page_no=page_no,
            width=1.0,
            height=1.0,
            blocks=[Block(
                block_id=block_id,
                page_id=page_id,
                block_type="paragraph",
                bbox=[0.0, 0.0, 1.0, 1.0],
                text=text,
                reading_order=0,
                confidence=1.0,
                source=SourceRef(
                    engine="native_extraction",
                    raw_page=page_no,
                    raw_block_id="0",
                    raw_path=relative_raw,
                ),
                metadata={"native_format": native.get("format"), "coordinate_system": "native_text"},
            )],
            source=SourceRef(engine="native_extraction", raw_page=page_no, raw_path=relative_raw),
            metadata={"native_only": True, "source_document_id": package_root.name},
        ))
    return pages


def _rebase_markdown(text: str, prefix: str) -> str:
    def target(value: str) -> str:
        if value.startswith(("/", "#")) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", value):
            return value
        return prefix + "/" + value.removeprefix("./")
    text = re.sub(r"(!?\[[^\]\n]*\]\()([^\s)]+)", lambda match: match[1] + target(match[2]) + ")", text)
    return re.sub(r"((?:src|href)=[\"\x27])([^\"\x27]+)", lambda match: match[1] + target(match[2]), text)


def _blank_package_page(
    package: dict,
    *,
    document_id: str,
    page_no: int,
    source_path: str,
    image_ref: str,
) -> Page:
    page_id = f"{document_id}_p{page_no:04d}"
    return Page(
        page_id=page_id,
        page_no=page_no,
        width=float(package["image"]["width"]),
        height=float(package["image"]["height"]),
        source=SourceRef(engine="prepare_only", raw_page=page_no, raw_path=source_path),
        image_ref=image_ref,
        metadata={
            "prepared_only": True,
            "source_document_id": package.get("document_id"),
            "source_page_no": package.get("page_no"),
        },
    )


def run_document(
    input_path: str | Path,
    output: str | Path,
    *,
    config: ExplicitParserConfig | None = None,
    dpi: int = 150,
    office_converter: str | None = None,
    prepare_only: bool = False,
    runtime: Any = None,
    runtime_factory: Callable[[], Any] | None = None,
    close_runtime: bool = True,
    max_pages: int | None = None,
) -> dict:
    """Prepare one file/archive and optionally parse every page into one IR."""
    config = config or ExplicitParserConfig()
    source, output_root = Path(input_path).resolve(), Path(output).resolve()
    prepared = prepare_document(
        source, output_root, dpi=dpi, office_converter=office_converter,
        max_pages=max_pages,
    )
    document_id = "doc_" + prepared["source_sha256"][:16]
    page_inputs = _collect_pages(output_root)
    results_root = output_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    pages: list[Page] = []
    all_routes: list[dict] = []
    page_summaries: list[dict] = []
    markdown_parts: list[str] = []
    owns_runtime = False

    if page_inputs and not prepare_only:
        if runtime is None:
            if runtime_factory is None:
                from .public_models import ExplicitParserRuntime
                runtime_factory = lambda: ExplicitParserRuntime(
                    Path(__file__).resolve().parents[2], config,
                )
            runtime = runtime_factory()
            owns_runtime = True
    try:
        for global_page, (package_path, image_path, package) in enumerate(page_inputs, start=1):
            local_page = int(package["page_no"])
            package_root = package_path.parents[2]
            image_ref = image_path.relative_to(output_root).as_posix()
            if prepare_only:
                page = _blank_package_page(
                    package, document_id=document_id, page_no=global_page,
                    source_path=str(source), image_ref=image_ref,
                )
                result = PageParseResult(page=page, records=[], markdown="", layout={})
            else:
                result_dir = package_root / "results" / "pages" / f"{local_page:04d}"
                result = runtime.parse_page(
                    image_path,
                    document_id=document_id,
                    page_no=global_page,
                    source_path=str(source),
                    image_ref=image_ref,
                    native_blocks=package.get("native_blocks", []),
                    output_dir=result_dir,
                )
                result.page.metadata.update({
                    "source_document_id": package.get("document_id"),
                    "source_page_no": local_page,
                })
                for route in result.records:
                    route["source_document_id"] = package.get("document_id")
                    route["source_page_no"] = local_page
                relative_result = os.path.relpath(result_dir, results_root).replace(os.sep, "/")
                markdown_parts.append(
                    f"<!-- Document {package['document_id']}, page {local_page} -->\n\n" +
                    _rebase_markdown(result.markdown, relative_result)
                )
            pages.append(result.page)
            all_routes.extend(result.records)
            page_summaries.append({
                "page": global_page,
                "page_id": result.page.page_id,
                "document_id": package.get("document_id"),
                "source_page_no": local_page,
                "package": package_path.relative_to(output_root).as_posix(),
                "image": image_ref,
                "route_count": len(result.records),
                "prepared_only": prepare_only,
            })
    finally:
        if owns_runtime and close_runtime and runtime is not None:
            runtime.close()

    native_pages = _native_pages(
        output_root, document_id=document_id, source_path=str(source),
        starting_page_no=len(pages) + 1,
    )
    for page in native_pages:
        pages.append(page)
        markdown_parts.append(page.blocks[0].text if page.blocks else "")
        page_summaries.append({
            "page": page.page_no, "page_id": page.page_id,
            "document_id": None, "source_page_no": page.page_no,
            "route_count": 0, "native_only": True,
        })
    document = DocumentIR(
        document_id=document_id,
        pages=pages,
        source_path=str(source),
        content_hash=prepared["source_sha256"],
        mime_type=prepared.get("mime_type"),
        title=source.stem,
        metadata={
            "pipeline": PIPELINE_VERSION,
            "prepared_manifest": "manifest.json",
            "prepare_only": prepare_only,
            "config": config.to_dict(),
        },
    )
    document.validate()
    document.to_json(results_root / "document.json")
    (results_root / "document.md").write_text(
        "\n\n---\n\n".join(part for part in markdown_parts if part), encoding="utf-8",
    )
    with (results_root / "routes.jsonl").open("w", encoding="utf-8") as stream:
        for route in all_routes:
            stream.write(json.dumps(route, ensure_ascii=False, default=str) + "\n")
    engines = {}
    for route in all_routes:
        engine = route.get("engine", "unknown")
        engines[engine] = engines.get(engine, 0) + 1
    manifest = {
        "status": "success",
        "mode": "explicit_folder_document_ir",
        "pipeline": PIPELINE_VERSION,
        "input": str(source),
        "document_id": document_id,
        "page_count": len(pages),
        "visual_page_count": len(page_inputs),
        "native_page_count": len(native_pages),
        "route_count": len(all_routes),
        "engines": engines,
        "prepare_only": prepare_only,
        "prepared_manifest": "manifest.json",
        "document_ir": "results/document.json",
        "pages": page_summaries,
    }
    _write_json(results_root / "manifest.json", manifest)
    return manifest


def discover_folder_files(input_root: str | Path, output_root: str | Path) -> list[Path]:
    """Recursively enumerate files without following directory symlinks."""
    root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current).resolve()
        kept = []
        for name in sorted(directories):
            candidate = current_path / name
            resolved = candidate.resolve()
            if name in IGNORED_DIRECTORIES or resolved == output or output.is_relative_to(resolved):
                continue
            if candidate.is_symlink():
                continue
            kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = current_path / name
            if name.lower() in IGNORED_FILES or path.is_symlink() or not path.is_file():
                continue
            if path.resolve().is_relative_to(output):
                continue
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().casefold())


def run_folder(
    input_root: str | Path,
    output_root: str | Path,
    *,
    config: ExplicitParserConfig | None = None,
    dpi: int = 150,
    office_converter: str | None = None,
    prepare_only: bool = False,
    runtime: Any = None,
    runtime_factory: Callable[[], Any] | None = None,
    max_pages_per_document: int | None = None,
) -> dict:
    """Process every file, isolate failures, and reuse one resident runtime."""
    config = config or ExplicitParserConfig()
    source_root, output = Path(input_root).resolve(), Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Batch output directory must be empty; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    files = discover_folder_files(source_root, output)
    if not files:
        raise ValueError(f"Input folder contains no files: {source_root}")

    shared_runtime = runtime
    runtime_error: Exception | None = None

    def shared_factory() -> Any:
        nonlocal shared_runtime, runtime_error
        if shared_runtime is not None:
            return shared_runtime
        if runtime_error is not None:
            raise RuntimeError(f"Model runtime initialization already failed: {runtime_error}")
        try:
            if runtime_factory is not None:
                shared_runtime = runtime_factory()
            else:
                from .public_models import ExplicitParserRuntime
                shared_runtime = ExplicitParserRuntime(Path(__file__).resolve().parents[2], config)
            return shared_runtime
        except Exception as exc:
            runtime_error = exc
            raise

    summaries: list[dict] = []
    try:
        for source in files:
            relative = source.relative_to(source_root)
            job_output = output / "documents" / relative
            summary: dict[str, Any] = {
                "source": relative.as_posix(),
                "output": job_output.relative_to(output).as_posix(),
            }
            try:
                result = run_document(
                    source,
                    job_output,
                    config=config,
                    dpi=dpi,
                    office_converter=office_converter,
                    prepare_only=prepare_only,
                    runtime=shared_runtime,
                    runtime_factory=shared_factory,
                    close_runtime=False,
                    max_pages=max_pages_per_document,
                )
                summary.update(result)
            except Exception as exc:
                unsupported = isinstance(exc, ValueError) and str(exc).startswith("Unsupported input format:")
                summary.update({
                    "status": "unsupported" if unsupported else "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                _write_json(job_output / "job-error.json", summary)
            summaries.append(summary)
    finally:
        if runtime is None and shared_runtime is not None:
            shared_runtime.close()

    counts = {
        status: sum(item.get("status") == status for item in summaries)
        for status in ("success", "failed", "unsupported")
    }
    status = "success" if counts["failed"] == 0 else "partial" if counts["success"] else "failed"
    manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "status": status,
        "pipeline": PIPELINE_VERSION,
        "input_root": str(source_root),
        "file_count": len(files),
        "counts": counts,
        "prepare_only": prepare_only,
        "config": config.to_dict(),
        "files": summaries,
    }
    _write_json(output / "batch-manifest.json", manifest)
    return manifest


def run_path(
    input_path: str | Path,
    output: str | Path,
    **kwargs,
) -> dict:
    """Dispatch a single file or a directory without changing the file API."""
    source = Path(input_path)
    if source.is_dir():
        return run_folder(source, output, **kwargs)
    if source.is_file():
        if "max_pages_per_document" in kwargs:
            kwargs["max_pages"] = kwargs.pop("max_pages_per_document")
        return run_document(source, output, **kwargs)
    raise FileNotFoundError(source)
