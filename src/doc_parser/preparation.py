"""Prepare multi-format documents as auditable page packages for single-page parsing."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import zipfile


SCHEMA_VERSION = "page-package.v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
MAX_ARCHIVE_MEMBERS = 10000
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_format(path: str | Path) -> str:
    """Identify supported formats from content where practical, extension otherwise."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    with source.open("rb") as stream:
        head = stream.read(16)
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"BM", b"II*\x00", b"MM\x00*", b"RIFF")):
        return "image"
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            names = set(archive.namelist())
        if "word/document.xml" in names:
            return "docx"
        if "ppt/presentation.xml" in names:
            return "pptx"
        if "xl/workbook.xml" in names:
            return "xlsx"
        return "zip"
    if suffix in {".tar", ".tgz", ".gz", ".bz2", ".xz"}:
        if tarfile.is_tarfile(source):
            return "tar"
    if suffix in {".7z", ".rar", ".iso", ".ofd"}:
        return suffix[1:]
    if suffix in {".doc", ".ppt", ".xls"}:
        return suffix[1:]
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".json":
        return "json"
    if suffix in {".txt", ".csv", ".md", ".xml", ".html", ".htm"}:
        return "text"
    raise ValueError(f"Unsupported input format: {source.suffix or '(no extension)'}")


def _extract_docx(path: Path) -> dict:
    from docx import Document

    document = Document(path)
    sections = []
    for index, section in enumerate(document.sections):
        sections.append({
            "section": index + 1,
            "header": [p.text for p in section.header.paragraphs if p.text.strip()],
            "footer": [p.text for p in section.footer.paragraphs if p.text.strip()],
        })
    return {
        "format": "docx",
        "paragraphs": [{"index": i, "style": p.style.name, "text": p.text}
                       for i, p in enumerate(document.paragraphs) if p.text.strip()],
        "tables": [[[[p.text for p in cell.paragraphs] for cell in row.cells]
                    for row in table.rows] for table in document.tables],
        "sections": sections,
        "images": [part.partname for part in document.part.related_parts.values()
                   if "image/" in str(part.content_type)],
    }


def _extract_pptx(path: Path) -> dict:
    from pptx import Presentation

    presentation = Presentation(path)
    slides = []
    for index, slide in enumerate(presentation.slides):
        shapes = []
        for shape_index, shape in enumerate(slide.shapes):
            item = {"index": shape_index, "type": str(shape.shape_type), "name": shape.name}
            if getattr(shape, "has_text_frame", False):
                item["text"] = shape.text
            if getattr(shape, "has_table", False):
                item["table"] = [[cell.text for cell in row.cells] for row in shape.table.rows]
            shapes.append(item)
        slides.append({"slide": index + 1, "shapes": shapes})
    return {"format": "pptx", "slides": slides}


def _extract_xlsx(path: Path) -> dict:
    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=False, read_only=False)
    sheets = []
    for sheet in workbook.worksheets:
        rows = []
        for row in sheet.iter_rows():
            values = [cell.value for cell in row]
            if any(value is not None for value in values):
                rows.append(values)
        sheets.append({"name": sheet.title, "state": sheet.sheet_state,
                       "max_row": sheet.max_row, "max_column": sheet.max_column,
                       "merged_cells": [str(rng) for rng in sheet.merged_cells.ranges],
                       "rows": rows})
    workbook.close()
    return {"format": "xlsx", "sheets": sheets}


def _extract_native(path: Path, kind: str) -> dict:
    if kind == "docx":
        return _extract_docx(path)
    if kind == "pptx":
        return _extract_pptx(path)
    if kind == "xlsx":
        return _extract_xlsx(path)
    if kind == "json":
        return {"format": "json", "value": json.loads(path.read_text(encoding="utf-8-sig"))}
    if kind == "text":
        return {"format": "text", "text": path.read_text(encoding="utf-8-sig")}
    if kind in {"doc", "ppt", "xls"}:
        return {"format": kind, "extraction": "office_to_pdf_text_layer"}
    return {"format": kind}


def _render_pdf(path: Path, pages_dir: Path, dpi: int, max_pages: int | None = None) -> list[dict]:
    import pypdfium2 as pdfium
    from PIL import Image

    if not 72 <= dpi <= 400:
        raise ValueError("dpi must be between 72 and 400")
    pages_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    rendered = []
    with pdfium.PdfDocument(str(path)) as document:
        if max_pages is not None and len(document) > max_pages:
            raise ValueError(f"Document page count exceeds configured limit of {max_pages}")
        for index in range(len(document)):
            page = document[index]
            target = pages_dir / f"{index + 1:04d}.png"
            try:
                bitmap = page.render(scale=scale)
                try:
                    with bitmap.to_pil() as image:
                        image.convert("RGB").save(target)
                        width, height = image.size
                finally:
                    bitmap.close()
            finally:
                page.close()
            rendered.append({"page_no": index + 1, "path": target, "width": width, "height": height})
    if not rendered:
        raise ValueError(f"PDF has no pages: {path}")
    return rendered


def _office_visible_blocks(pdf_path: Path, page_no: int, scale: float) -> list[dict]:
    """Extract text boxes from a PDF generated locally from an Office source."""
    import fitz

    with fitz.open(pdf_path) as document:
        if page_no >= len(document):
            return []
        page = document[page_no]
        result = []
        for block_index, block in enumerate(page.get_text("blocks")):
            text = str(block[4]).strip()
            if not text:
                continue
            result.append({"block_id": f"rendered_text_p{page_no + 1:04d}_{block_index:04d}",
                           "block_type": "native_visible_text", "text": text,
                           "bbox": [round(float(block[0]) * scale, 2), round(float(block[1]) * scale, 2),
                                    round(float(block[2]) * scale, 2), round(float(block[3]) * scale, 2)],
                           "source": {"engine": "office_render_pdf_text", "path": f"page:{page_no + 1}"}})
        return result


def _render_office(source: Path, staging: Path, converter: str | None) -> Path:
    executable = converter or os.environ.get("DOCUMENT_OFFICE_CONVERTER") or shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise RuntimeError("Office rendering requires LibreOffice; set DOCUMENT_OFFICE_CONVERTER to its executable")
    staging.mkdir(parents=True, exist_ok=True)
    # 并发转换必须隔离 LibreOffice 用户配置目录，否则多个 soffice 会争抢同一实例。
    profile = (staging / "libreoffice-profile").resolve()
    command = [
        executable, f"-env:UserInstallation={profile.as_uri()}", "--headless",
        "--convert-to", "pdf", "--outdir", str(staging), str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    target = staging / f"{source.stem}.pdf"
    if completed.returncode or not target.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Office to PDF conversion failed for {source.name}: {detail or completed.returncode}")
    return target


def _render_image(path: Path, pages_dir: Path, max_pages: int | None = None) -> list[dict]:
    from PIL import Image, ImageSequence

    pages_dir.mkdir(parents=True, exist_ok=True)
    result = []
    with Image.open(path) as source:
        for index, frame in enumerate(ImageSequence.Iterator(source)):
            if max_pages is not None and index >= max_pages:
                raise ValueError(f"Document page count exceeds configured limit of {max_pages}")
            target = pages_dir / f"{index + 1:04d}.png"
            image = frame.convert("RGB")
            image.save(target)
            result.append({"page_no": index + 1, "path": target, "width": image.width, "height": image.height})
    return result


def _safe_archive_name(name: str) -> Path:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or any(part in {"..", ""} for part in pure.parts):
        raise ValueError(f"Unsafe archive member: {name}")
    return Path(*pure.parts)


def _extract_archive(path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = [item for item in archive.infolist() if not item.is_dir()]
            if len(entries) > MAX_ARCHIVE_MEMBERS or sum(item.file_size for item in entries) > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds extraction limits")
            for item in entries:
                mode = item.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    raise ValueError(f"Archive contains a symbolic link: {item.filename}")
                target = destination / _safe_archive_name(item.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return [destination / _safe_archive_name(item.filename) for item in entries]
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            entries = [item for item in archive.getmembers() if item.isfile()]
            if len(entries) > MAX_ARCHIVE_MEMBERS or sum(item.size for item in entries) > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds extraction limits")
            for item in entries:
                target = destination / _safe_archive_name(item.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                src = archive.extractfile(item)
                if src is None:
                    continue
                with src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return [destination / _safe_archive_name(item.name) for item in entries]
    raise RuntimeError(f"Archive type {path.suffix.lower()} needs an external extractor")


def prepare_document(source: str | Path, output: str | Path, *, dpi: int = 150,
                     office_converter: str | None = None, max_pages: int | None = None,
                     _depth: int = 0, _page_budget: list[int] | None = None) -> dict:
    """Create native artifacts and stable per-page packages without running models."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if _depth > 4:
        raise ValueError("Archive nesting exceeds four levels")
    if _page_budget is None and max_pages is not None:
        _page_budget = [max_pages]
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new job directory")
    output.mkdir(parents=True, exist_ok=True)
    kind = detect_format(source)
    manifest = {"schema_version": SCHEMA_VERSION, "source_name": source.name,
                "source_sha256": _sha256(source), "detected_format": kind,
                "mime_type": mimetypes.guess_type(source.name)[0], "status": "success",
                "page_count": 0, "pages": [], "children": []}
    try:
        if kind in {"zip", "tar"}:
            extracted = _extract_archive(source, output / "children" / "_extracted")
            for index, child in enumerate(extracted, 1):
                if child.suffix.lower() in {".ini", ".db", ".ds_store"}:
                    continue
                child_output = output / "children" / f"{index:04d}-{child.stem}"
                try:
                    child_manifest = prepare_document(child, child_output, dpi=dpi,
                                                      office_converter=office_converter, max_pages=max_pages,
                                                      _depth=_depth + 1, _page_budget=_page_budget)
                except (ValueError, RuntimeError) as exc:
                    child_manifest = {"source_name": child.name, "status": "unsupported", "error": str(exc)}
                    _write_json(child_output / "manifest.json", child_manifest)
                manifest["children"].append({"path": child_output.relative_to(output).as_posix(),
                                             "source_name": child.name, "status": child_manifest["status"]})
        elif kind in {"txt", "text", "json"}:
            native = _extract_native(source, kind)
            _write_json(output / "native" / "objects.json", native)
            manifest["native_only"] = True
            manifest["native_path"] = "native/objects.json"
        else:
            native = _extract_native(source, kind)
            _write_json(output / "native" / "objects.json", native)
            pdf_for_pages = source
            office_generated = False
            with tempfile.TemporaryDirectory(prefix="document-orchestrator-") as temp_dir:
                if kind in {"docx", "pptx", "xlsx", "doc", "ppt", "xls"}:
                    pdf_for_pages = _render_office(source, Path(temp_dir), office_converter)
                    office_generated = True
                if kind == "pdf" or office_generated or kind in {"doc", "ppt", "xls"}:
                    rendered = _render_pdf(pdf_for_pages, output / "pages", dpi,
                                           max_pages=_page_budget[0] if _page_budget is not None else None)
                    if _page_budget is not None:
                        _page_budget[0] -= len(rendered)
                    scale = dpi / 72.0
                    for page in rendered:
                        native_blocks = (_office_visible_blocks(pdf_for_pages, page["page_no"] - 1, scale)
                                         if office_generated else [])
                        page_no = page["page_no"]
                        page_id = f"p{page_no:04d}"
                        relative_image = page["path"].relative_to(output).as_posix()
                        package = {"schema_version": SCHEMA_VERSION, "document_id": source.stem,
                                   "page_id": page_id, "page_no": page_no,
                                   "image": {"path": relative_image, "width": page["width"],
                                             "height": page["height"], "dpi": dpi,
                                             "coordinate_system": "pixel_top_left"},
                                   "native_blocks": native_blocks,
                                   "visual_tasks": [{"task_id": f"layout_{page_id}", "task_type": "page_layout",
                                                     "bbox": [0, 0, page["width"], page["height"]],
                                                     "preferred_engine": "paddleocr_vl"}],
                                   "policy": {"native_text_preferred": True,
                                              "native_skip_requires_bbox_overlap": 0.9,
                                              "pdf_text_layer_policy": "ignore" if kind == "pdf" else None,
                                              "visual_fallback": True}}
                        page_dir = output / "page_packages" / f"{page_no:04d}"
                        _write_json(page_dir / "page.json", package)
                        manifest["pages"].append({"page_no": page_no, "page_id": page_id,
                                                  "image": relative_image,
                                                  "package": (page_dir / "page.json").relative_to(output).as_posix(),
                                                  "native_block_count": len(native_blocks)})
                elif kind == "image":
                    rendered = _render_image(source, output / "pages",
                                             max_pages=_page_budget[0] if _page_budget is not None else None)
                    if _page_budget is not None:
                        _page_budget[0] -= len(rendered)
                    for page in rendered:
                        page_no = page["page_no"]
                        page_id = f"p{page_no:04d}"
                        relative_image = page["path"].relative_to(output).as_posix()
                        package = {"schema_version": SCHEMA_VERSION, "document_id": source.stem,
                                   "page_id": page_id, "page_no": page_no,
                                   "image": {"path": relative_image, "width": page["width"],
                                             "height": page["height"], "dpi": None,
                                             "coordinate_system": "pixel_top_left"},
                                   "native_blocks": [],
                                   "visual_tasks": [{"task_id": f"layout_{page_id}", "task_type": "page_layout",
                                                     "bbox": [0, 0, page["width"], page["height"]],
                                                     "preferred_engine": "paddleocr_vl"}],
                                   "policy": {"native_text_preferred": False, "visual_fallback": True}}
                        page_dir = output / "page_packages" / f"{page_no:04d}"
                        _write_json(page_dir / "page.json", package)
                        manifest["pages"].append({"page_no": page_no, "page_id": page_id,
                                                  "image": relative_image,
                                                  "package": (page_dir / "page.json").relative_to(output).as_posix(),
                                                  "native_block_count": 0})
                elif kind not in {"pdf", "docx", "pptx", "xlsx", "doc", "ppt", "xls"}:
                    raise RuntimeError(f"Format {kind} has no configured page renderer")
            manifest["native_path"] = "native/objects.json"
        manifest["page_count"] = len(manifest["pages"])
        _write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        _write_json(output / "manifest.json", manifest)
        raise

