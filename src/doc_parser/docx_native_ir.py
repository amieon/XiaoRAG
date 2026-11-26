"""DOCX 原生段落直读为 DocumentIR，并把标题预测回写到新文件。

该通道不运行 OCR／版面模型；native_id 对每个非空正文段落保留，供样式回写使用。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from chapter_tree.heading_classifier import HEADING_LABEL, HeadingExample

from .docx_numbering import DocxNumbering
from .document_ir import Block, DocumentIR, Page, SourceRef
from .preparation import _docx_native_heading


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_text(table: Any) -> str:
    return "\n".join(
        "\t".join("\n".join(p.text for p in cell.paragraphs).strip() for cell in row.cells)
        for row in table.rows
    ).strip()


def native_docx_to_ir(source: str | Path) -> DocumentIR:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    path = Path(source).resolve()
    document = Document(path)
    numbering = DocxNumbering(document)
    paragraph_indices = {paragraph._p: index for index, paragraph in enumerate(document.paragraphs)}
    digest = _source_sha256(path)
    document_id = "doc_" + digest[:16]
    page_id = document_id + "_p0001"
    blocks: list[Block] = []
    paragraph_count = table_count = 0

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            index = paragraph_indices.get(item._p)
            text = item.text.strip()
            if index is None or not text:
                continue
            native_id = f"word/document.xml/body/paragraph/{index}"
            heading = _docx_native_heading(item, native_id)
            list_structure = numbering.list_structure(item, native_id)
            native_structures = [heading] if heading is not None else []
            list_structures = [list_structure] if list_structure is not None else []
            block_type = "paragraph"
            if heading is not None:
                block_type = "document_title" if heading["role"] == "document_title" else "section_heading"
            blocks.append(Block(
                block_id=f"{page_id}_b{len(blocks):06d}",
                page_id=page_id,
                block_type=block_type,
                bbox=[0.0, 0.0, 1.0, 1.0],
                text=text,
                reading_order=len(blocks),
                confidence=1.0,
                source=SourceRef(
                    engine="docx_native", raw_page=1, raw_block_id=native_id,
                    raw_path="word/document.xml", extra={"paragraph_index": index},
                ),
                metadata={
                    "native_id": native_id,
                    "paragraph_index": index,
                    "style": item.style.name if item.style is not None else "",
                    "raw_label": "docx_paragraph",
                    "native_structures": native_structures,
                    "list_structures": list_structures,
                    "coordinate_system": "docx_native_order",
                },
            ))
            paragraph_count += 1
        elif isinstance(item, Table):
            table_index = table_count
            table_count += 1
            text = _table_text(item)
            if not text:
                continue
            native_id = f"word/document.xml/body/table/{table_index}"
            blocks.append(Block(
                block_id=f"{page_id}_b{len(blocks):06d}",
                page_id=page_id,
                block_type="table",
                bbox=[0.0, 0.0, 1.0, 1.0],
                text=text,
                reading_order=len(blocks),
                confidence=1.0,
                source=SourceRef(
                    engine="docx_native", raw_page=1, raw_block_id=native_id,
                    raw_path="word/document.xml", extra={"table_index": table_index},
                ),
                metadata={
                    "native_id": native_id, "table_index": table_index,
                    "raw_label": "docx_table", "native_structures": [],
                    "list_structures": [], "coordinate_system": "docx_native_order",
                },
            ))

    # 上下文只在原生阅读顺序中构造，表格保留在 IR，但不交给标题模型分类。
    for index, block in enumerate(blocks):
        previous = next((blocks[pos] for pos in range(index - 1, -1, -1) if blocks[pos].text.strip()), None)
        following = next((blocks[pos] for pos in range(index + 1, len(blocks)) if blocks[pos].text.strip()), None)
        block.metadata["context"] = {
            "previous": None if previous is None else {
                "block_id": previous.block_id, "text": previous.text[-300:],
                "char_start": max(0, len(previous.text) - 300), "char_end": len(previous.text),
                "truncated": len(previous.text) > 300,
            },
            "next": None if following is None else {
                "block_id": following.block_id, "text": following.text[:300],
                "char_start": 0, "char_end": min(300, len(following.text)),
                "truncated": len(following.text) > 300,
            },
        }

    page = Page(
        page_id=page_id, page_no=1, width=1.0, height=1.0, blocks=blocks,
        source=SourceRef(engine="docx_native", raw_page=1, raw_path=str(path)),
        metadata={"native_only": True, "logical_page": True, "reading_order": "docx_xml"},
    )
    result = DocumentIR(
        document_id=document_id, pages=[page], source_path=str(path), content_hash=digest,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        title=path.stem,
        metadata={
            "pipeline": "docx-native-heading-dataset.v1", "native_only": True,
            "paragraph_count": paragraph_count, "table_count": table_count,
        },
    )
    result.validate()
    return result


def heading_examples_from_native_ir(document: DocumentIR) -> list[HeadingExample]:
    examples = []
    for page in sorted(document.pages, key=lambda item: item.page_no):
        for block in sorted(page.blocks, key=lambda item: item.reading_order or 0):
            if block.block_type not in {"paragraph", "section_heading", "document_title"}:
                continue
            examples.append(HeadingExample(
                block_id=block.block_id, page_id=block.page_id, document_id=document.document_id,
                text=block.text, context=block.metadata.get("context") or {},
                raw_label=str(block.metadata.get("raw_label") or block.block_type),
                reading_order=block.reading_order,
                source_page_result=str(document.source_path or "native_docx"),
                native_structures=list(block.metadata.get("native_structures") or []),
            ))
    return examples


def _heading_style(document: Any, level: int) -> Any:
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    name = f"Heading {level}"
    try:
        return document.styles[name]
    except KeyError:
        style = document.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = document.styles["Normal"]
        style.font.bold = True
        properties = style.element.get_or_add_pPr()
        outline = OxmlElement("w:outlineLvl")
        outline.set(qn("w:val"), str(level - 1))
        properties.append(outline)
        return style


def write_styled_docx(
    source: str | Path,
    output: str | Path,
    document_ir: DocumentIR,
    predictions: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """按块的原生段落下标回写；绝不按重复标题文本猜段落。"""
    from docx import Document

    blocks = {
        block.block_id: block
        for page in document_ir.pages for block in page.blocks
    }
    seen = set()
    assignments = []
    for row in predictions:
        block_id = row.get("block_id")
        if block_id in seen:
            raise ValueError(f"duplicate heading prediction: {block_id}")
        seen.add(block_id)
        if block_id not in blocks:
            raise ValueError(f"prediction references unknown block: {block_id}")
        if row.get("label") != HEADING_LABEL:
            continue
        level = row.get("heading_level")
        if type(level) is not int or not 1 <= level <= 6:
            raise ValueError(f"invalid heading level for {block_id}")
        block = blocks[block_id]
        paragraph_index = block.metadata.get("paragraph_index")
        if type(paragraph_index) is not int:
            raise ValueError(f"heading block has no DOCX paragraph mapping: {block_id}")
        assignments.append((paragraph_index, level, row))

    source_document = Document(Path(source))
    changed = 0
    model_changed = 0
    for paragraph_index, level, row in assignments:
        if paragraph_index >= len(source_document.paragraphs):
            raise ValueError(f"DOCX paragraph index out of range: {paragraph_index}")
        paragraph = source_document.paragraphs[paragraph_index]
        if paragraph.text.strip() != blocks[row["block_id"]].text.strip():
            raise ValueError(f"DOCX source text changed at paragraph {paragraph_index}")
        if not (paragraph.style is not None and paragraph.style.name == f"Heading {level}"):
            changed += 1
        paragraph.style = _heading_style(source_document, level)
        if row.get("decision_source") == "zero_shot_llm":
            model_changed += 1
    target = Path(output)
    if target.exists():
        raise ValueError(f"Styled DOCX already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_document.save(target)
    return {
        "heading_count": len(assignments),
        "style_assignment_count": changed,
        "model_heading_count": model_changed,
    }
