"""Natural Office units from parsed IR/native objects; no source-file access."""
from __future__ import annotations

from copy import deepcopy
import re


def _chars(value):
    return len(value)


def _format(ir, native, manifest):
    """Only declared format fields participate; names and corpus labels do not."""
    source = ir.get("source") or {}
    mime_formats = {
        "application/vnd.ms-powerpoint": "ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/vnd.ms-excel": "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "text/csv": "csv",
    }
    for record, keys in ((ir, ("source_format", "format", "mime_type")), (native, ("format",)),
                         (manifest, ("detected_format", "original_format", "format", "mime_type")),
                         (source, ("format",))):
        for key in keys:
            value = record.get(key)
            if isinstance(value, str):
                normalized = value.lower().lstrip(".")
                normalized = mime_formats.get(normalized.split(";", 1)[0].strip(), normalized)
                if normalized in {"pptx", "ppt", "pps", "dps", "xlsx", "xls", "et", "csv"}:
                    return normalized
    return None


def _render_block(block):
    text = block.get("text", "")
    if block.get("kind") in {"speaker_notes", "speaker_note", "slide_notes"}:
        return "[演讲者备注]\n" + text if text else ""
    if block.get("kind") == "image":
        refs = (block.get("metadata") or {}).get("refs") or []
        links = ["![图片](" + ref + ")" for ref in refs if isinstance(ref, str)]
        return "\n".join(([text] if text else []) + links)
    return text


def _native_ref(pointer, **fields):
    return {"native_path": "native/objects.json", "pointer": pointer, **fields}


def _presentation(blocks, ir, native, options, excluded_units=(), original_units=(), excluded_blocks=()):
    units = {}
    for block in blocks:
        if block.get("unit_type") == "slide":
            key = block.get("unit_id") or "slide_" + str(block.get("unit_index"))
            units.setdefault(key, {"index": block.get("unit_index"), "blocks": [], "is_slide": True})["blocks"].append(block)
    pages = ir.get("pages") or []
    excluded_numbers = set()
    for index, page in enumerate(pages, 1):
        if page.get("page_id") in excluded_units:
            excluded_numbers.add(index)
            continue
        if (page.get("metadata") or {}).get("unit_type") == "slide":
            key = page.get("page_id") or "slide_" + str(index)
            units.setdefault(key, {"index": index, "blocks": [], "is_slide": True})
    native_slides = native.get("slides")
    native_slides = native_slides if isinstance(native_slides, list) else []
    warnings, groups = [], []
    for index, slide in enumerate(native_slides, 1):
        if isinstance(slide, dict):
            number = slide.get("slide", index)
            if number in excluded_numbers:
                continue
            if not any(unit["index"] == number for unit in units.values()):
                units["native_slide_" + str(number)] = {"index": number, "blocks": [], "is_slide": True}
    for block in blocks:
        if block.get("unit_type") != "slide":
            key = block.get("unit_id") or "unit_" + str(block.get("unit_index"))
            units.setdefault(key, {"index": block.get("unit_index"), "blocks": [], "is_slide": False})["blocks"].append(block)
            warnings.append("presentation_non_slide_unit_preserved")
    if not units:
        return {"channel": "presentation", "strategy": "slide", "groups": [],
                "metrics": {"slide_count": 0}, "warnings": ["presentation_slide_units_unavailable"]}
    by_number = {}
    for index, slide in enumerate(native_slides):
        if isinstance(slide, dict):
            number = slide.get("slide", index + 1)
            if number in by_number:
                raise ValueError("duplicate_native_slide_number")
            by_number[number] = (index, slide)
    excluded_assets = {ref for block in excluded_blocks for ref in ((block.get("metadata") or {}).get("refs") or []) if isinstance(ref, str)}
    excluded_shapes = {ref.get("native_id") for block in excluded_blocks for ref in block.get("sources", []) if isinstance(ref.get("native_id"), str)}
    for unit_id, unit in sorted(units.items(), key=lambda item: (item[1]["index"] or 0, item[0])):
        content = sorted(unit["blocks"], key=lambda block: block.get("order", 0))
        notes = {" ".join(block.get("text", "").split()) for block in content if
                 block.get("kind") in {"speaker_notes", "speaker_note", "slide_notes"} and block.get("text", "").strip()}
        parts, seen_notes = [], set()
        for block in content:
            if block.get("kind") in {"speaker_notes", "speaker_note", "slide_notes"}:
                key = " ".join(block.get("text", "").split())
                if key in seen_notes:
                    continue
                seen_notes.add(key)
            rendered = _render_block(block)
            if rendered:
                parts.append(rendered)
        refs, notes_observed = [], bool(notes)
        native_record = by_number.get(unit["index"]) if unit["is_slide"] else None
        if native_record:
            native_index, slide = native_record
            notes_key = "notes" if "notes" in slide else "notes_text"
            value = slide.get(notes_key)
            notes_observed = slide.get("notes_observed") is True or notes_observed
            if isinstance(value, str) and value.strip():
                refs.append(_native_ref(f"/slides/{native_index}/{notes_key}", slide_id=unit_id,
                                        slide_number=unit["index"], kind="speaker_notes"))
                if " ".join(value.split()) not in notes:
                    parts.append("[演讲者备注]\n" + value)
            if not content and unit_id not in original_units:
                for shape_index, shape in enumerate(slide.get("shapes") or []):
                    if not isinstance(shape, dict):
                        raise ValueError("native_slide_shape_requires_object")
                    shape_parts = []
                    if isinstance(shape.get("text"), str) and shape["text"]:
                        shape_parts.append(shape["text"])
                    if isinstance(shape.get("table"), list):
                        shape_parts += [" | ".join(_value(value) for value in row) for row in shape["table"]]
                    if isinstance(shape.get("asset_ref"), str):
                        shape_parts.append("![图片](" + shape["asset_ref"] + ")")
                    if shape_parts:
                        parts.insert(max(0, len(parts) - int(bool(value))), "\n".join(shape_parts))
                        refs.append(_native_ref(f"/slides/{native_index}/shapes/{shape_index}",
                                                slide_id=unit_id, slide_number=unit["index"], kind="shape"))
            else:
                for shape_index, shape in enumerate(slide.get("shapes") or []):
                    asset = shape.get("asset_ref") if isinstance(shape, dict) else None
                    excluded_shape = isinstance(shape, dict) and any(raw_id.endswith("/shape/" + str(shape.get("shape_id"))) for raw_id in excluded_shapes)
                    if isinstance(asset, str) and asset not in excluded_assets and not excluded_shape and not any(asset in part for part in parts):
                        parts.append("![图片](" + asset + ")")
                        refs.append(_native_ref(f"/slides/{native_index}/shapes/{shape_index}/asset_ref",
                                                slide_id=unit_id, slide_number=unit["index"], kind="image_reference"))
        elif not notes:
            warnings.append("presentation_native_notes_unavailable:" + str(unit["index"]))
        if not content and unit_id in original_units and not parts:
            continue
        titles = [block["text"] for block in content if block.get("kind") == "heading" and block.get("text", "").strip()]
        groups.append({"kind": "slide" if unit["is_slide"] else "presentation_metadata", "strategy": "slide", "title_path": titles, "clause_path": [],
                       "blocks": content, "text_override": "\n\n".join(parts), "native_refs": refs, "atomic": True,
                       "context": {"slide_ids": [unit_id] if unit["is_slide"] else [],
                                   "slide_numbers": [unit["index"]] if unit["is_slide"] else [],
                                   "non_slide_unit_ids": [] if unit["is_slide"] else [unit_id],
                                   "notes_observed": notes_observed, "merged_slides": False}})
    threshold = options.get("slide_merge_below_chars", 0)
    maximum = options.get("slide_merge_max_chars", 800)
    if type(threshold) is not int or threshold < 0 or type(maximum) is not int or maximum < 1:
        raise ValueError("invalid_slide_merge_limits")
    merged = []
    for group in groups:
        length = _chars(group["text_override"])
        previous = merged[-1] if merged else None
        if (threshold > 0 and previous and group["context"]["slide_numbers"] and previous["context"]["slide_numbers"]
                and length < threshold and previous["context"]["all_slides_short"]
                and group["context"]["slide_numbers"][0] == previous["context"]["slide_numbers"][-1] + 1
                and _chars(previous["text_override"]) + length + len("\n\n[下一幻灯片]\n\n") <= maximum):
            previous["text_override"] += "\n\n[下一幻灯片]\n\n" + group["text_override"]
            previous["blocks"] += group["blocks"]
            previous["native_refs"] += group["native_refs"]
            previous["title_path"] += [title for title in group["title_path"] if title not in previous["title_path"]]
            previous["context"]["slide_ids"] += group["context"]["slide_ids"]
            previous["context"]["slide_numbers"] += group["context"]["slide_numbers"]
            previous["context"]["notes_observed"] &= group["context"]["notes_observed"]
            previous["context"]["merged_slides"] = True
        else:
            group["context"]["all_slides_short"] = length < threshold
            merged.append(group)
    return {"channel": "presentation", "strategy": "slide", "groups": merged, "warnings": warnings,
            "metrics": {"slide_count": sum(bool(group["context"]["slide_ids"]) for group in groups), "slide_group_count": len(merged),
                        "merged_slide_count": len(groups) - len(merged)}}


def _column(number):
    value = ""
    while number:
        number, rest = divmod(number - 1, 26)
        value = chr(65 + rest) + value
    return value


def _cell_parts(value):
    match = re.fullmatch(r"\$?([A-Za-z]+)\$?([1-9]\d*)", value)
    if not match:
        raise ValueError("invalid_native_cell_coordinate")
    column = 0
    for char in match.group(1).upper():
        column = column * 26 + ord(char) - 64
    return int(match.group(2)), column


def _bounds(value):
    parts = value.split(":")
    top, left = _cell_parts(parts[0])
    bottom, right = _cell_parts(parts[-1])
    if len(parts) > 2 or bottom < top or right < left:
        raise ValueError("invalid_native_merged_range")
    return [top, bottom, left, right]


def _intersects(a, b):
    return a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]


def _union(a, b):
    return [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]


def _row_values(row):
    values = row.get("values")
    if isinstance(values, list):
        return {index: value for index, value in enumerate(values, 1)}
    cells = row.get("cells")
    if not isinstance(cells, list):
        raise ValueError("native_row_requires_values_or_cells")
    result = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("native_cell_requires_object")
        column = cell.get("column")
        if type(column) is not int or column < 1:
            _, column = _cell_parts(cell.get("coordinate", ""))
        result[column] = cell.get("value")
    return result


def _value(value):
    return "null" if value is None else str(value).replace("\r\n", "\\n").replace("\n", "\\n")


def _regions(sheet, rows, merges, warnings):
    result = []
    regions = sheet.get("data_regions")
    if not isinstance(regions, list):
        regions = []
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError("native_data_region_requires_object")
        bounds = [region.get(key) for key in ("start_row", "end_row", "start_column", "end_column")]
        if not all(type(v) is int and v >= 1 for v in bounds) or bounds[1] < bounds[0] or bounds[3] < bounds[2]:
            if isinstance(region.get("range"), str):
                bounds = _bounds(region["range"])
            else:
                raise ValueError("native_data_region_requires_bounds")
        result.append({"bounds": bounds, "records": [(index, region)], "derived": False})
    # Retain rows even when extraction supplied no region list or an incomplete one.
    for row_number, (_, row, values) in rows.items():
        missing = [column for column, value in values.items() if value is not None and str(value).strip()
                   and not any(r["bounds"][0] <= row_number <= r["bounds"][1] and
                               r["bounds"][2] <= column <= r["bounds"][3] for r in result)]
        if missing:
            result.append({"bounds": [row_number, row_number, min(missing), max(missing)], "records": [], "derived": True})
    changed = True
    while changed:
        changed = False
        for region in result:
            for merge in merges:
                if _intersects(region["bounds"], merge["bounds"]):
                    expanded = _union(region["bounds"], merge["bounds"])
                    if expanded != region["bounds"]:
                        region["bounds"] = expanded
                        changed = True
        for index, region in enumerate(result):
            for other in result[index + 1:]:
                a, b = region["bounds"], other["bounds"]
                # Adjacent derived rows may form a region; declared blank-row
                # boundaries stay intact unless a real merged range connects them.
                adjacent = (region["derived"] and other["derived"] and a[2:] == b[2:]
                            and b[0] <= a[1] + 1 and a[0] <= b[1] + 1)
                if _intersects(a, b) or adjacent:
                    region["bounds"] = _union(a, b)
                    region["records"] += other["records"]
                    region["derived"] |= other["derived"]
                    result.remove(other)
                    changed = True
                    break
            if changed:
                break
    if any(region["derived"] for region in result):
        warnings.append("excel_region_supplemented_from_native_cells")
    return sorted(result, key=lambda region: region["bounds"])


def _spreadsheet(blocks, ir, native, options, excluded_blocks=(), excluded_units=()):
    sheets = native.get("sheets")
    if not isinstance(sheets, list):
        fallback = [{"kind": "table" if block.get("kind") == "table" else "sheet_metadata",
                     "strategy": "ir_table_fallback", "title_path": [], "clause_path": [],
                     "blocks": [deepcopy(block)], "text_override": _render_block(block),
                     "native_refs": [], "atomic": True, "context": {"row_grouping_performed": False,
                     "reason": "native_rows_unavailable", "ir_evidence_scope": "whole_ir_block"}} for block in blocks]
        return {"channel": "spreadsheet", "strategy": "ir_table_fallback", "groups": fallback,
                "metrics": {"native_sheet_count": None}, "warnings": ["excel_native_rows_unavailable_use_ir_tables"]}
    limit = options.get("excel_rows_per_group", 10)
    if type(limit) is not int or limit < 1:
        raise ValueError("invalid_excel_rows_per_group")
    header_mode = options.get("excel_header_mode", "first_candidate")
    if header_mode not in {"first_candidate", "none"}:
        raise ValueError("invalid_excel_header_mode")
    groups, warnings, total_regions = [], [], 0
    seen_sheets = set()
    for sheet_index, sheet in enumerate(sheets):
        if not isinstance(sheet, dict) or not isinstance(sheet.get("rows"), list):
            raise ValueError("native_sheet_requires_rows")
        sheet_name = sheet.get("name")
        if not isinstance(sheet_name, str):
            raise ValueError("native_sheet_requires_name")
        if sheet_name in seen_sheets:
            raise ValueError("duplicate_native_sheet_name")
        seen_sheets.add(sheet_name)
        page = (ir.get("pages") or [])[sheet_index] if sheet_index < len(ir.get("pages") or []) else {}
        if page.get("page_id") in excluded_units:
            continue
        excluded_ranges = []
        for block in excluded_blocks:
            metadata = block.get("metadata") or {}
            if metadata.get("sheet_name", sheet_name) != sheet_name or block.get("unit_index") != sheet_index + 1:
                continue
            if isinstance(metadata.get("range"), str):
                excluded_ranges.append(_bounds(metadata["range"]))
            else:
                matched = [region for region in sheet.get("data_regions") or [] if
                           region.get("region_id") is not None and region.get("region_id") == metadata.get("region_id")]
                excluded_ranges.extend(_bounds(region["range"]) for region in matched if isinstance(region.get("range"), str))
        def cell_excluded(number, column):
            return any(top <= number <= bottom and left <= column <= right for top, bottom, left, right in excluded_ranges)
        rows = {}
        for index, row in enumerate(sheet["rows"]):
            if not isinstance(row, dict):
                raise ValueError("native_row_requires_object")
            number = row.get("row_number")
            if type(number) is not int or number < 1 or number in rows:
                raise ValueError("invalid_or_duplicate_native_row_number")
            values = {column: value for column, value in _row_values(row).items() if not cell_excluded(number, column)}
            rows[number] = (index, row, values)
        merges = []
        for index, value in enumerate(sheet.get("merged_cells") or []):
            if not isinstance(value, str):
                raise ValueError("native_merged_range_requires_string")
            merges.append({"range": value, "bounds": _bounds(value), "index": index})
        region_sheet = sheet
        if excluded_ranges:
            region_sheet = dict(sheet)
            region_sheet["data_regions"] = [region for region in sheet.get("data_regions") or [] if
                                           not any(_intersects(_bounds(region["range"]), bounds) for bounds in excluded_ranges)]
            warnings.append("excel_explicit_excluded_region_not_rendered")
        if not isinstance(sheet.get("merged_cells"), list):
            warnings.append("excel_merge_metadata_unavailable")
        regions = _regions(region_sheet, rows, merges, warnings)
        total_regions += len(regions)
        for region_index, region in enumerate(regions):
            top, bottom, left, right = region["bounds"]
            records = region["records"]
            region_ids = [record.get("region_id", f"region_{index + 1}") for index, record in records]
            if not region_ids:
                region_ids = [f"native_cells_region_{region_index + 1}"]
            candidate_rows = []
            for _, record in records:
                candidates = record.get("header_rows")
                if not isinstance(candidates, list):
                    candidates = [item.get("row_number") for item in record.get("header_candidates", []) if isinstance(item, dict)]
                candidate_rows += [number for number in candidates if type(number) is int and top <= number <= bottom and number in rows]
            header_rows = sorted(set(candidate_rows))[:1] if header_mode == "first_candidate" else []
            headers = rows[header_rows[0]][2] if header_rows else {}
            labels = {column: str(headers[column]) if headers.get(column) is not None and str(headers[column]).strip()
                      else _column(column) for column in range(left, right + 1)}
            duplicates = {label for label in labels.values() if list(labels.values()).count(label) > 1}
            labels = {column: label + f" [{_column(column)}]" if label in duplicates else label for column, label in labels.items()}
            data_rows = [number for number in range(top, bottom + 1) if number not in header_rows and
                         (number in rows or any(merge["bounds"][0] <= number <= merge["bounds"][1] and
                                               merge["bounds"][2] <= right and left <= merge["bounds"][3] for merge in merges))]
            row_groups = []
            while data_rows:
                end = min(limit, len(data_rows))
                extended = True
                while extended and end < len(data_rows):
                    boundary = data_rows[end - 1]
                    furthest = max([boundary] + [merge["bounds"][1] for merge in merges if
                                   merge["bounds"][0] <= boundary < merge["bounds"][1] and
                                   merge["bounds"][2] <= right and left <= merge["bounds"][3]])
                    new_end = end
                    while new_end < len(data_rows) and data_rows[new_end] <= furthest:
                        new_end += 1
                    extended = new_end != end
                    end = new_end
                row_groups.append(data_rows[:end])
                data_rows = data_rows[end:]
            if not row_groups:
                row_groups = [[]]  # Header-only regions must not disappear.
            original_ranges = {record.get("range") for _, record in records if record.get("range")}
            region_blocks = [block for block in blocks if
                             ((block.get("metadata") or {}).get("sheet_name") == sheet_name if
                              isinstance((block.get("metadata") or {}).get("sheet_name"), str) else block.get("unit_index") == sheet_index + 1)
                             and ((block.get("metadata") or {}).get("region_id") in region_ids or
                                  (block.get("metadata") or {}).get("range") in original_ranges)]
            for selected in row_groups:
                refs, lines = [], ["工作表: " + sheet_name, "数据区: " + ", ".join(map(str, region_ids)),
                                  "列名: " + "; ".join(_column(column) + "=" + labels[column] for column in labels)]
                for number in header_rows:
                    lines.append("重复表头 行" + str(number) + ": " + "; ".join(_column(column) + ": " + _value(rows[number][2].get(column)) for column in labels))
                included = header_rows + selected
                relevant_merges = [merge for merge in merges if any(merge["bounds"][0] <= number <= merge["bounds"][1] for number in included)
                                   and merge["bounds"][2] <= right and left <= merge["bounds"][3]]
                for merge in relevant_merges:
                    refs.append(_native_ref(f"/sheets/{sheet_index}/merged_cells/{merge['index']}",
                                            sheet_name=sheet_name, merged_range=merge["range"], kind="merged_cells"))
                for number in included:
                    row_index, row, values = rows.get(number, (None, {}, {}))
                    if row_index is not None:
                        refs.append(_native_ref(f"/sheets/{sheet_index}/rows/{row_index}", sheet_name=sheet_name,
                                                row_number=number, repeated_header=number in header_rows, kind="row"))
                    values_text = []
                    for column in labels:
                        coordinate = _column(column) + str(number)
                        if cell_excluded(number, column):
                            values_text.append(labels[column] + ": [已排除]")
                            continue
                        value = values.get(column)
                        merge = next((merge for merge in relevant_merges if merge["bounds"][0] <= number <= merge["bounds"][1]
                                      and merge["bounds"][2] <= column <= merge["bounds"][3]), None)
                        if merge and (number, column) != (merge["bounds"][0], merge["bounds"][2]):
                            anchor_row, _, anchor_col, _ = merge["bounds"]
                            anchor = _column(anchor_col) + str(anchor_row)
                            anchor_value = rows.get(anchor_row, (None, None, {}))[2].get(anchor_col)
                            rendered = "[合并 " + merge["range"] + "; 锚点 " + anchor + ": " + _value(anchor_value) + "]"
                        else:
                            rendered = _value(value)
                        values_text.append(labels[column] + ": " + rendered)
                        cells = row.get("cells") or []
                        cell_index = next((index for index, cell in enumerate(cells) if isinstance(cell, dict) and
                                           (cell.get("coordinate") == coordinate or cell.get("column") == column)), None)
                        if cell_index is not None:
                            pointer = f"/sheets/{sheet_index}/rows/{row_index}/cells/{cell_index}"
                        elif row_index is not None and isinstance(row.get("values"), list) and column <= len(row["values"]):
                            pointer = f"/sheets/{sheet_index}/rows/{row_index}/values/{column - 1}"
                        else:
                            continue
                        refs.append(_native_ref(pointer, sheet_name=sheet_name, row_number=number, column=column,
                                                cell=coordinate, repeated_header=number in header_rows, kind="cell"))
                    if number in selected:
                        lines.append("行" + str(number) + " | " + "; ".join(values_text))
                if relevant_merges:
                    lines.append("合并范围: " + ", ".join(merge["range"] for merge in relevant_merges))
                groups.append({"kind": "row_group", "strategy": "row_group", "title_path": [sheet_name], "clause_path": [],
                    "blocks": deepcopy(region_blocks), "text_override": "\n".join(lines), "native_refs": refs, "atomic": True,
                    "context": {"sheet_name": sheet_name, "sheet_index": sheet_index + 1, "sheet_state": sheet.get("state"),
                                "region_ids": region_ids, "region_bounds": region["bounds"], "row_numbers": selected,
                                "header_rows": header_rows, "header_candidates": sorted(set(candidate_rows)),
                                "header_status": "heuristic" if header_rows else "unavailable",
                                "header_source_statuses": [record.get("header_status") for _, record in records],
                                "repeated_header": bool(header_rows), "merged_ranges": [merge["range"] for merge in relevant_merges],
                                "merged_cells_observed": isinstance(sheet.get("merged_cells"), list),
                                "row_limit_exceeded_for_merged_cells": len(selected) > limit,
                                "ir_evidence_scope": "enclosing_data_region", "formula_evaluation": "not_performed"}})
    represented = {block.get("block_id") for group in groups for block in group["blocks"]}
    for block in blocks:
        if block.get("block_id") in represented:
            continue
        metadata = block.get("metadata") or {}
        target = next((group for group in groups if group["context"].get("sheet_name") == metadata.get("sheet_name")
                       or group["context"].get("sheet_index") == block.get("unit_index")), None)
        if target is None:
            target = {"kind": "sheet_metadata", "strategy": "row_group", "title_path": [], "clause_path": [],
                      "blocks": [], "text_override": "", "native_refs": [], "atomic": True,
                      "context": {"sheet_index": block.get("unit_index"), "row_grouping_performed": False}}
            groups.append(target)
        target["blocks"].append(deepcopy(block))
        rendered = _render_block(block)
        if rendered:
            target["text_override"] += "\n\n[数据区之外的IR内容]\n" + rendered
        target["context"].setdefault("supplemental_ir_block_ids", []).append(block.get("block_id"))
        warnings.append("excel_unmapped_ir_block_preserved")
    return {"channel": "spreadsheet", "strategy": "row_group", "groups": groups, "warnings": list(dict.fromkeys(warnings)),
            "metrics": {"native_sheet_count": len(sheets), "data_region_count": total_regions, "row_group_count": len(groups)}}


def format_groups(blocks, ir, native, source_manifest, config):
    """Return format groups using explicit IR/native format evidence, or None."""
    native = native if isinstance(native, dict) else {}
    source_manifest = source_manifest if isinstance(source_manifest, dict) else {}
    config = config if isinstance(config, dict) else {}
    options = config.get("formats") or {}
    fmt = _format(ir, native, source_manifest)
    if fmt in {"pptx", "ppt", "pps", "dps"} or (not fmt and blocks and all(block.get("unit_type") == "slide" for block in blocks)):
        from .preprocess import _exclusion, _toc_units, NOTE_KINDS
        toc_units = _toc_units(ir)
        body = [block for block in blocks if _exclusion(block, toc_units, config.get("preprocess") or {}) is None
                or {block.get("kind"), block.get("native_kind")} & NOTE_KINDS]
        excluded = [block for block in blocks if block not in body]
        return _presentation(body, ir, native, options, toc_units if config.get("preprocess", {}).get("strip_toc", True) else (),
                             {block.get("unit_id") for block in blocks}, excluded)
    if fmt in {"xlsx", "xls", "et", "csv"}:
        from .preprocess import _exclusion, _toc_units
        toc_units = _toc_units(ir)
        body = [block for block in blocks if _exclusion(block, toc_units, config.get("preprocess") or {}) is None]
        excluded = [block for block in blocks if block not in body]
        return _spreadsheet(body, ir, native, options, excluded,
                            toc_units if config.get("preprocess", {}).get("strip_toc", True) else ())
    return None
