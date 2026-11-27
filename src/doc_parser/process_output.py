"""Decode external-process diagnostics without weakening the OCR JSON protocol."""
from __future__ import annotations

import locale


def decode_process_output(value: bytes | str | None) -> str:
    """Decode a diagnostic stream; replacement is allowed only for display."""
    if value is None or isinstance(value, str):
        return value or ""
    encodings = list(dict.fromkeys([locale.getpreferredencoding(False), "utf-8", "gb18030"]))
    for encoding in encodings:
        try:
            return value.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return value.decode("utf-8", errors="replace")
