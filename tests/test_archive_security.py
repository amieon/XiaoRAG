from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.archive_security import ArchiveSafetyError, iter_archive_documents_safely
from doc_parser.preparation import detect_format, prepare_document


class ArchiveSecurityTests(unittest.TestCase):
    def test_first_document_is_yielded_before_later_archive_is_opened(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_bytes = io.BytesIO()
            with zipfile.ZipFile(bad_bytes, "w") as archive:
                archive.writestr("../outside.txt", "bad")
            source = root / "stream.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("first.txt", "ready")
                archive.writestr("bad.zip", bad_bytes.getvalue())
            expanded = iter_archive_documents_safely(
                source, root / "out", classify=detect_format,
            )
            first = next(expanded)
            self.assertEqual(first.source_path, "stream.zip/first.txt")
            with self.assertRaisesRegex(ArchiveSafetyError, "Unsafe archive"):
                next(expanded)

    def test_encrypted_zip_fails_explicitly_without_extracting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "encrypted.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("secret.txt", "not really encrypted; flag is sufficient")
            payload = bytearray(source.read_bytes())
            local = payload.index(b"PK\x03\x04")
            central = payload.index(b"PK\x01\x02")
            for offset in (local + 6, central + 8):
                flags = int.from_bytes(payload[offset:offset + 2], "little") | 0x1
                payload[offset:offset + 2] = flags.to_bytes(2, "little")
            source.write_bytes(payload)

            output = root / "out"
            with self.assertRaisesRegex(ArchiveSafetyError, "Encrypted archive member is unsupported"):
                prepare_document(source, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("Encrypted archive member is unsupported", manifest["error"])
            self.assertFalse((output / "_extracted/archive-0001/secret.txt").exists())

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "link.zip"
            with zipfile.ZipFile(source, "w") as archive:
                info = zipfile.ZipInfo("escape")
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                archive.writestr(info, "/tmp")
            with self.assertRaisesRegex(ArchiveSafetyError, "symbolic link"):
                list(iter_archive_documents_safely(
                    source, root / "out", classify=detect_format,
                ))

    def test_nested_archive_shares_member_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inner_bytes = io.BytesIO()
            with zipfile.ZipFile(inner_bytes, "w") as archive:
                archive.writestr("inside.txt", "inside")
            source = root / "outer.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("outside.txt", "outside")
                archive.writestr("inner.zip", inner_bytes.getvalue())
            with self.assertRaisesRegex(ArchiveSafetyError, "extraction limits"):
                list(iter_archive_documents_safely(
                    source, root / "out", classify=detect_format, max_members=2,
                ))

    def test_nesting_depth_is_limited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inner_bytes = io.BytesIO()
            with zipfile.ZipFile(inner_bytes, "w") as archive:
                archive.writestr("inside.txt", "inside")
            source = root / "outer.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("inner.zip", inner_bytes.getvalue())
            with self.assertRaisesRegex(ArchiveSafetyError, "nesting exceeds"):
                list(iter_archive_documents_safely(
                    source, root / "out", classify=detect_format, max_depth=1,
                ))

    def test_suspicious_compression_ratio_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "ratio.zip"
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("zeros.bin", b"\0" * (1024 * 1024))
            with self.assertRaisesRegex(ArchiveSafetyError, "compression ratio"):
                list(iter_archive_documents_safely(
                    source, root / "out", classify=detect_format,
                    max_compression_ratio=10,
                ))


if __name__ == "__main__":
    unittest.main()
