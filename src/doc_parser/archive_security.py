"""压缩包安全边界：以惰性 BFS 展开成员，并保留完整来源链。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile
from typing import Callable, Iterator
import zipfile


DEFAULT_MAX_ARCHIVE_MEMBERS = 10_000
DEFAULT_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_DEPTH = 4
DEFAULT_MAX_COMPRESSION_RATIO = 1_000
_COPY_CHUNK_BYTES = 1024 * 1024


class ArchiveSafetyError(ValueError):
    """压缩包违反路径、类型或资源预算。"""


@dataclass(frozen=True, slots=True)
class ExpandedDocument:
    """一个非压缩包成员；path 是本地文件，其余字段只描述来源。"""

    path: Path
    source_path: str
    archive_path: str
    relative_path: str


@dataclass(slots=True)
class _WorkItem:
    path: Path
    origin_parts: tuple[str, ...]
    archive_parts: tuple[str, ...]
    relative_path: str
    depth: int


@dataclass(slots=True)
class _Budget:
    members: int
    bytes: int


def _safe_member_name(name: str) -> PurePosixPath:
    if "\x00" in name:
        raise ArchiveSafetyError(f"Unsafe archive member: {name!r}")
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or PureWindowsPath(normalized).drive
        or any(part in {"", ".."} for part in pure.parts)
    ):
        raise ArchiveSafetyError(f"Unsafe archive member: {name}")
    parts = tuple(part for part in pure.parts if part != ".")
    if not parts:
        raise ArchiveSafetyError(f"Unsafe archive member: {name}")
    return PurePosixPath(*parts)


def _checked_target(destination: Path, relative: PurePosixPath) -> Path:
    target = destination.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ArchiveSafetyError(f"Unsafe archive member: {relative.as_posix()}")
    return target


def _reserve(budget: _Budget, members: int, size: int) -> None:
    if members > budget.members or size > budget.bytes:
        raise ArchiveSafetyError("Archive exceeds extraction limits")
    budget.members -= members
    budget.bytes -= size


def _copy_exact(source, target: Path, expected_size: int) -> None:
    written = 0
    with target.open("wb") as output:
        while True:
            chunk = source.read(min(_COPY_CHUNK_BYTES, expected_size - written + 1))
            if not chunk:
                break
            written += len(chunk)
            if written > expected_size:
                raise ArchiveSafetyError(f"Archive member exceeds declared size: {target.name}")
            output.write(chunk)
    if written != expected_size:
        raise ArchiveSafetyError(f"Archive member size mismatch: {target.name}")


def _extract_zip(
    source: Path,
    destination: Path,
    budget: _Budget,
    max_compression_ratio: int,
) -> list[tuple[Path, str]]:
    extracted: list[tuple[Path, str]] = []
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        files = [item for item in entries if not item.is_dir()]
        if any(item.file_size < 0 or item.compress_size < 0 for item in files):
            raise ArchiveSafetyError("Archive contains a negative member size")
        total_size = sum(item.file_size for item in files)
        _reserve(budget, len(files), total_size)
        seen: set[str] = set()
        for item in entries:
            relative = _safe_member_name(item.filename)
            if item.is_dir():
                continue
            key = relative.as_posix()
            folded = key.casefold()
            if folded in seen:
                raise ArchiveSafetyError(f"Duplicate archive member: {key}")
            seen.add(folded)
            mode = item.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ArchiveSafetyError(f"Archive contains a symbolic link: {item.filename}")
            if mode & 0o170000 not in {0, 0o100000}:
                raise ArchiveSafetyError(f"Archive contains a non-regular file: {item.filename}")
            if item.flag_bits & 0x1:
                raise ArchiveSafetyError(f"Encrypted archive member is unsupported: {item.filename}")
            if item.file_size > 0 and item.file_size / max(item.compress_size, 1) > max_compression_ratio:
                raise ArchiveSafetyError(f"Archive compression ratio is suspicious: {item.filename}")
            target = _checked_target(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as member:
                _copy_exact(member, target, item.file_size)
            extracted.append((target, key))
    return extracted


def _extract_tar(source: Path, destination: Path, budget: _Budget) -> list[tuple[Path, str]]:
    extracted: list[tuple[Path, str]] = []
    with tarfile.open(source, "r:*") as archive:
        entries = archive.getmembers()
        files = [item for item in entries if item.isfile()]
        if any(item.size < 0 for item in files):
            raise ArchiveSafetyError("Archive contains a negative member size")
        _reserve(budget, len(files), sum(item.size for item in files))
        seen: set[str] = set()
        for item in entries:
            relative = _safe_member_name(item.name)
            if item.isdir():
                continue
            if item.issym() or item.islnk():
                raise ArchiveSafetyError(f"Archive contains a link: {item.name}")
            if not item.isfile():
                raise ArchiveSafetyError(f"Archive contains a non-regular file: {item.name}")
            key = relative.as_posix()
            folded = key.casefold()
            if folded in seen:
                raise ArchiveSafetyError(f"Duplicate archive member: {key}")
            seen.add(folded)
            target = _checked_target(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            member = archive.extractfile(item)
            if member is None:
                raise ArchiveSafetyError(f"Archive member cannot be read: {item.name}")
            with member:
                _copy_exact(member, target, item.size)
            extracted.append((target, key))
    return extracted


def iter_archive_documents_safely(
    source: str | Path,
    staging: str | Path,
    *,
    classify: Callable[[Path], str],
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_depth: int = DEFAULT_MAX_ARCHIVE_DEPTH,
    max_compression_ratio: int = DEFAULT_MAX_COMPRESSION_RATIO,
) -> Iterator[ExpandedDocument]:
    """以 BFS 逐个产出普通文件；压缩包只入队，不递归调用文档处理。

    classify 必须先把 DOCX/PPTX/XLSX 识别为 Office 文档，再识别普通 ZIP。
    数量、声明解压字节数与嵌套深度使用整次展开共享的预算。
    """
    if min(max_members, max_bytes, max_compression_ratio) <= 0 or max_depth < 0:
        raise ValueError("Archive limits must be positive and max_depth must be non-negative")
    root = Path(source).resolve()
    staging_root = Path(staging).resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    budget = _Budget(members=max_members, bytes=max_bytes)
    queue = deque([_WorkItem(root, (root.name,), (), root.name, 0)])
    archive_index = 0

    while queue:
        item = queue.popleft()
        try:
            kind = classify(item.path)
        except ValueError:
            kind = "unsupported"
        if kind not in {"zip", "tar"}:
            yield ExpandedDocument(
                path=item.path,
                source_path="/".join(item.origin_parts),
                archive_path="/".join(item.archive_parts),
                relative_path=item.relative_path,
            )
            continue
        if item.depth >= max_depth:
            raise ArchiveSafetyError(f"Archive nesting exceeds {max_depth} levels")

        archive_index += 1
        destination = staging_root / f"archive-{archive_index:04d}"
        destination.mkdir(parents=True, exist_ok=False)
        if kind == "zip":
            members = _extract_zip(item.path, destination, budget, max_compression_ratio)
        else:
            members = _extract_tar(item.path, destination, budget)
        archive_parts = item.origin_parts
        for path, member_name in members:
            queue.append(_WorkItem(
                path=path,
                origin_parts=item.origin_parts + (member_name,),
                archive_parts=archive_parts,
                relative_path=member_name,
                depth=item.depth + 1,
            ))
