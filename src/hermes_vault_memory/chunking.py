from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
import uuid
from typing import Iterable

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
LIST_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(slots=True)
class Block:
    text: str
    start_line: int
    end_line: int


@dataclass(slots=True)
class Chunk:
    chunk_index: int
    chunk_id: str
    text: str
    section_path: tuple[str, ...]
    start_line: int
    end_line: int
    char_count: int
    content_hash: str


@dataclass(slots=True)
class ParsedDocument:
    vault: str
    root: Path
    path: Path
    relative_path: str
    title: str
    file_hash: str
    size: int
    mtime: float
    chunks: list[Chunk] = field(default_factory=list)


@dataclass(slots=True)
class Section:
    heading_path: tuple[str, ...]
    blocks: list[Block] = field(default_factory=list)


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _clean_text(text: str) -> str:
    lines = [line.rstrip() for line in _normalise_newlines(text).split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _extract_title(text: str, fallback: str) -> str:
    for raw_line in _normalise_newlines(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        heading = HEADING_RE.match(line)
        if heading and len(heading.group(1)) == 1:
            return heading.group(2).strip()
        return line[:120]
    return fallback


def _split_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx]
        stripped = line.strip()

        if not stripped:
            idx += 1
            continue

        start = idx + 1

        heading = HEADING_RE.match(stripped)
        if heading:
            blocks.append(Block(text=stripped, start_line=start, end_line=start))
            idx += 1
            continue

        fence = FENCE_RE.match(stripped)
        if fence:
            fence_marker = fence.group(1)
            block_lines = [line]
            idx += 1
            while idx < total:
                block_lines.append(lines[idx])
                if lines[idx].strip().startswith(fence_marker):
                    idx += 1
                    break
                idx += 1
            blocks.append(Block(text="\n".join(block_lines), start_line=start, end_line=start + len(block_lines) - 1))
            continue

        if LIST_RE.match(stripped):
            block_lines = [line]
            idx += 1
            while idx < total:
                nxt = lines[idx]
                nxt_stripped = nxt.strip()
                if not nxt_stripped:
                    break
                if HEADING_RE.match(nxt_stripped) or FENCE_RE.match(nxt_stripped):
                    break
                if LIST_RE.match(nxt.lstrip()):
                    block_lines.append(nxt)
                    idx += 1
                    continue
                if nxt.startswith((" ", "\t")):
                    block_lines.append(nxt)
                    idx += 1
                    continue
                break
            blocks.append(Block(text="\n".join(block_lines), start_line=start, end_line=start + len(block_lines) - 1))
            continue

        block_lines = [line]
        idx += 1
        while idx < total:
            nxt = lines[idx]
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                idx += 1
                break
            if HEADING_RE.match(nxt_stripped) or FENCE_RE.match(nxt_stripped):
                break
            if LIST_RE.match(nxt_stripped) and not nxt.startswith((" ", "\t")):
                break
            block_lines.append(nxt)
            idx += 1
        blocks.append(Block(text="\n".join(block_lines), start_line=start, end_line=start + len(block_lines) - 1))

    return blocks


def _section_blocks(blocks: list[Block]) -> list[Section]:
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current = Section(heading_path=())

    def flush_current() -> None:
        nonlocal current
        if current.blocks:
            sections.append(current)
        current = Section(heading_path=tuple(h for _, h in heading_stack))

    for block in blocks:
        heading = HEADING_RE.match(block.text.strip())
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            flush_current()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current.heading_path = tuple(h for _, h in heading_stack)
            current.blocks.append(block)
            continue
        current.blocks.append(block)

    flush_current()
    return sections


def _split_long_block(text: str, max_chars: int) -> list[str]:
    cleaned = _clean_text(text)
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    paragraphs = re.split(r"\n\s*\n", cleaned)
    buffer = ""

    def push_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        candidate = paragraph.strip()
        if len(candidate) > max_chars:
            push_buffer()
            for start in range(0, len(candidate), max_chars):
                chunks.append(candidate[start : start + max_chars].strip())
            continue
        if not buffer:
            buffer = candidate
        elif len(buffer) + len(candidate) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{candidate}"
        else:
            push_buffer()
            buffer = candidate
    push_buffer()
    return [chunk for chunk in chunks if chunk.strip()]


def _chunk_with_prefix(
    *,
    prefix: str,
    blocks: Iterable[Block],
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    block_list = list(blocks)
    if not block_list:
        return []

    chunks: list[tuple[str, int, int]] = []
    current_blocks: list[str] = []
    current_start = block_list[0].start_line
    current_end = block_list[0].end_line

    def flush() -> None:
        nonlocal current_blocks, current_start, current_end
        if not current_blocks:
            return
        body = "\n\n".join(current_blocks).strip()
        if body:
            chunks.append((f"{prefix}{body}" if prefix else body, current_start, current_end))
        current_blocks = []

    effective_limit = max(200, max_chars - len(prefix))
    for block in block_list:
        pieces = _split_long_block(block.text, effective_limit)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            candidate = piece if not current_blocks else f"\n\n".join(current_blocks + [piece])
            if current_blocks and len(candidate) > effective_limit:
                flush()
                current_start = block.start_line
            if not current_blocks:
                current_start = block.start_line
            current_blocks.append(piece)
            current_end = block.end_line
            candidate = "\n\n".join(current_blocks)
            if len(candidate) >= effective_limit:
                flush()
                if overlap_chars > 0 and chunks:
                    tail = chunks[-1][0][-overlap_chars:].strip()
                    if tail:
                        current_blocks = [tail]
                        current_start = block.start_line
                        current_end = block.end_line
    flush()
    return chunks


def parse_markdown_file(path: Path, vault: str, root: Path, *, chunk_size: int, chunk_overlap: int) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _normalise_newlines(raw)
    relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    title = _extract_title(text, fallback=path.stem)
    file_hash = sha256(text.encode("utf-8")).hexdigest()
    stat = path.stat()

    lines = text.split("\n")
    blocks = _split_blocks(lines)
    sections = _section_blocks(blocks)

    chunks: list[Chunk] = []
    seen: dict[tuple[tuple[str, ...], str], int] = {}
    for section_index, section in enumerate(sections):
        heading_path = section.heading_path
        prefix_parts = [f"Vault: {vault}", f"Path: {relative_path}"]
        if heading_path:
            prefix_parts.append(f"Section: {' > '.join(heading_path)}")
        prefix = "\n".join(prefix_parts) + "\n\n"
        raw_chunks = _chunk_with_prefix(
            prefix=prefix,
            blocks=section.blocks,
            max_chars=chunk_size,
            overlap_chars=chunk_overlap,
        )
        if not raw_chunks and heading_path:
            # Keep heading-only sections visible for lookup.
            raw_chunks = [(prefix.rstrip(), 1, 1)]
        for raw_text, start_line, end_line in raw_chunks:
            content_hash = sha256(raw_text.encode("utf-8")).hexdigest()
            occurrence_key = (heading_path, content_hash)
            ordinal = seen.get(occurrence_key, 0)
            seen[occurrence_key] = ordinal + 1
            stable_key = f"{vault}:{relative_path}:{' > '.join(heading_path)}:{content_hash}:{ordinal}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    chunk_id=chunk_id,
                    text=raw_text,
                    section_path=heading_path,
                    start_line=start_line,
                    end_line=end_line,
                    char_count=len(raw_text),
                    content_hash=content_hash,
                )
            )

    if not chunks:
        prefix = f"Vault: {vault}\nPath: {relative_path}\n\n"
        content_hash = sha256(prefix.encode("utf-8")).hexdigest()
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{vault}:{relative_path}:{content_hash}:0"))
        chunks.append(
            Chunk(
                chunk_index=0,
                chunk_id=chunk_id,
                text=prefix.rstrip(),
                section_path=(),
                start_line=1,
                end_line=1,
                char_count=len(prefix),
                content_hash=content_hash,
            )
        )

    return ParsedDocument(
        vault=vault,
        root=root,
        path=path,
        relative_path=relative_path,
        title=title,
        file_hash=file_hash,
        size=stat.st_size,
        mtime=stat.st_mtime,
        chunks=chunks,
    )
