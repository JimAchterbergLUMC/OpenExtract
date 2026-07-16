"""
Structure-aware chunking on top of the sentence-packing chunker.

The cleaned document (Markdown when parsed with pymupdf4llm/docling) is
split at section boundaries first; sentences are packed into chunks *within*
a section, never across one. Each chunk carries its section path as
metadata, and the path is prepended to the text that gets embedded
("[3. Methods > 3.4. MFCC computation] ...") — section names are strong
retrieval signals and cost only a few tokens.

Markdown tables are kept as atomic blocks: a table is never split
mid-table (oversized tables are split at row boundaries with the header
rows repeated) and is flagged `is_table` in the chunk metadata.

For plain-text parses (the pypdf fallback) there is no structure signal;
the whole document becomes one unnamed section and the behavior matches
the previous pipeline.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .text_chunking import chunk_text

_HEADING_RE = re.compile(r"^\s{0,3}(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")

# Keep the embedded section prefix short: at most the last two path levels.
_MAX_PREFIX_LEVELS = 2


@dataclass
class Chunk:
    """One retrievable unit with its provenance metadata."""

    chunk_id: int
    text: str                    # chunk content, without the section prefix
    section: str = ""            # full section path, " > "-separated
    is_table: bool = False
    n_tokens: int = 0            # tokens of embed_text, per the embedder's tokenizer

    @property
    def embed_text(self) -> str:
        """Text as it is embedded and shown to the LLM: section-prefixed."""
        if not self.section:
            return self.text
        levels = self.section.split(" > ")[-_MAX_PREFIX_LEVELS:]
        return f"[{' > '.join(levels)}] {self.text}"

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "section": self.section,
            "is_table": self.is_table,
            "n_tokens": self.n_tokens,
            "n_chars": len(self.text),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Chunk":
        return cls(
            chunk_id=int(data["chunk_id"]),
            text=data["text"],
            section=data.get("section", ""),
            is_table=bool(data.get("is_table", False)),
            n_tokens=int(data.get("n_tokens", 0)),
        )


def _clean_heading_title(title: str) -> str:
    return title.strip().strip("*_").strip()


@dataclass
class _Section:
    path: List[str] = field(default_factory=list)
    prose: List[str] = field(default_factory=list)   # paragraph texts
    tables: List[Tuple[int, str]] = field(default_factory=list)  # (position, table md)


def _split_into_sections(markdown: str) -> List[_Section]:
    """Walk the Markdown line by line, tracking the heading stack and
    collecting paragraphs and table blocks per section."""
    sections: List[_Section] = [_Section()]
    # Heading stack of (level, title) pairs.
    stack: List[Tuple[int, str]] = []
    paragraph: List[str] = []
    table: List[str] = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            sections[-1].prose.append(" ".join(paragraph))
            paragraph = []

    def flush_table():
        nonlocal table
        if table:
            sections[-1].tables.append(
                (len(sections[-1].prose), "\n".join(table))
            )
            table = []

    for line in markdown.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_table()
            level = len(heading.group("hashes"))
            title = _clean_heading_title(heading.group("title"))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(_Section(path=[t for _, t in stack]))
            continue

        if _TABLE_LINE_RE.match(line):
            flush_paragraph()
            table.append(line.strip())
            continue
        flush_table()

        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.strip())

    flush_paragraph()
    flush_table()
    return [s for s in sections if s.prose or s.tables]


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_table(table_md: str, budget: int, tokenizer) -> List[str]:
    """Split an oversized Markdown table at row boundaries, repeating the
    header (first two lines: header row + separator) in every piece."""
    lines = table_md.splitlines()
    if len(lines) <= 2:
        return [table_md]
    header, rows = lines[:2], lines[2:]

    pieces: List[str] = []
    current = list(header)
    for row in rows:
        candidate = "\n".join(current + [row])
        if len(current) > 2 and _count_tokens(tokenizer, candidate) > budget:
            pieces.append("\n".join(current))
            current = list(header)
        current.append(row)
    if len(current) > 2:
        pieces.append("\n".join(current))
    return pieces


def chunk_document(
    text: str,
    chunk_tokens: int,
    chunk_overlap: int,
    tokenizer,
    max_tokens: Optional[int] = None,
    is_markdown: bool = True,
) -> List[Chunk]:
    """
    Chunk a cleaned document into section-aware, embedder-sized units.

    Args:
        text: Cleaned document text (Markdown for structured parses).
        chunk_tokens: Target maximum tokens per chunk *including* the section
            prefix, measured with `tokenizer`.
        chunk_overlap: Approximate token overlap between consecutive chunks
            within a section.
        tokenizer: The embedder's tokenizer (used for counting only).
        max_tokens: The embedder's maximum sequence length; chunks are capped
            so embed_text never gets silently truncated at encoding time.
        is_markdown: When False (pypdf fallback), the document is treated as
            a single unnamed section.

    Returns:
        List of Chunk objects in document order.
    """
    if is_markdown:
        sections = _split_into_sections(text)
    else:
        sections = [_Section(prose=[p for p in re.split(r"\n\s*\n", text) if p.strip()])]

    chunks: List[Chunk] = []
    for section in sections:
        path = " > ".join(section.path)
        prefix_levels = section.path[-_MAX_PREFIX_LEVELS:]
        prefix = f"[{' > '.join(prefix_levels)}] " if prefix_levels else ""
        prefix_tokens = _count_tokens(tokenizer, prefix) if prefix else 0

        budget = max(32, chunk_tokens - prefix_tokens)
        section_max = max_tokens - prefix_tokens if max_tokens is not None else None

        # Prose: pack sentences within this section only.
        section_text = "\n\n".join(section.prose)
        if section_text.strip():
            for piece in chunk_text(
                section_text,
                chunk_tokens=budget,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer,
                max_tokens=section_max,
            ):
                chunks.append(Chunk(chunk_id=-1, text=piece, section=path))

        # Tables: atomic blocks, split at row boundaries only if oversized.
        for _, table_md in section.tables:
            effective_budget = min(budget, section_max) if section_max else budget
            for piece in _split_table(table_md, effective_budget, tokenizer):
                chunks.append(
                    Chunk(chunk_id=-1, text=piece, section=path, is_table=True)
                )

    for i, chunk in enumerate(chunks):
        chunk.chunk_id = i
        chunk.n_tokens = _count_tokens(tokenizer, chunk.embed_text)
    return chunks
