"""
Two-tier PDF parsing backends with a last-resort fallback.

Backends
--------
- "pymupdf"  (default): pymupdf4llm — fast, layout/column-aware, emits
  Markdown with `#` headings and Markdown tables. One self-contained wheel.
- "docling"  (opt-in, high fidelity): IBM's ML layout model — best section
  and table structure. Heavy (downloads model weights on first use), slow on
  CPU; requires `pip install docling`.
- "pypdf"    (last resort): the original plain-text extractor. No layout or
  structure awareness; used automatically when the requested backend fails
  or yields no text.

Every parse returns a `ParsedPaper` carrying the (Markdown) text, per-page
texts when the backend can provide them (needed by the repeated-header
cleaner), and full parser provenance for the cache.
"""

import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PARSER_BACKENDS = ("pymupdf", "docling", "pypdf")
DEFAULT_BACKEND = "pymupdf"
FALLBACK_BACKEND = "pypdf"


@dataclass
class ParsedPaper:
    """Result of parsing one PDF."""

    text: str                      # full document text (Markdown for pymupdf/docling)
    parser: str                    # backend that actually produced the text
    parser_version: str
    requested_parser: str          # backend the caller asked for
    fallback_used: bool
    n_pages: int
    pages: Optional[List[str]] = None   # per-page text, when the backend provides it
    is_markdown: bool = False           # True when headings/tables are Markdown
    pdf_metadata: Dict = field(default_factory=dict)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parse_with_pymupdf(pdf_path: Path, requested: str) -> ParsedPaper:
    import pymupdf4llm  # noqa: PLC0415 (lazy: optional dependency)

    page_dicts = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    pages = [p.get("text", "") or "" for p in page_dicts]
    pdf_meta = {}
    if page_dicts:
        meta = page_dicts[0].get("metadata", {}) or {}
        pdf_meta = {
            k: meta.get(k)
            for k in ("title", "author", "subject", "creationDate", "producer")
            if meta.get(k)
        }
    return ParsedPaper(
        text="\n\n".join(pages),
        parser="pymupdf4llm",
        parser_version=_package_version("pymupdf4llm"),
        requested_parser=requested,
        fallback_used=False,
        n_pages=len(pages),
        pages=pages,
        is_markdown=True,
        pdf_metadata=pdf_meta,
    )


def _parse_with_docling(pdf_path: Path, requested: str) -> ParsedPaper:
    try:
        from docling.document_converter import DocumentConverter  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The 'docling' high-fidelity parser is not installed. "
            "Run `pip install docling` (downloads ~500 MB of model weights "
            "on first use), or use the default parser."
        ) from exc

    result = DocumentConverter().convert(str(pdf_path))
    document = result.document
    # Docling classifies page headers/footers as "furniture" and already
    # excludes them from the Markdown export.
    markdown = document.export_to_markdown()
    n_pages = len(getattr(document, "pages", {}) or {})
    return ParsedPaper(
        text=markdown,
        parser="docling",
        parser_version=_package_version("docling"),
        requested_parser=requested,
        fallback_used=False,
        n_pages=n_pages,
        pages=None,  # docling's Markdown export is not page-segmented
        is_markdown=True,
        pdf_metadata={},
    )


def _parse_with_pypdf(pdf_path: Path, requested: str, fallback_used: bool) -> ParsedPaper:
    from pypdf import PdfReader  # noqa: PLC0415

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    pages = [p.replace("\u00ad", "") for p in pages]
    return ParsedPaper(
        text="\n\n".join(pages),
        parser="pypdf",
        parser_version=_package_version("pypdf"),
        requested_parser=requested,
        fallback_used=fallback_used,
        n_pages=len(pages),
        pages=pages,
        is_markdown=False,
        pdf_metadata={},
    )


def parse_pdf(pdf_path: Path, backend: str = DEFAULT_BACKEND) -> ParsedPaper:
    """
    Parse a PDF with the requested backend, falling back to pypdf when the
    backend raises or produces no text.

    Args:
        pdf_path: Path to the PDF file.
        backend: One of "pymupdf" (default), "docling" (high fidelity,
            requires `pip install docling`), or "pypdf".

    Returns:
        A ParsedPaper. `parsed.parser` records which backend actually ran,
        `parsed.fallback_used` whether the pypdf fallback kicked in.

    Raises:
        ValueError: If `backend` is not a known backend name.
        RuntimeError: If "docling" is requested but not installed (an
            explicit user choice should fail loudly, not silently degrade).
    """
    if backend not in PARSER_BACKENDS:
        raise ValueError(
            f"Unknown parser backend '{backend}'. Choose from {PARSER_BACKENDS}."
        )

    if backend == "pypdf":
        return _parse_with_pypdf(pdf_path, requested=backend, fallback_used=False)

    parse_fn = _parse_with_pymupdf if backend == "pymupdf" else _parse_with_docling
    try:
        parsed = parse_fn(pdf_path, requested=backend)
        if parsed.text.strip():
            return parsed
        print(
            f"Warning: parser '{backend}' extracted no text from "
            f"{pdf_path.name}; falling back to pypdf."
        )
    except RuntimeError:
        # Missing optional dependency for an explicitly requested backend.
        raise
    except Exception as exc:
        print(
            f"Warning: parser '{backend}' failed on {pdf_path.name} "
            f"({type(exc).__name__}: {exc}); falling back to pypdf."
        )

    return _parse_with_pypdf(pdf_path, requested=backend, fallback_used=True)
