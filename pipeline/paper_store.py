"""
Content-addressed paper cache: parsed text, chunks, and embeddings.

Layout (one directory per paper, keyed by content hash — filenames are
metadata, never identifiers):

    cache/
    └── {paper_id}/                      # sha256(pdf bytes)[:16]
        ├── meta.json                    # provenance: file info + one entry per parser
        ├── text.{parser}.md             # cleaned text per parser backend
        └── {index_key}/                 # parser + chunk params + embedder
            ├── config.json              # exact config that produced this index
            ├── chunks.jsonl             # one chunk per line — human-inspectable
            └── embeddings.npy           # float32 matrix, row i = chunk i

`chunks.jsonl` is the artifact to open when judging chunk quality: each line
holds chunk_id, section path, token/char counts, table flag, and the text.

The index key is human-readable, e.g.
`pymupdf4llm-1.28.0_t300_o50_neuml-pubmedbert-base-embeddings`, so a config
change can never silently reuse stale chunks or embeddings. The cache is
disposable: deleting it only costs recompute.

All writes are atomic (tmp file + rename), so an interrupted run can never
leave a truncated artifact behind.

Inspect from the command line:

    python -m pipeline.paper_store cache/            # list cached papers
    python -m pipeline.paper_store cache/{paper_id}  # paper detail + chunk preview
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .section_chunking import Chunk

SCHEMA_VERSION = 1

# Below this many extracted characters per page the PDF is probably scanned
# images and needs OCR; answers would otherwise silently degrade.
_NEEDS_OCR_CHARS_PER_PAGE = 200


def compute_paper_id(pdf_path: Path) -> str:
    """First 16 hex chars of the sha256 of the PDF bytes."""
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip("-")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Dict) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


@dataclass
class PaperIndex:
    """Everything retrieval needs for one paper, plus provenance."""

    paper_id: str
    chunks: List[Chunk]
    embeddings: np.ndarray
    meta: Dict
    parser_info: Dict            # the meta["parsers"] entry used for this index
    index_dir: Path
    chunks_from_cache: bool = False
    embeddings_from_cache: bool = False


class PaperStore:
    """Reads and writes the per-paper cache directory."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    # -- paths ------------------------------------------------------------

    def paper_dir(self, paper_id: str) -> Path:
        return self.cache_dir / paper_id

    def index_dir(self, paper_id: str, index_key: str) -> Path:
        return self.paper_dir(paper_id) / index_key

    @staticmethod
    def make_index_key(
        parser: str,
        parser_version: str,
        chunk_tokens: int,
        chunk_overlap: int,
        embedder_name: str,
    ) -> str:
        """Human-readable key covering everything that shapes chunks and
        embeddings. The embedder belongs here because chunk boundaries are
        measured with *its* tokenizer."""
        return (
            f"{_slugify(parser)}-{_slugify(parser_version)}"
            f"_t{chunk_tokens}_o{chunk_overlap}_{_slugify(embedder_name)}"
        )

    # -- meta + text -------------------------------------------------------

    def load_meta(self, paper_id: str) -> Optional[Dict]:
        path = self.paper_dir(paper_id) / "meta.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_meta(self, paper_id: str, meta: Dict) -> None:
        directory = self.paper_dir(paper_id)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(directory / "meta.json", meta)

    def save_text(self, paper_id: str, parser_used: str, text: str) -> None:
        directory = self.paper_dir(paper_id)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(directory / f"text.{_slugify(parser_used)}.md", text)

    def load_text(self, paper_id: str, parser_used: str) -> Optional[str]:
        path = self.paper_dir(paper_id) / f"text.{_slugify(parser_used)}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    # -- chunks + embeddings ------------------------------------------------

    def load_chunks(self, paper_id: str, index_key: str) -> Optional[List[Chunk]]:
        path = self.index_dir(paper_id, index_key) / "chunks.jsonl"
        if not path.exists():
            return None
        try:
            return [
                Chunk.from_dict(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def save_chunks(
        self, paper_id: str, index_key: str, chunks: List[Chunk], config: Dict
    ) -> None:
        directory = self.index_dir(paper_id, index_key)
        directory.mkdir(parents=True, exist_ok=True)
        lines = "\n".join(
            json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks
        )
        _atomic_write_text(directory / "chunks.jsonl", lines + "\n")
        _atomic_write_json(
            directory / "config.json",
            {"schema_version": SCHEMA_VERSION, "created_at": _utc_now(), **config},
        )

    def load_embeddings(
        self, paper_id: str, index_key: str, n_chunks: int
    ) -> Optional[np.ndarray]:
        path = self.index_dir(paper_id, index_key) / "embeddings.npy"
        if not path.exists():
            return None
        try:
            embeddings = np.load(path)
        except (ValueError, OSError):
            return None
        return embeddings if embeddings.shape[0] == n_chunks else None

    def save_embeddings(
        self, paper_id: str, index_key: str, embeddings: np.ndarray
    ) -> None:
        directory = self.index_dir(paper_id, index_key)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "embeddings.npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, embeddings.astype(np.float32))
        os.replace(tmp, directory / "embeddings.npy")


def build_or_load_paper_index(
    pdf_path: Path,
    store: PaperStore,
    parser_backend: str,
    chunk_tokens: int,
    chunk_overlap: int,
    embedder,
    embedder_name: str,
    batch_size: int = 8,
    use_cache: bool = True,
) -> PaperIndex:
    """
    Return chunks + embeddings for a paper, reusing cached artifacts when the
    full configuration (parser + version, chunk params, embedder) matches.

    Cache validity is enforced structurally: the index directory name encodes
    the configuration, so a different parser, chunk size, or embedder can
    never hit a stale entry. The paper itself is addressed by content hash,
    so renamed or duplicated PDFs share one cache entry.
    """
    from .pdf_parsing import parse_pdf  # noqa: PLC0415 (avoid import cycle)
    from .section_chunking import chunk_document  # noqa: PLC0415
    from .text_cleaning import clean_parsed_paper  # noqa: PLC0415

    paper_id = compute_paper_id(pdf_path)
    tokenizer = embedder.tokenizer
    max_tokens = getattr(embedder, "max_seq_length", None)

    meta = store.load_meta(paper_id) or {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "filename": pdf_path.name,
        "file_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "file_size_bytes": pdf_path.stat().st_size,
        "created_at": _utc_now(),
        "parsers": {},
    }

    # -- parse + clean (or reuse cached text) --------------------------------
    chunks: Optional[List[Chunk]] = None
    text: Optional[str] = None
    parser_info = meta.get("parsers", {}).get(parser_backend) if use_cache else None
    if parser_info is not None:
        index_key = PaperStore.make_index_key(
            parser_info["used"],
            parser_info["version"],
            chunk_tokens,
            chunk_overlap,
            embedder_name,
        )
        chunks = store.load_chunks(paper_id, index_key)
        text = store.load_text(paper_id, parser_info["used"])
        if chunks is None and text is None:
            parser_info = None  # cache entry incomplete; rebuild below

    if parser_info is None:
        parsed = parse_pdf(pdf_path, backend=parser_backend)
        text, cleaning_report = clean_parsed_paper(parsed)
        chars_per_page = len(text) / max(1, parsed.n_pages)
        needs_ocr = chars_per_page < _NEEDS_OCR_CHARS_PER_PAGE
        if needs_ocr:
            print(
                f"Warning: {pdf_path.name} yields only {chars_per_page:.0f} "
                "chars/page; likely a scanned PDF that needs OCR."
            )
        parser_info = {
            "requested": parsed.requested_parser,
            "used": parsed.parser,
            "version": parsed.parser_version,
            "fallback_used": parsed.fallback_used,
            "is_markdown": parsed.is_markdown,
            "pdf_metadata": parsed.pdf_metadata,
            "cleaning": cleaning_report,
            "n_pages": parsed.n_pages,
            "chars_per_page": round(chars_per_page, 1),
            "needs_ocr": needs_ocr,
            "parsed_at": _utc_now(),
        }
        meta.setdefault("parsers", {})[parser_backend] = parser_info
        meta["n_pages"] = parsed.n_pages
        store.save_meta(paper_id, meta)
        store.save_text(paper_id, parsed.parser, text)
        index_key = PaperStore.make_index_key(
            parsed.parser, parsed.parser_version, chunk_tokens, chunk_overlap,
            embedder_name,
        )
        chunks = None

    # -- chunk (or reuse) -----------------------------------------------------
    chunks_from_cache = chunks is not None
    if chunks is None:
        chunks = chunk_document(
            text,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            is_markdown=parser_info["is_markdown"],
        )
        store.save_chunks(
            paper_id,
            index_key,
            chunks,
            config={
                "parser": {
                    k: parser_info[k]
                    for k in ("requested", "used", "version", "fallback_used")
                },
                "chunk_tokens": chunk_tokens,
                "chunk_overlap": chunk_overlap,
                "embedder": embedder_name,
                "embedder_max_seq_length": max_tokens,
                "n_chunks": len(chunks),
            },
        )

    # -- embed (or reuse) ------------------------------------------------------
    embeddings = (
        store.load_embeddings(paper_id, index_key, n_chunks=len(chunks))
        if (use_cache and chunks_from_cache)
        else None
    )
    embeddings_from_cache = embeddings is not None
    if embeddings is None:
        embeddings = np.asarray(
            embedder.encode(
                [c.embed_text for c in chunks],
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
        )
        store.save_embeddings(paper_id, index_key, embeddings)

    return PaperIndex(
        paper_id=paper_id,
        chunks=chunks,
        embeddings=embeddings,
        meta=meta,
        parser_info=parser_info,
        index_dir=store.index_dir(paper_id, index_key),
        chunks_from_cache=chunks_from_cache,
        embeddings_from_cache=embeddings_from_cache,
    )


# ---------------------------------------------------------------------------
# Command-line inspection: python -m pipeline.paper_store cache/ [paper_id]
# ---------------------------------------------------------------------------

def _print_paper_summary(store: PaperStore, paper_id: str, preview_chars: int) -> None:
    meta = store.load_meta(paper_id) or {}
    print(f"paper_id : {paper_id}")
    print(f"filename : {meta.get('filename', '?')}")
    print(f"pages    : {meta.get('n_pages', '?')}")
    for requested, parser in meta.get("parsers", {}).items():
        cleaning = parser.get("cleaning", {})
        print(
            f"parser   : {requested} -> {parser.get('used', '?')} "
            f"{parser.get('version', '')}"
            f"{' (FALLBACK)' if parser.get('fallback_used') else ''}   "
            f"chars/page: {parser.get('chars_per_page', '?')}   "
            f"needs_ocr: {parser.get('needs_ocr', '?')}"
        )
        print(
            f"cleaning : references_truncated={cleaning.get('references_truncated')}"
            f"  boilerplate_lines={len(cleaning.get('boilerplate_lines_removed', []))}"
            f"  tiny_fragments_dropped={cleaning.get('tiny_fragments_dropped')}"
        )
    for index_dir in sorted(p for p in store.paper_dir(paper_id).iterdir() if p.is_dir()):
        chunks = store.load_chunks(paper_id, index_dir.name) or []
        n_tables = sum(1 for c in chunks if c.is_table)
        sizes = [c.n_tokens for c in chunks] or [0]
        print(f"\nindex    : {index_dir.name}")
        print(
            f"chunks   : {len(chunks)} ({n_tables} tables), "
            f"tokens min/median/max = {min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}"
        )
        for chunk in chunks[: 3 if preview_chars else 0]:
            flag = " [TABLE]" if chunk.is_table else ""
            print(f"  --- chunk {chunk.chunk_id}{flag} ({chunk.n_tokens} tok) "
                  f"section='{chunk.section}'")
            print(f"      {chunk.text[:preview_chars]}")


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Inspect the paper cache (chunks, parser provenance, sizes)."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Cache directory (lists all papers) or cache/{paper_id} (detail view).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=200,
        help="Characters of chunk text to preview in detail view (0 to disable).",
    )
    args = parser.parse_args()

    path = args.path
    if (path / "meta.json").exists():
        store = PaperStore(path.parent)
        _print_paper_summary(store, path.name, args.preview_chars)
        return

    store = PaperStore(path)
    rows = []
    for paper_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        meta = store.load_meta(paper_dir.name) or {}
        parsers = ",".join(
            info.get("used", "?") for info in meta.get("parsers", {}).values()
        )
        n_indexes = sum(1 for p in paper_dir.iterdir() if p.is_dir())
        rows.append(
            (paper_dir.name, parsers or "?",
             str(meta.get("n_pages", "?")), str(n_indexes),
             meta.get("filename", "?"))
        )
    if not rows:
        print(f"No cached papers found in {path}")
        return
    print(f"{'paper_id':<18} {'parser(s)':<20} {'pages':>5} {'idx':>3}  filename")
    for row in rows:
        print(f"{row[0]:<18} {row[1]:<20} {row[2]:>5} {row[3]:>3}  {row[4]}")


if __name__ == "__main__":
    main()
