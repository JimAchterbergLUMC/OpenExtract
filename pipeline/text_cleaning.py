"""
Post-extraction text cleaning, applied before chunking regardless of parser.

Steps (in order):
1. Strip repeated headers/footers: lines recurring (near-)identically on a
   large share of pages are boilerplate (journal banners, page numbers,
   "Authorized licensed use..." watermarks). Requires per-page text.
2. Normalize glyphs: Unicode NFKC (resolves ligatures like fi/fl), soft
   hyphens, exotic spaces; collapse space runs within lines.
3. De-hyphenate line breaks: join `word-\nword` when both sides are
   lowercase letters.
4. Truncate the bibliography: cut everything after the last
   References/Bibliography heading in the latter part of the document.
5. Drop tiny fragments: isolated letterless or 1-3 word remnant lines
   (figure axis labels, stray numbers).

`clean_parsed_paper` runs all steps and returns the cleaned text plus a
report dict (stored in the cache's meta.json) describing what was removed.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from .pdf_parsing import ParsedPaper

# Share of pages on which a (normalized) line must recur to count as boilerplate.
_BOILERPLATE_PAGE_FRACTION = 0.4
_BOILERPLATE_MAX_LINE_CHARS = 200

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*$")
_BOLD_LINE_RE = re.compile(r"^\s*\*\*(?P<title>[^*]+)\*\*\s*$")
_REFERENCES_TITLE_RE = re.compile(
    r"^\s*(?:[IVXLC\d]+\.?\s*)?(references|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$",
    re.IGNORECASE,
)

_SOFT_HYPHEN = "\u00ad"
_LINE_BREAK_HYPHEN_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _normalize_line_for_matching(line: str) -> str:
    """Fold a line to a form where page headers/footers from different pages
    compare equal: lowercase, digits removed (page numbers, dates), and
    non-alphanumeric characters removed."""
    folded = line.casefold()
    folded = re.sub(r"\d+", "", folded)
    folded = re.sub(r"[^a-z]+", "", folded)
    return folded


def strip_repeated_page_lines(
    pages: List[str],
    min_fraction: float = _BOILERPLATE_PAGE_FRACTION,
) -> Tuple[List[str], List[str]]:
    """
    Remove lines that recur (near-)identically on >= `min_fraction` of pages.

    Returns:
        (cleaned pages, list of example boilerplate lines removed)
    """
    if len(pages) < 3:
        return pages, []

    # Count on how many *distinct pages* each normalized line occurs.
    page_counts: Dict[str, int] = {}
    examples: Dict[str, str] = {}
    for page in pages:
        seen_on_page = set()
        for line in page.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > _BOILERPLATE_MAX_LINE_CHARS:
                continue
            key = _normalize_line_for_matching(stripped)
            if not key or key in seen_on_page:
                continue
            seen_on_page.add(key)
            page_counts[key] = page_counts.get(key, 0) + 1
            examples.setdefault(key, stripped)

    # Floor, not ceil: headers/footers often alternate between two variants
    # (recto/verso) or lose a page to OCR, landing just under an exact 40%.
    threshold = max(2, int(min_fraction * len(pages)))
    boilerplate_keys = {k for k, n in page_counts.items() if n >= threshold}
    if not boilerplate_keys:
        return pages, []

    cleaned_pages = []
    for page in pages:
        kept = [
            line
            for line in page.splitlines()
            if not (
                line.strip()
                and len(line.strip()) <= _BOILERPLATE_MAX_LINE_CHARS
                and _normalize_line_for_matching(line.strip()) in boilerplate_keys
            )
        ]
        cleaned_pages.append("\n".join(kept))

    removed_examples = sorted(examples[k] for k in boilerplate_keys)
    return cleaned_pages, removed_examples


def normalize_glyphs(text: str) -> str:
    """NFKC normalization (resolves fi/fl ligatures etc.), soft-hyphen
    removal, and collapse of horizontal whitespace runs."""
    text = text.replace(_SOFT_HYPHEN, "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = _SPACE_RUN_RE.sub(" ", text)
    return text


def dehyphenate_line_breaks(text: str) -> str:
    """Join words split by end-of-line hyphenation (`classifi-\\ncation`).
    Only merges when a lowercase letter sits on both sides of the break,
    which avoids mangling genuine hyphenated compounds and equations."""
    return _LINE_BREAK_HYPHEN_RE.sub("", text)


def _is_references_heading(line: str) -> bool:
    match = _HEADING_RE.match(line) or _BOLD_LINE_RE.match(line)
    title = match.group("title") if match else line
    title = title.strip().strip("*_ ")
    return bool(_REFERENCES_TITLE_RE.match(title))


def truncate_bibliography(
    text: str, min_position_fraction: float = 0.3
) -> Tuple[str, bool]:
    """
    Cut everything from the last References/Bibliography heading onward.

    The heading must sit past `min_position_fraction` of the document, which
    protects against tables of contents and in-text mentions near the top.
    Appendices *preceding* the references are kept. Works on Markdown
    headings, bold-only lines, and bare "REFERENCES" lines (pypdf output).

    Returns:
        (possibly truncated text, whether a cut was made)
    """
    lines = text.splitlines()
    min_offset = int(len(text) * min_position_fraction)

    offset = 0
    cut_at: Optional[int] = None
    for i, line in enumerate(lines):
        if offset >= min_offset and _is_references_heading(line):
            cut_at = i
        offset += len(line) + 1

    if cut_at is None:
        return text, False
    return "\n".join(lines[:cut_at]).rstrip() + "\n", True


def drop_tiny_fragments(text: str) -> Tuple[str, int]:
    """
    Remove isolated remnant lines: lines with no letters at all (stray page
    numbers, axis ticks) or 1-3 word fragments without sentence punctuation.
    Headings, list items, and table rows are always kept.

    Returns:
        (cleaned text, number of lines dropped)
    """
    kept: List[str] = []
    dropped = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        is_structural = (
            stripped.startswith(("#", "|", "-", "*", ">"))
            or re.match(r"^\d+[.)]\s", stripped) is not None
        )
        if is_structural:
            kept.append(line)
            continue
        has_letters = re.search(r"[A-Za-z]", stripped) is not None
        n_words = len(stripped.split())
        ends_sentence = stripped[-1] in ".!?:;,"
        if not has_letters or (n_words <= 3 and not ends_sentence and len(stripped) < 40):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def clean_parsed_paper(parsed: ParsedPaper) -> Tuple[str, Dict]:
    """
    Run the full cleaning pipeline on a parse result.

    Returns:
        (cleaned text, report dict for meta.json)
    """
    report: Dict = {}

    if parsed.pages is not None:
        pages, removed = strip_repeated_page_lines(parsed.pages)
        text = "\n\n".join(pages)
        report["boilerplate_lines_removed"] = removed
    else:
        text = parsed.text
        report["boilerplate_lines_removed"] = []

    raw_chars = len(text)
    text = normalize_glyphs(text)
    text = dehyphenate_line_breaks(text)
    text, references_cut = truncate_bibliography(text)
    report["references_truncated"] = references_cut

    text, fragments_dropped = drop_tiny_fragments(text)
    report["tiny_fragments_dropped"] = fragments_dropped

    text = _BLANK_RUN_RE.sub("\n\n", text).strip() + "\n"
    report["chars_before_cleaning"] = raw_chars
    report["chars_after_cleaning"] = len(text)
    return text, report
