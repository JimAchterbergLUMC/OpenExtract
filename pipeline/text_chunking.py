"""
Structure-aware text chunking measured by the embedder's own tokenizer.

Chunks are built by packing whole sentences (grouped per paragraph) up to a
token budget. Sizes are measured with the *same tokenizer the retrieval
embedder uses*, so a chunk can never exceed the embedder's maximum sequence
length and get silently truncated at encoding time.

The tokenizer is used exclusively for counting; chunk text is always composed
from the original sentence strings, never reconstructed by decoding token IDs
(decoding would lowercase text for uncased models and mangle spacing).
"""

import re
from typing import List, Optional

# Common abbreviations in scientific prose whose trailing period does not end
# a sentence ("et al.", "Fig. 3", "e.g.", ...). Compared lowercased, without
# the period.
_ABBREVIATIONS = {
    "al", "fig", "figs", "eq", "eqs", "ref", "refs", "sec", "no", "vol",
    "pp", "etc", "vs", "cf", "ca", "resp", "approx", "dr", "prof", "st",
    "jr", "inc", "ltd",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


def _count_tokens(tokenizer, text: str) -> int:
    """Number of tokens the embedder's tokenizer produces for `text`
    (excluding special tokens such as [CLS]/[SEP])."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _ends_with_abbreviation(sentence: str) -> bool:
    """True if the sentence ends in a period that likely belongs to an
    abbreviation or a single initial rather than a sentence terminator."""
    match = re.search(r"([A-Za-z]+)\.$", sentence)
    if not match:
        return False
    word = match.group(1)
    return len(word) == 1 or word.lower() in _ABBREVIATIONS


def _split_sentences(paragraph: str) -> List[str]:
    """
    Split a paragraph into sentences with a lightweight regex heuristic.

    Splits after `.`, `!`, or `?` followed by whitespace, then repairs the
    most common false splits in scientific text: abbreviations ("et al.",
    "Fig."), single initials ("J."), and continuations that start lowercase.
    """
    parts = _SENTENCE_BOUNDARY.split(paragraph)
    sentences: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences and (
            _ends_with_abbreviation(sentences[-1]) or part[0].islower()
        ):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


def _split_long_sentence(sentence: str, budget: int, tokenizer) -> List[str]:
    """
    Split a sentence whose token count exceeds `budget` into pieces that each
    fit. Splits on whitespace; falls back to character slicing for a single
    unbroken blob (e.g. garbled PDF output without spaces).
    """
    words = sentence.split(" ")

    if len(words) == 1:
        # No whitespace to split on: slice by characters, verified per piece.
        pieces = []
        rest = sentence
        while rest:
            cut = len(rest)
            while cut > 1 and _count_tokens(tokenizer, rest[:cut]) > budget:
                cut = max(1, cut // 2)
            pieces.append(rest[:cut])
            rest = rest[cut:]
        return pieces

    pieces = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and _count_tokens(tokenizer, candidate) > budget:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))

    # A single word can itself exceed the budget; recurse into char slicing.
    result: List[str] = []
    for piece in pieces:
        if _count_tokens(tokenizer, piece) > budget:
            result.extend(_split_long_sentence(piece.replace(" ", ""), budget, tokenizer))
        else:
            result.append(piece)
    return result


def chunk_text(
    text: str,
    chunk_tokens: int,
    chunk_overlap: int,
    tokenizer,
    max_tokens: Optional[int] = None,
) -> List[str]:
    """
    Split text into chunks of whole sentences, sized by the embedder's tokenizer.

    Args:
        text: The input text to chunk.
        chunk_tokens: Target maximum tokens per chunk, counted with `tokenizer`.
        chunk_overlap: Approximate token overlap between consecutive chunks,
            realized as the trailing sentence(s) of the previous chunk.
            Clamped to at most half of chunk_tokens.
        tokenizer: The embedder's tokenizer (a Hugging Face tokenizer exposing
            `encode(text, add_special_tokens=False)`). Used only for counting.
        max_tokens: The embedder's maximum sequence length (e.g.
            `SentenceTransformer.max_seq_length`). When given, the effective
            budget is capped so that a chunk plus special tokens always fits;
            a warning is printed if this shrinks the requested chunk_tokens.

    Returns:
        List of text chunks. Every chunk is guaranteed to fit the budget as
        measured by `tokenizer`.
    """
    chunk_tokens = max(32, int(chunk_tokens))
    chunk_overlap = max(0, int(chunk_overlap))
    chunk_overlap = min(chunk_overlap, chunk_tokens // 2)

    budget = chunk_tokens
    if max_tokens is not None:
        try:
            n_special = tokenizer.num_special_tokens_to_add()
        except AttributeError:
            n_special = 2
        hard_cap = int(max_tokens) - n_special
        if budget > hard_cap:
            print(
                f"Warning: chunk_tokens={budget} exceeds the embedder's "
                f"window of {max_tokens} tokens; clamping chunks to "
                f"{hard_cap} tokens to avoid silent truncation."
            )
            budget = hard_cap

    # Collect sentences (paragraph by paragraph), pre-splitting any sentence
    # that alone exceeds the budget. Token counts are cached alongside.
    sentences: List[str] = []
    for paragraph in _PARAGRAPH_BOUNDARY.split(text):
        paragraph = " ".join(paragraph.split())  # normalize internal whitespace
        if not paragraph:
            continue
        for sentence in _split_sentences(paragraph):
            if _count_tokens(tokenizer, sentence) > budget:
                sentences.extend(_split_long_sentence(sentence, budget, tokenizer))
            else:
                sentences.append(sentence)

    # Greedy packing. Counting is done on the joined candidate string (exact,
    # regardless of how the tokenizer treats whitespace between sentences).
    chunks: List[str] = []
    current: List[str] = []
    has_new_content = False

    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and _count_tokens(tokenizer, candidate) > budget:
            if has_new_content:
                chunks.append(" ".join(current))
                # Carry trailing sentences as overlap into the next chunk.
                overlap: List[str] = []
                for prev in reversed(current):
                    extended = " ".join([prev] + overlap)
                    if _count_tokens(tokenizer, extended) > chunk_overlap:
                        break
                    overlap.insert(0, prev)
                current = overlap
            else:
                # Chunk holds only carried-over overlap; emitting it would
                # duplicate content. Drop it and start fresh.
                current = []
            has_new_content = False

            if current:
                candidate = " ".join(current + [sentence])
                if _count_tokens(tokenizer, candidate) > budget:
                    # Overlap plus this sentence does not fit; drop overlap.
                    current = []

        current.append(sentence)
        has_new_content = True

    if current and has_new_content:
        chunks.append(" ".join(current))

    return chunks
