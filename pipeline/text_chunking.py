"""
Text chunking utilities for document processing.

This module provides functionality for splitting large text documents into
smaller chunks for processing by language models, with support for both
token-based and character-based chunking methods.
"""

from typing import List

# Optional dependency handling
try:
    import tiktoken  # type: ignore
except ImportError:
    tiktoken = None


def _split_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    model_name: str = "gpt-4o-mini",
) -> List[str]:
    """
    Split text into chunks based on token count.
    
    Args:
        text: The input text to split
        chunk_tokens: Maximum number of tokens per chunk
        overlap_tokens: Number of tokens to overlap between consecutive chunks
        model_name: The model name to use for tokenization (if tiktoken is available)
        
    Returns:
        List of text chunks
        
    Note:
        If tiktoken is not available, falls back to character-based estimation
        (4 characters ≈ 1 token).
    """
    if tiktoken is None:
        # Fallback to character-based chunking
        approx_chars = chunk_tokens * 4
        approx_overlap = overlap_tokens * 4
        chunks = []
        i = 0
        n = len(text)
        
        while i < n:
            end = min(n, i + approx_chars)
            chunks.append(text[i:end])
            if end == n:
                break
            i = max(i + approx_chars - approx_overlap, end)
        return chunks

    # Use tiktoken for precise token-based chunking
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        # Fallback to a default encoding if model is not recognized
        enc = tiktoken.get_encoding("o200k_base")
    
    ids = enc.encode(text)
    chunks = []
    i = 0
    n = len(ids)
    
    while i < n:
        end = min(n, i + chunk_tokens)
        chunk_ids = ids[i:end]
        chunks.append(enc.decode(chunk_ids))
        
        if end == n:
            break
            
        # Calculate next starting position with overlap
        i = i + chunk_tokens - overlap_tokens
        if i <= 0:  # Ensure we make progress
            i = end
    
    return chunks


def chunk_text(
    text: str,
    chunk_tokens: int,
    chunk_overlap: int,
    model_name: str,
) -> List[str]:
    """
    Split text into overlapping chunks with specified token limits.
    
    Args:
        text: The input text to chunk
        chunk_tokens: Maximum number of tokens per chunk (minimum 128)
        chunk_overlap: Number of tokens to overlap between chunks
        model_name: Model name for tokenization
        
    Returns:
        List of text chunks
        
    Note:
        - chunk_tokens is clamped to a minimum of 128
        - chunk_overlap is clamped to be non-negative and at most half of chunk_tokens
    """
    # Validate and clamp parameters
    chunk_tokens = max(128, int(chunk_tokens))
    chunk_overlap = max(0, int(chunk_overlap))
    chunk_overlap = min(chunk_overlap, chunk_tokens // 2)
    
    return _split_by_tokens(text, chunk_tokens, chunk_overlap, model_name)
