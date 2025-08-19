from __future__ import annotations

#!/usr/bin/env python3
"""
Paper Q&A (per PDF) via OpenRouter
----------------------------------

Given a directory of research-paper PDFs and a JSON file of questions, this script:
  1) extracts text from each PDF,
  2) chunks the text (token- or char-approx),
  3) retrieves the top-k relevant chunks per question using TF‑IDF similarity,
  4) calls an OpenRouter chat completion model with only those chunks as context,
  5) writes answers per paper to a JSON file.

Why chunking? You can control chunk size to keep latency/cost reasonable. By default we
retrieve only a few chunks per question; the model never sees the full paper unless you set a very large chunk size and k.

Requirements (install as needed):
  pip install pypdf scikit-learn tiktoken requests

Usage:
  export OPENROUTER_API_KEY=...  # required
  python main.py --papers-dir ./papers --questions-file ./questions.json --output-dir ./answers --model "qwen/qwen2.5-vl-32b-instruct:free" --chunk-tokens 800 --chunk-overlap 160 --top-k 3

Notes:
- Models on OpenRouter have different context windows. Keep (top_k * chunk_tokens) conservative.
- If tiktoken is not available, we fall back to a simple char-based splitter (~4 chars/token heuristic).
- Answers are constrained to the provided chunks; if not found, the assistant should reply "Unknown from this paper".
"""


import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# Optional deps
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # fallback to char-based splitting

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # very minimal fallback if pypdf is missing
    PdfReader = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["OPENROUTER_API_KEY"] = open("api_keys/openrouter.txt", "r").read().strip()


@dataclass
class Question:
    id: str
    text: str


def load_questions(path: Path) -> List[Question]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    qs = []
    for q in raw.get("questions", []):
        qs.append(Question(id=q["id"], text=q["text"]))
    return qs


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf. (Good enough for most research PDFs.)"""
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is not installed. Run `pip install pypdf` or install pdfplumber/pdfminer."
        )
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    # Basic cleanup
    text = text.replace("\u00ad", "")  # soft hyphens
    return text


# ------------------ Chunking ------------------


def _tokenize_len(s: str, model_name: str = "gpt-4o-mini") -> int:
    if tiktoken is None:
        # Rough heuristic: ~4 char per token
        return max(1, len(s) // 4)
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(s))


def _split_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    model_name: str = "gpt-4o-mini",
) -> List[str]:
    if tiktoken is None:
        # char-based approx: assume 4 chars/token; convert to char window
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
            i = max(i + approx_chars - approx_overlap, end)  # prevent infinite loop
        return chunks

    # token-true splitting
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
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
        i = i + chunk_tokens - overlap_tokens
        if i <= 0:
            i = end  # safety
    return chunks


def chunk_text(
    text: str,
    chunk_tokens: int,
    chunk_overlap: int,
    model_name: str,
) -> List[str]:
    chunk_tokens = max(128, int(chunk_tokens))
    chunk_overlap = max(0, int(chunk_overlap))
    chunk_overlap = min(chunk_overlap, chunk_tokens // 2)  # sanity
    return _split_by_tokens(text, chunk_tokens, chunk_overlap, model_name)


# ------------------ Retrieval ------------------


def build_retriever(chunks: List[str]) -> Tuple[TfidfVectorizer, any]:
    vectorizer = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, ngram_range=(1, 2)
    )
    mat = vectorizer.fit_transform(chunks)
    return vectorizer, mat


def top_k_chunks(
    query: str,
    chunks: List[str],
    vectorizer: TfidfVectorizer,
    chunk_mat,
    k: int = 3,
) -> List[Tuple[int, float]]:
    qvec = vectorizer.transform([query])
    sims = cosine_similarity(qvec, chunk_mat)[0]
    idx_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
    return idx_scores


# ------------------ OpenRouter call ------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(
    model: str,
    question: str,
    paper_title: str,
    context_chunks: List[str],
    api_key: str,
    referer: Optional[str] = None,
    x_title: Optional[str] = "Paper QA",
) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a meticulous research assistant. Answer the user's question strictly "
        "from the provided paper excerpts (context). If the answer is not present in the "
        "context, reply exactly: 'Unknown from this paper'. Be concise and specific. Make replies as short as possible, not in full sentences."
    )

    user_prompt = (
        f"Paper: {paper_title}\n\n"
        f"Question: {question}\n\n"
        f"Context (selected excerpts):\n{context}\n\n"
        "Return only the answer (one or two sentences)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if x_title:
        headers["X-Title"] = x_title

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    for attempt in range(3):
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                return "Unknown from this paper"
        # simple backoff
        time.sleep(1.5 * (attempt + 1))
    # last resort
    try:
        err = resp.text[:500]
    except Exception:
        err = ""
    return f"Unknown from this paper"  # do not leak errors into answer body


# ------------------ Orchestration ------------------


def answer_questions_for_paper(
    pdf_path: Path,
    questions: List[Question],
    model: str,
    api_key: str,
    chunk_tokens: int,
    chunk_overlap: int,
    top_k: int,
    referer: Optional[str],
) -> List[dict]:
    raw_text = extract_pdf_text(pdf_path)
    if not raw_text.strip():
        return [
            {"id": q.id, "question": q.text, "answer": "Unknown from this paper"}
            for q in questions
        ]

    chunks = chunk_text(raw_text, chunk_tokens, chunk_overlap, model_name=model)
    # Build retriever once per paper
    vectorizer, mat = build_retriever(chunks)

    answers = []
    title = pdf_path.name
    for q in questions:
        idx_scores = top_k_chunks(q.text, chunks, vectorizer, mat, k=top_k)
        selected = [chunks[idx] for idx, _ in idx_scores]
        answer = call_openrouter(
            model=model,
            question=q.text,
            paper_title=title,
            context_chunks=selected,
            api_key=api_key,
            referer=referer,
            x_title="Paper QA",
        )
        answers.append(
            {
                "id": q.id,
                "question": q.text,
                "answer": answer,
                "chunks_used": [idx for idx, score in idx_scores],
            }
        )
    return answers


def main():
    parser = argparse.ArgumentParser(
        description="Answer JSON questions per PDF using OpenRouter."
    )
    parser.add_argument(
        "--papers-dir", required=True, type=Path, help="Directory containing PDF files."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where to write individual answer files.",
    )

    parser.add_argument(
        "--questions-file",
        required=True,
        type=Path,
        help="Path to JSON file with questions.",
    )

    parser.add_argument(
        "--model", default="openai/gpt-4o-mini", help="OpenRouter model id."
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="# of chunks to retrieve per question."
    )

    # Chunking controls
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=800,
        help="Approx tokens per chunk (or char/4 if no tiktoken).",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=160, help="Overlap tokens between chunks."
    )

    parser.add_argument(
        "--referer", default=None, help="Optional HTTP-Referer header for OpenRouter."
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in your environment.")

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.questions_file)

    papers = sorted(
        [p for p in args.papers_dir.iterdir() if p.suffix.lower() == ".pdf"]
    )
    if not papers:
        raise RuntimeError(f"No PDFs found in {args.papers_dir}")

    for pdf in papers:
        print(f"Processing: {pdf.name}")
        answers = answer_questions_for_paper(
            pdf_path=pdf,
            questions=questions,
            model=args.model,
            api_key=api_key,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            referer=args.referer,
        )

        # Clean paper name for filename (remove spaces and special characters)
        clean_name = pdf.stem.replace(" ", "_").replace("-", "_")
        # Remove any other potentially problematic characters
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in "._-")

        output_file = args.output_dir / f"{clean_name}_answers.json"

        result = {
            "paper": pdf.name,
            "answers": answers,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
