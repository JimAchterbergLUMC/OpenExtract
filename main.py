#!/usr/bin/env python3
"""
Paper Q&A (per PDF) via OpenRouter — dense retrieval + per-question choices (ID-based)
-------------------------------------------------------------------------------------

Behavior
- If a question includes "choices", the model must output exactly one *choice ID* from that list.
  Supported forms:
    1) ["tabular","time-series","images","text","video","audio"]   (IDs are the strings)
    2) [{"id":"ts","label":"time-series"}, {"id":"img","label":"images"}, ...]
- If a question has no "choices", the model answers in concise free text.
- If evidence isn't in the retrieved context, the model must output: "Unknown from this paper".

Retrieval
- Dense semantic retrieval only (SentenceTransformers).
- Cosine similarity via L2-normalized embeddings.

Requirements:
  pip install pypdf tiktoken requests sentence-transformers numpy

Usage:
  export OPENROUTER_API_KEY=...  # or use --api-key-file path/to/key.txt
  python main.py \
    --papers-dir ./papers \
    --questions-file ./questions_free_text.json \
    --output-dir ./answers_free_text \
    --model "qwen/qwen2.5-vl-32b-instruct:free" \
    --chunk-tokens 800 --chunk-overlap 160 --top-k 3 \
    --dense-model "thenlper/gte-small" --dense-batch-size 8
"""

import argparse
import json
import os
import random
from pathlib import Path

from utils import answer_questions_for_paper, load_questions


def main() -> None:
    """Main CLI interface for the Paper Q&A system."""
    parser = argparse.ArgumentParser(
        description="Answer JSON questions per PDF using OpenRouter (dense retrieval)."
    )

    # Required arguments
    parser.add_argument(
        "--papers-dir", required=True, type=Path, help="Directory with PDF files."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write answers."
    )
    parser.add_argument(
        "--questions-file", required=True, type=Path, help="JSON with questions."
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-chat-v3.1:free",
        help="OpenRouter model id.",
    )

    # Retrieval parameters
    parser.add_argument(
        "--top-k", type=int, default=3, help="# of chunks to retrieve per question."
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=800,
        help="Approx tokens per chunk (or char/4).",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=160, help="Overlap tokens between chunks."
    )

    # API configuration
    parser.add_argument(
        "--referer", default=None, help="Optional HTTP-Referer header for OpenRouter."
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Optional file with the OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.",
    )

    # Dense retrieval parameters
    parser.add_argument(
        "--dense-model",
        default="kamalkraj/BioSimCSE-BioLinkBERT-BASE",
        help="SentenceTransformers embedding model.",
    )
    parser.add_argument(
        "--dense-device",
        default=None,
        help="Force device for dense model: 'cpu' or 'cuda'. Default: auto.",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=8,
        help="Batch size for encoding chunks.",
    )

    # Random selection parameters
    parser.add_argument(
        "--random-subset",
        type=int,
        default=None,
        help="Randomly select this many papers from the papers directory. If not specified, use all papers.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible paper selection (default: 42).",
    )

    args = parser.parse_args()

    # Get API key
    if args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Provide OpenRouter API key via --api-key-file or OPENROUTER_API_KEY env var."
        )

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions = load_questions(args.questions_file)

    # Get list of PDF papers
    all_papers = sorted(
        [p for p in args.papers_dir.iterdir() if p.suffix.lower() == ".pdf"]
    )
    if not all_papers:
        raise RuntimeError(f"No PDFs found in {args.papers_dir}")

    # Apply random selection if specified
    if args.random_subset is not None:
        if args.random_subset <= 0:
            raise RuntimeError("--random-subset must be a positive integer")
        if args.random_subset >= len(all_papers):
            print(
                f"Warning: Requested {args.random_subset} papers but only {len(all_papers)} available. Using all papers."
            )
            papers = all_papers
        else:
            random.seed(args.random_seed)
            papers = sorted(random.sample(all_papers, args.random_subset))
            print(
                f"Randomly selected {len(papers)} papers out of {len(all_papers)} (seed: {args.random_seed})"
            )
    else:
        papers = all_papers

    # Process each paper
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
            dense_model=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
        )

        # Generate clean filename for output
        clean_name = pdf.stem.replace(" ", "_").replace("-", "_")
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in "._-")
        output_file = args.output_dir / f"{clean_name}_answers.json"

        # Save results
        result = {"paper": pdf.name, "answers": answers}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
