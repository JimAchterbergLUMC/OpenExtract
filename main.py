import argparse
import json
import os
import random
from pathlib import Path
from time import time

from pipeline import answer_questions_for_paper, load_questions


def main() -> None:
    """Main CLI interface for the Paper Q&A system."""
    parser = argparse.ArgumentParser(
        description="Answer JSON questions per PDF using OpenRouter (dense retrieval)."
    )

    # Required arguments
    parser.add_argument(
        "--papers-dir", default="./papers", type=Path, help="Directory with PDF files."
    )
    parser.add_argument(
        "--output-dir",
        default="./answers",
        type=Path,
        help="Directory to write answers.",
    )
    parser.add_argument(
        "--questions-file",
        default="./questions.json",
        type=Path,
        help="JSON with questions.",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="qwen/qwen-2.5-7b-instruct",  # qwen/qwen-2.5-7b-instruct, #deepseek/deepseek-chat-v3.1
        help="OpenRouter model id.",
    )

    # Retrieval parameters
    parser.add_argument(
        "--top-k", type=int, default=3, help="# of chunks to retrieve per question."
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1000,
        help="Approx tokens per chunk (or char/4).",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=500, help="Overlap tokens between chunks."
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
        default="neuml/pubmedbert-base-embeddings",
        help="SentenceTransformers embedding model (from Hugging Face).",
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

    # Structured output parameters
    parser.add_argument(
        "--use-structured-output",
        type=bool,
        default=False,
        help="Force OpenRouter's structured output functionality. Not every model supports this.",
    )

    parser.add_argument(
        "--stop-after-n-papers",
        type=int,
        default=None,
        help="Stop after processing the first n papers.",
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
    all_papers = sorted(Path(args.papers_dir).glob("*.pdf"))
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
    args.stop_after_n_papers = (
        len(papers) if args.stop_after_n_papers is None else args.stop_after_n_papers
    )
    for i, pdf in enumerate(papers):
        if i >= args.stop_after_n_papers:
            break
        print(f"Processing: {pdf.name}")
        start_time = time()
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
            use_structured_output=args.use_structured_output,
        )
        runtime = time() - start_time
        print(f"Time taken for {pdf.name}: {runtime} seconds")

        # Generate clean filename for output
        clean_name = pdf.stem.replace(" ", "_").replace("-", "_")
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in "._-")[
            :50
        ]  # ensure filenames are not too long
        output_file = args.output_dir / f"{clean_name}_answers.json"

        # Save results
        result = {"paper": pdf.name, "answers": answers, "runtime": runtime}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
