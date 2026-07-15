"""
Question-answering orchestration for paper analysis.

This module provides the main orchestration logic for processing papers
and answering questions using dense retrieval and language models.
"""

import ast
import string
from pathlib import Path
from typing import Dict, List, Optional

from .choice_utils import extract_choice_id, normalize_choices
from .data_models import Question
from .openrouter_client import call_openrouter
from .pdf_utils import extract_pdf_text
from .retrieval import DenseRetriever, get_sentence_transformer, resolve_device
from .text_chunking import chunk_text

UNKNOWN_ANSWER = "Unknown from this paper"


def _resolve_choice_id(candidate: str, valid_ids: List[str]) -> Optional[str]:
    """
    Map a model-emitted token (candidate) to a valid ID.

    Rules (in order):
    - Exact match (case-sensitive) or case-insensitive match.
    - If candidate has no colon, try matching against the prefix before ':' in valid IDs.
    - If candidate is a prefix of exactly one valid ID, use that ID
      (e.g. 'C' -> 'C00-D48: Neoplasms', 'Random Forest' -> 'Random Forest: ...').
    - As a last resort, if candidate is a letter (A, B, C, ...), map to the
      Nth valid_id in order.
    - Return None if ambiguous or not found.
    """
    if not isinstance(candidate, str):
        candidate = str(candidate)

    cand = candidate.strip()
    if not cand:
        return None

    # 1) Exact match
    if cand in valid_ids:
        return cand

    # 2) Case-insensitive exact match
    for vid in valid_ids:
        if cand.casefold() == vid.casefold():
            return vid

    # 3) Prefix match if candidate has no colon
    if ":" not in cand:
        base_to_fulls = {}
        for vid in valid_ids:
            base = vid.split(":", 1)[0]
            base_to_fulls.setdefault(base.casefold(), []).append(vid)

        matches = base_to_fulls.get(cand.casefold(), [])
        if len(matches) == 1:
            return matches[0]

    # 4) Unique prefix match: candidate abbreviates exactly one valid ID.
    # This must run before the positional letter mapping so that e.g. 'C'
    # resolves to 'C00-D48: Neoplasms' instead of the third choice in the list.
    prefix_matches = [
        vid for vid in valid_ids if vid.casefold().startswith(cand.casefold())
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    # 5) Last resort: single letter mapping (A, B, C...) -> index into valid_ids
    upper_cand = cand.upper()
    if len(upper_cand) == 1 and upper_cand in string.ascii_uppercase:
        idx = ord(upper_cand) - ord("A")
        if 0 <= idx < len(valid_ids):
            return valid_ids[idx]

    return None


def answer_questions_for_paper(
    pdf_path: Path,
    questions: List[Question],
    model: str,
    api_key: str,
    chunk_tokens: int,
    chunk_overlap: int,
    top_k: int,
    referer: Optional[str],
    dense_model: str,
    dense_device: Optional[str],
    dense_batch_size: int,
    use_structured_output: Optional[bool] = False,
) -> List[Dict]:
    """
    Process a single PDF paper and answer all questions for it.

    Args:
        pdf_path: Path to the PDF file to process
        questions: List of Question objects to answer
        model: Language model identifier for OpenRouter
        api_key: OpenRouter API key
        chunk_tokens: Maximum tokens per text chunk, measured with the dense
            embedder's tokenizer (capped at the embedder's max sequence length)
        chunk_overlap: Token overlap between consecutive chunks
        top_k: Number of top chunks to retrieve for each question
        referer: Optional HTTP referer for API requests
        dense_model: Sentence transformer model for dense retrieval
        dense_device: Device for dense model ('cpu', 'cuda', or None for auto)
        dense_batch_size: Batch size for encoding operations

    Returns:
        List of answer dictionaries, each containing:
        - id: Question ID
        - question: Question text
        - answer: Final answer (choice ID or free text)
        - raw_answer: Raw model output (for choice questions)
        - choices_ids: Available choice IDs (for choice questions)
        - answer_label: Human-readable answer label (for choice questions)
        - chunks_id: Indices of retrieved chunks
        - chunks_str: Text of retrieved chunks
        - sent_transformer: Dense retrieval model used
        - LLM: Language model used (if available in logs)
        - finish_reason: API response finish reason (if available)
        - total_len: Total tokens used (if available in logs)

    Note:
        If the PDF cannot be read or contains no text, returns "Unknown from this paper"
        for all questions.
    """
    # Extract text from PDF
    raw_text = extract_pdf_text(pdf_path)
    if not raw_text.strip():
        # Return default answers if no text could be extracted
        return [
            {"id": q.id, "question": q.text, "answer": UNKNOWN_ANSWER}
            for q in questions
        ]

    # Load the embedder first: its tokenizer measures chunk sizes and its
    # max sequence length caps them, so chunks can never be silently
    # truncated at encoding time. (The model is cached; DenseRetriever below
    # reuses the same instance.)
    device = resolve_device(dense_device)
    embedder = get_sentence_transformer(dense_model, device)

    # Split text into chunks on sentence/paragraph boundaries
    chunks = chunk_text(
        raw_text,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        tokenizer=embedder.tokenizer,
        max_tokens=getattr(embedder, "max_seq_length", None),
    )

    # Initialize dense retriever once per paper
    retriever = DenseRetriever(
        chunks=chunks,
        model_name=dense_model,
        device=device,
        batch_size=dense_batch_size,
    )

    answers = []
    paper_title = pdf_path.name

    for question in questions:
        print(f"Processing question: {question.id}")

        # Retrieve most relevant chunks for this question
        chunk_scores = retriever.top_k(question.text, k=top_k)
        selected_chunks = [chunks[idx] for idx, _ in chunk_scores]

        # Prepare choice information for multiple-choice questions
        question_choice_ids: Optional[List[str]] = None
        question_id_to_label: Optional[Dict[str, str]] = None

        if question.choices and len(question.choices) > 0:
            choice_ids, id_to_label = normalize_choices(question.choices)
            if choice_ids:
                question_choice_ids = choice_ids
                question_id_to_label = id_to_label

        # Get answer from language model
        raw_answer, api_logs = call_openrouter(
            model=model,
            question=question.text,
            paper_title=paper_title,
            context_chunks=selected_chunks,
            api_key=api_key,
            referer=referer,
            x_title="Paper QA",
            choice_ids=question_choice_ids,
            id_to_label=question_id_to_label,
            use_structured_output=use_structured_output,
        )

        # API failure: record it explicitly instead of parsing a pseudo-answer
        if raw_answer is None:
            answers.append(
                {
                    "id": question.id,
                    "question": question.text,
                    "answer": UNKNOWN_ANSWER,
                    "error": "api_request_failed",
                    "raw_answer": None,
                    "choices_ids": question_choice_ids,
                    "answer_label": None,
                    "chunks_id": [idx for idx, _ in chunk_scores],
                    "chunks_str": selected_chunks,
                    "sent_transformer": dense_model,
                }
            )
            continue

        # Process the answer based on question type
        if question_choice_ids:
            valid_list = list(question_choice_ids)  # preserve order

            def _normalize_many(candidates: List[str]) -> List[str]:
                seen = set()
                out: List[str] = []
                for c in candidates:
                    resolved = _resolve_choice_id(str(c), valid_list)
                    if resolved is not None and resolved not in seen:
                        seen.add(resolved)
                        out.append(resolved)
                return out

            print(f"raw_answer: {raw_answer}")

            # Must be reset for every question; otherwise an unparseable reply
            # would silently reuse the previous question's answer.
            final_ids: Optional[List[str]] = None

            # Explicit abstention (check the original reply, before any slicing)
            is_unknown = (
                raw_answer.strip().rstrip(".").casefold() == UNKNOWN_ANSWER.casefold()
            )

            if not is_unknown:
                # Extract the array portion of the reply, e.g. '["A","C"]'
                start = raw_answer.find("[")
                end = raw_answer.find("]", start)
                list_str = (
                    raw_answer[start : end + 1] if start != -1 and end != -1 else ""
                )

                parsed = None
                if list_str:
                    try:
                        parsed = ast.literal_eval(list_str)
                    except (ValueError, SyntaxError):
                        parsed = None

                if isinstance(parsed, list):
                    final_ids = _normalize_many(parsed)
                else:
                    # Reply was not a list; look for a valid ID in the full reply
                    # (handles prose like "The answer is Random Forest.")
                    resolved = extract_choice_id(raw_answer, valid_list)
                    if resolved is None:
                        # Models often abbreviate 'Random Forest: ...' to just
                        # 'Random Forest'; search for unambiguous pre-colon bases.
                        base_to_full: Dict[str, str] = {}
                        for vid in valid_list:
                            base = vid.split(":", 1)[0].strip()
                            # Keep only bases that map to a single full ID
                            base_to_full[base] = (
                                vid if base not in base_to_full else None
                            )
                        bases = [b for b, v in base_to_full.items() if v]
                        base_match = extract_choice_id(raw_answer, bases)
                        if base_match is not None:
                            resolved = base_to_full[base_match]
                    if resolved is not None:
                        final_ids = [resolved]

                print(f"parsed answer: {parsed}")

            if not final_ids:
                final_answer = UNKNOWN_ANSWER
                answer_labels = None
            else:
                final_answer = final_ids  # <-- list of IDs
                answer_labels = (
                    [question_id_to_label[i] for i in final_ids]
                    if question_id_to_label
                    else None
                )
        else:
            final_answer = raw_answer
            answer_labels = None

        # Construct answer record with all relevant information
        answer_record = {
            "id": question.id,
            "question": question.text,
            "answer": final_answer,
            "raw_answer": raw_answer if question_choice_ids else None,
            "choices_ids": question_choice_ids if question_choice_ids else None,
            "answer_label": answer_labels,  # <-- now a list for MCQ, or None
            "chunks_id": [idx for idx, _ in chunk_scores],
            "chunks_str": [chunks[idx] for idx, _ in chunk_scores],
            "sent_transformer": dense_model,
        }

        # Add API logging information if available
        if api_logs:
            try:
                answer_record["LLM"] = api_logs["model"]
                answer_record["finish_reason"] = api_logs["choices"][0]["finish_reason"]
                answer_record["total_len"] = api_logs["usage"]["total_tokens"]
            except (KeyError, IndexError):
                # If logging information is malformed, continue without it
                pass

        answers.append(answer_record)

    return answers
