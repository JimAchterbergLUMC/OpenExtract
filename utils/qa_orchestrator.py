"""
Question-answering orchestration for paper analysis.

This module provides the main orchestration logic for processing papers
and answering questions using dense retrieval and language models.
"""

from pathlib import Path
from typing import Dict, List, Optional

from .choice_utils import extract_choice_id, normalize_choices
from .data_models import Question
from .openrouter_client import call_openrouter
from .pdf_utils import extract_pdf_text
from .retrieval import DenseRetriever
from .text_chunking import chunk_text


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
) -> List[Dict]:
    """
    Process a single PDF paper and answer all questions for it.
    
    Args:
        pdf_path: Path to the PDF file to process
        questions: List of Question objects to answer
        model: Language model identifier for OpenRouter
        api_key: OpenRouter API key
        chunk_tokens: Maximum tokens per text chunk
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
            {"id": q.id, "question": q.text, "answer": "Unknown from this paper"}
            for q in questions
        ]
    
    # Split text into chunks
    chunks = chunk_text(raw_text, chunk_tokens, chunk_overlap, model_name=model)
    
    # Initialize dense retriever once per paper
    retriever = DenseRetriever(
        chunks=chunks,
        model_name=dense_model,
        device=dense_device,
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
        )
        
        # Process the answer based on question type
        if question_choice_ids:
            # Multiple choice: extract valid choice ID or default to "Unknown"
            extracted_choice = extract_choice_id(raw_answer, question_choice_ids)
            final_answer = extracted_choice if extracted_choice is not None else "Unknown from this paper"
        else:
            # Free text: use raw answer directly
            final_answer = raw_answer
        
        # Construct answer record with all relevant information
        answer_record = {
            "id": question.id,
            "question": question.text,
            "answer": final_answer,
            "raw_answer": raw_answer if question_choice_ids else None,
            "choices_ids": question_choice_ids if question_choice_ids else None,
            "answer_label": (
                question_id_to_label[final_answer] 
                if question_id_to_label and final_answer in question_id_to_label 
                else None
            ),
            "chunks_id": [idx for idx, _ in chunk_scores],
            "chunks_str": [chunks[idx] for idx, _ in chunk_scores],
            "sent_transformer": dense_model,
        }
        
        # Add API logging information if available
        if api_logs:
            try:
                answer_record["LLM"] = api_logs['model']
                answer_record["finish_reason"] = api_logs['choices'][0]['finish_reason']
                answer_record["total_len"] = api_logs['usage']['total_tokens']
            except (KeyError, IndexError):
                # If logging information is malformed, continue without it
                pass
        
        answers.append(answer_record)
    
    return answers
