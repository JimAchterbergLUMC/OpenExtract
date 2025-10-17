"""
Utils package for the Paper Q&A system.

This package contains modular utilities for processing research papers
and answering questions using dense retrieval and language models.
"""

from .choice_utils import extract_choice_id, normalize_choices
from .data_models import ChoiceType, Question, load_questions
from .openrouter_client import call_openrouter
from .pdf_utils import extract_pdf_text
from .qa_orchestrator import answer_questions_for_paper
from .retrieval import DenseRetriever
from .text_chunking import chunk_text

__all__ = [
    # Data models
    "Question",
    "ChoiceType",
    "load_questions",
    
    # PDF processing
    "extract_pdf_text",
    
    # Text processing
    "chunk_text",
    
    # Retrieval
    "DenseRetriever",
    
    # Choice handling
    "normalize_choices",
    "extract_choice_id",
    
    # API client
    "call_openrouter",
    
    # Main orchestration
    "answer_questions_for_paper",
]
