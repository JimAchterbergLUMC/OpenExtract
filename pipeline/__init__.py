"""
Utils package for the Paper Q&A system.

This package contains modular utilities for processing research papers
and answering questions using dense retrieval and language models.
"""

from .choice_utils import extract_choice_id, normalize_choices
from .data_models import ChoiceType, Question, load_questions
from .openrouter_client import call_openrouter
from .paper_store import PaperStore, build_or_load_paper_index, compute_paper_id
from .pdf_parsing import PARSER_BACKENDS, ParsedPaper, parse_pdf
from .pdf_utils import extract_pdf_text
from .qa_orchestrator import answer_questions_for_paper
from .retrieval import DenseRetriever
from .section_chunking import Chunk, chunk_document
from .text_chunking import chunk_text
from .text_cleaning import clean_parsed_paper

__all__ = [
    # Data models
    "Question",
    "ChoiceType",
    "load_questions",
    
    # PDF processing
    "parse_pdf",
    "ParsedPaper",
    "PARSER_BACKENDS",
    "extract_pdf_text",
    
    # Text processing
    "clean_parsed_paper",
    "chunk_text",
    "chunk_document",
    "Chunk",
    
    # Cache / vector store
    "PaperStore",
    "build_or_load_paper_index",
    "compute_paper_id",
    
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
