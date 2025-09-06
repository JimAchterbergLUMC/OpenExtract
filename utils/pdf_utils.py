"""
PDF text extraction utilities.

This module provides functionality for extracting text content from PDF files
using the pypdf library with error handling and text cleaning.
"""

from pathlib import Path

# Optional dependency handling
try:
    from pypdf import PdfReader  # type: ignore
except ImportError:
    PdfReader = None


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using pypdf.
    
    Args:
        pdf_path: Path to the PDF file to extract text from
        
    Returns:
        Extracted text as a string, with pages separated by double newlines
        
    Raises:
        RuntimeError: If pypdf is not installed
        FileNotFoundError: If the PDF file doesn't exist
        
    Note:
        - Soft hyphens (U+00AD) are removed from the extracted text
        - Each page is extracted separately and joined with double newlines
        - If a page fails to extract, an empty string is used for that page
    """
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is not installed. Run `pip install pypdf` to use PDF extraction functionality."
        )
    
    reader = PdfReader(str(pdf_path))
    pages = []
    
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        except Exception:
            # If a specific page fails to extract, append empty string
            pages.append("")
    
    # Join all pages with double newlines
    text = "\n\n".join(pages)
    
    # Clean up soft hyphens which can interfere with text processing
    text = text.replace("\u00ad", "")
    
    return text
