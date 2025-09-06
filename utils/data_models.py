"""
Data models for the Paper Q&A system.

This module contains the core data structures used throughout the application.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


# Type aliases for better readability
ChoiceType = Union[str, Dict[str, str]]


@dataclass
class Question:
    """
    Represents a question in the Q&A system.
    
    Attributes:
        id: Unique identifier for the question
        text: The actual question text
        choices: Optional list of choices for multiple-choice questions.
                Can be either strings or dictionaries with 'id' and 'label' keys.
    """
    id: str
    text: str
    choices: Optional[List[ChoiceType]] = None


def load_questions(path: Path) -> List[Question]:
    """
    Load questions from a JSON file.
    
    Args:
        path: Path to the JSON file containing questions
        
    Returns:
        List of Question objects
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        KeyError: If required fields are missing from the JSON structure
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    questions = []
    for q in raw.get("questions", []):
        questions.append(
            Question(
                id=q["id"],
                text=q["text"],
                choices=q.get("choices") or None,
            )
        )
    return questions
