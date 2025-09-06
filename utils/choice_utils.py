"""
Choice handling utilities for multiple-choice questions.

This module provides functionality for normalizing choice formats and parsing
model outputs to extract valid choice IDs from various response formats.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from .data_models import ChoiceType


def normalize_choices(choices: List[ChoiceType]) -> Tuple[List[str], Dict[str, str]]:
    """
    Normalize choice formats and extract IDs and labels.
    
    Args:
        choices: List of choices, either as strings or dicts with 'id' and 'label' keys
        
    Returns:
        Tuple of (choice_ids, id_to_label_mapping)
        - choice_ids: List of canonical choice IDs the model must choose from
        - id_to_label_mapping: Dictionary mapping IDs to human-readable labels
        
    Examples:
        >>> normalize_choices(["option1", "option2"])
        (["option1", "option2"], {"option1": "option1", "option2": "option2"})
        
        >>> normalize_choices([{"id": "a", "label": "Option A"}, {"id": "b", "label": "Option B"}])
        (["a", "b"], {"a": "Option A", "b": "Option B"})
    """
    ids: List[str] = []
    id_to_label: Dict[str, str] = {}
    
    if not choices:
        return ids, id_to_label
    
    if isinstance(choices[0], str):
        # Choices are simple strings
        ids = [str(choice) for choice in choices]
        id_to_label = {choice: choice for choice in ids}
    else:
        # Choices are dictionaries with 'id' and 'label' keys
        for choice_obj in choices:  # type: ignore[assignment]
            choice_id = str(choice_obj.get("id"))
            choice_label = str(choice_obj.get("label", choice_id))
            ids.append(choice_id)
            id_to_label[choice_id] = choice_label
    
    return ids, id_to_label


def extract_choice_id(answer: str, valid_ids: List[str]) -> Optional[str]:
    """
    Parse model output and extract a valid choice ID if present.
    
    This function attempts to extract a valid choice ID from various response formats:
    1. JSON objects with an "id" field (fenced or direct)
    2. JSON strings
    3. Bare choice IDs
    4. Choice IDs embedded in longer text
    
    Args:
        answer: The model's response text
        valid_ids: List of valid choice IDs to match against
        
    Returns:
        Valid choice ID if found, None otherwise
        
    Examples:
        >>> extract_choice_id('{"id": "option1"}', ["option1", "option2"])
        "option1"
        
        >>> extract_choice_id("The answer is option2 based on the context.", ["option1", "option2"])
        "option2"
        
        >>> extract_choice_id("option1", ["option1", "option2"])
        "option1"
    """
    if not answer:
        return None
    
    answer = answer.strip()
    
    # Try to extract from fenced JSON blocks
    fence_match = re.search(
        r"```json\s*(\{.*?\}|\s*\".*?\"\s*)\s*```", 
        answer, 
        flags=re.DOTALL
    )
    if fence_match:
        json_block = fence_match.group(1).strip()
        try:
            parsed = json.loads(json_block)
            if isinstance(parsed, dict) and "id" in parsed and parsed["id"] in valid_ids:
                return parsed["id"]
            if isinstance(parsed, str) and parsed in valid_ids:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Try to parse as direct JSON
    if answer.startswith("{") or answer.startswith('"'):
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, dict) and "id" in parsed and parsed["id"] in valid_ids:
                return parsed["id"]
            if isinstance(parsed, str) and parsed in valid_ids:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Check for exact match as bare token
    if answer in valid_ids:
        return answer
    
    # Use regex to find choice IDs embedded in text
    if valid_ids:
        # Sort by length (descending) to match longer IDs first
        sorted_ids = sorted(valid_ids, key=len, reverse=True)
        # Escape special regex characters and create alternation pattern
        escaped_ids = [re.escape(id_str) for id_str in sorted_ids]
        pattern = r"(?<![A-Za-z0-9_])(" + "|".join(escaped_ids) + r")(?![A-Za-z0-9_])"
        
        match = re.search(pattern, answer)
        if match:
            return match.group(1)
    
    return None
