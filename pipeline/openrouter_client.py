"""
OpenRouter API client for language model interactions.

This module provides functionality for communicating with the OpenRouter API
to get responses from various language models for question-answering tasks.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Resolve prompts.json relative to the repository root (parent of this package)
# so the pipeline works regardless of the current working directory.
_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts.json"
_prompts_cache: Optional[Dict[str, str]] = None


def _load_prompts() -> Dict[str, str]:
    """Load prompts.json once and cache it for the rest of the run."""
    global _prompts_cache
    if _prompts_cache is None:
        with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
            _prompts_cache = json.load(f)
    return _prompts_cache


def call_openrouter(
    model: str,
    question: str,
    paper_title: str,
    context_chunks: List[str],
    api_key: str,
    referer: Optional[str] = None,
    x_title: Optional[str] = "Paper QA",
    choice_ids: Optional[List[str]] = None,
    id_to_label: Optional[Dict[str, str]] = None,
    use_structured_output: Optional[bool] = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Call the OpenRouter API to get an answer to a question based on context.

    Args:
        model: The model identifier to use (e.g., "deepseek/deepseek-chat-v3.1:free")
        question: The question to answer
        paper_title: Title of the paper being analyzed
        context_chunks: List of relevant text chunks to use as context
        api_key: OpenRouter API key for authentication
        referer: Optional HTTP referer header
        x_title: Optional X-Title header for the request
        choice_ids: Optional list of valid choice IDs for multiple-choice questions
        id_to_label: Optional mapping from choice IDs to human-readable labels

    Returns:
        Tuple of (answer_text, full_api_response)
        - answer_text: The model's response text, or None if the request failed
        - full_api_response: Complete API response for logging (None if request failed)

    Note:
        Transient errors (429 and 5xx, network errors) are retried up to 6 times
        with backoff. Permanent client errors (e.g. 401/400/402) fail immediately.
    """
    # Format context with chunk numbers for better organization
    context = "\n\n\n".join(
        [f"CHUNK {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompts = _load_prompts()

    base_rules = prompts["base"]

    user_suffix = prompts["user"]

    if choice_ids:

        user_prompt = (
            f"INSTRUCTIONS: {user_suffix}\n\n"
            f"QUESTION: {question}\n\n"
            f"PAPER: {paper_title}\n\n"
            f"ANSWERS: {choice_ids}\n\n"
            f"CONTEXT: \n\n{context}\n\n"
        )

    else:

        user_prompt = (
            f"INSTRUCTIONS: {user_suffix}\n\n"
            f"QUESTION: {question}\n\n"
            f"PAPER: {paper_title}\n\n"
            f"CONTEXT: \n\n{context}\n\n"
        )

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    if referer:
        headers["HTTP-Referer"] = referer
    if x_title:
        headers["X-Title"] = x_title

    # Prepare messages
    messages = [
        {"role": "system", "content": base_rules},
        {"role": "user", "content": user_prompt},
    ]

    # Prepare request payload
    if use_structured_output:
        payload = {
            "model": model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answers_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answers": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["answers"],
                        "additionalProperties": False,
                    },
                },
            },
            "provider": {"require_parameters": True},
        }
    else:
        payload = {
            "model": model,
            "messages": messages,
        }

    # Retry transient errors with backoff; fail fast on permanent client errors.
    last_error = None
    for attempt in range(6):
        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=60
            )
        except requests.RequestException as exc:
            # Network errors, timeouts, etc. are worth retrying
            last_error = f"request error: {exc}"
            time.sleep(1.5 * (attempt + 1))
            continue

        if response.status_code == 200:
            data = response.json()
            try:
                answer_text = data["choices"][0]["message"]["content"].strip()
                return answer_text, data
            except (KeyError, IndexError, AttributeError):
                last_error = f"malformed 200 response: {str(data)[:500]}"
                return None, data

        last_error = f"HTTP {response.status_code}: {response.text[:500]}"

        if response.status_code == 429 or response.status_code >= 500:
            # Transient: honor Retry-After if the server provides one
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = 1.5 * (attempt + 1)
            time.sleep(delay)
            continue

        # Permanent client error (401 bad key, 400 bad request, 402 credits, ...)
        break

    print(f"OpenRouter request failed for model '{model}': {last_error}")
    return None, None
