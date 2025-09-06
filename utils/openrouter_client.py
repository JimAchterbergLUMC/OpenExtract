"""
OpenRouter API client for language model interactions.

This module provides functionality for communicating with the OpenRouter API
to get responses from various language models for question-answering tasks.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        - answer_text: The model's response text
        - full_api_response: Complete API response for logging (None if request failed)
        
    Note:
        The function implements retry logic with exponential backoff (3 attempts).
        If all attempts fail, returns "Unknown from this paper".
    """
    # Format context with chunk numbers for better organization
    context = "\n\n\n".join([
        f"CHUNK {i+1}: {chunk}" 
        for i, chunk in enumerate(context_chunks)
    ])
    
    base_rules = (
        "You are a helpful and meticulous research assistant, that answers questions about study details carefully and adequately from context chunks of provided research papers."
    )

    if choice_ids:
        user_suffix = " Carefully read the QUESTION, ANSWERS, AND CONTEXT. Answer only and exactly with the correct option from the list of answers. Never provide explanations. If the answer is not present in the context, output exactly: Unknown from this paper."
        
        user_prompt = (
        f"INSTRUCTIONS: {user_suffix}\n\n"
        f"QUESTION: {question}\n\n"
        f"PAPER: {paper_title}\n\n"
        f"ANSWERS: {choice_ids}\n\n"
        f"CONTEXT: \n\n{context}\n\n"
    )
    
    else:
        user_suffix = " Carefully read the QUESTION, ANSWERS, AND CONTEXT. Answer in no more than 12 words and be concise. Never include explanations. If the answer is not present in the context, output exactly: Unknown from this paper."
        
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
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    
    # Retry logic with exponential backoff
    for attempt in range(3):
        try:
            response = requests.post(
                OPENROUTER_URL, 
                headers=headers, 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                try:
                    answer_text = data["choices"][0]["message"]["content"].strip()
                    return answer_text, data
                except (KeyError, IndexError):
                    return "Unknown from this paper", data
            
            # If status code is not 200, wait before retrying
            time.sleep(1.5 * (attempt + 1))
            
        except requests.RequestException:
            # Handle network errors, timeouts, etc.
            time.sleep(1.5 * (attempt + 1))
    
    # All attempts failed
    return "Unknown from this paper", None
