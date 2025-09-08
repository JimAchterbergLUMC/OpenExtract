"""
Parser for research paper answer datasets.

This module provides functionality to parse JSON files from both 'answers_labeled' 
and 'answers_free_text' folders and create a pandas DataFrame in long format.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Optional, Union


def parse_answers_to_dataframe(
    answers_labeled_dir: str = "answers_labeled",
    answers_free_text_dir: str = "answers_free_text",
    base_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse answer JSON files from both labeled and free text directories into a pandas DataFrame.
    
    This function reads JSON files from two directories:
    1. answers_labeled: Contains structured questions with predefined answer choices
    2. answers_free_text: Contains open-ended questions with free text answers
    
    The resulting DataFrame has a long format where each row represents one question-answer pair
    from a paper, with labeled questions prioritized first, followed by free text questions.
    
    Parameters:
    -----------
    answers_labeled_dir : str, default "answers_labeled"
        Directory name containing labeled answer JSON files
    answers_free_text_dir : str, default "answers_free_text"
        Directory name containing free text answer JSON files
    base_path : str, optional
        Base path where the answer directories are located. If None, uses current working directory.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - paper_id: Paper identifier (filename without extension)
        - question_id: Question identifier (e.g., Q1, Q2, etc.)
        - question: The question text
        - answer: The final answer
        - raw_answer: Raw answer from the model
        - answer_label: Label for the answer
        - choices_ids: List of available answer choices 
        - chunks_id: List of chunk IDs used for answering
        - chunks_str: List of text chunks used for answering
        - sent_transformer: Name of the sentence transformer model used
        - LLM: Name of the LLM model used
        - finish_reason: LLM-returned information on why the generation was finished
        - total_len: Total length of the input + response
        - source_type: 'labeled' or 'free_text' indicating the source directory
        
    """
    
    if base_path is None:
        base_path = os.getcwd()
    
    labeled_path = os.path.join(base_path, answers_labeled_dir)
    free_text_path = os.path.join(base_path, answers_free_text_dir)
    
    # Check if directories exist
    if not os.path.exists(labeled_path):
        raise FileNotFoundError(f"Directory not found: {labeled_path}")
    if not os.path.exists(free_text_path):
        raise FileNotFoundError(f"Directory not found: {free_text_path}")
    
    all_data = []
    
    # Process labeled answers first (priority questions)
    print("Processing labeled answers...")
    labeled_data = _process_directory(labeled_path, 'labeled')
    all_data.extend(labeled_data)
    
    # Process free text answers
    print("Processing free text answers...")
    free_text_data = _process_directory(free_text_path, 'free_text')
    
    # Merge free text data
    # Create a set of (paper_id, question_id) pairs from labeled data for quick lookup
    labeled_questions = {(row['paper_id'], row['question_id']) for row in labeled_data}
    
    for row in free_text_data:
            all_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    return df


def _process_directory(directory_path: str, source_type: str) -> List[Dict]:
    """
    Process all JSON files in a directory and extract question-answer data.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory containing JSON files
    source_type : str
        Type of source ('labeled' or 'free_text')
        
    Returns:
    --------
    List[Dict]
        List of dictionaries containing question-answer data
    """
    data = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            
            # Extract paper identifier (filename without extension)
            paper_id = os.path.splitext(filename)[0].replace('_answers', '')
            
            # Process each answer in the file
            answers = json_data.get('answers', [])
            for answer_data in answers:
                row = {
                    'paper_id': paper_id,
                    'question_id': answer_data.get('id'),
                    'question': answer_data.get('question'),
                    'answer': answer_data.get('answer'),
                    'raw_answer': answer_data.get('raw_answer'),
                    'answer_label': answer_data.get('answer_label'),
                    'choices_ids': answer_data.get('choices_ids'),
                    'chunks_id': answer_data.get('chunks_id'),
                    'chunks_str': answer_data.get('chunks_str'),
                    'sent_transformer': answer_data.get('sent_transformer'),
                    'LLM': answer_data.get('LLM'),
                    'finish_reason': answer_data.get('finish_reason'),
                    'total_len': answer_data.get('total_len'),
                    'source_type': source_type
                }
                data.append(row)
                
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue
    
    return data


def save_dataset(df: pd.DataFrame, output_path: str, format: str = 'csv') -> None:
    """
    Save the processed dataset to a file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    output_path : str
        Path where to save the file
    format : str, default 'csv'
        Format to save in ('csv', 'excel', 'json', 'parquet')
    """
    if format.lower() == 'csv':
        df.to_csv(output_path, index=False)
    elif format.lower() == 'excel':
        df.to_excel(output_path, index=False)
    elif format.lower() == 'json':
        df.to_json(output_path, orient='records', indent=2)
    elif format.lower() == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Dataset saved to {output_path}")

