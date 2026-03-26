"""
Utility Functions Module
Provides helper functions for the document processing pipeline.
"""

import os
import json
import csv
from typing import Dict, List, Optional, Any
from pathlib import Path


def load_labels(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load ground truth labels from CSV file.
    
    Args:
        csv_path: Path to the labels CSV file
        
    Returns:
        Dictionary mapping filename to label data
    """
    labels = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            labels[filename] = {
                'name': row['name'],
                'date': row['date'],
                'amount': row['amount']
            }
    
    return labels


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save processing results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load processing results from JSON file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        List of result dictionaries
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy(predictions: List[Dict[str, Any]], 
                       ground_truth: Dict[str, Dict[str, str]]) -> Dict[str, float]:
    """
    Calculate accuracy metrics for predictions.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: Dictionary of ground truth labels
        
    Returns:
        Dictionary with accuracy metrics
    """
    total = len(predictions)
    if total == 0:
        return {'overall': 0.0, 'name': 0.0, 'date': 0.0, 'amount': 0.0}
    
    correct_name = 0
    correct_date = 0
    correct_amount = 0
    
    for pred in predictions:
        filename = pred.get('filename', '')
        if filename not in ground_truth:
            continue
        
        gt = ground_truth[filename]
        
        # Check name (case-insensitive)
        pred_name = pred.get('fields', {}).get('name', '') or ''
        if pred_name.lower().strip() == gt['name'].lower().strip():
            correct_name += 1
        
        # Check date
        pred_date = pred.get('fields', {}).get('date', '') or ''
        if pred_date == gt['date']:
            correct_date += 1
        
        # Check amount
        pred_amount = pred.get('fields', {}).get('amount', '') or ''
        gt_amount = gt['amount']
        
        # Compare as floats
        try:
            if abs(float(pred_amount) - float(gt_amount)) < 0.01:
                correct_amount += 1
        except:
            pass
    
    return {
        'overall': (correct_name + correct_date + correct_amount) / (total * 3) * 100,
        'name': correct_name / total * 100,
        'date': correct_date / total * 100,
        'amount': correct_amount / total * 100
    }


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def format_response(extracted_text: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the API response.
    
    Args:
        extracted_text: Raw extracted text
        fields: Extracted fields (can be strings or arrays)
        
    Returns:
        Formatted response dictionary
    """
    # Clean up None/empty values - handle both strings and arrays
    cleaned_fields = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                cleaned_fields[key] = value
        elif isinstance(value, list):
            # Keep non-empty lists
            if value:  # Non-empty list
                cleaned_fields[key] = value
        else:
            # Keep other types as is
            cleaned_fields[key] = value
    
    return {
        'extracted_text': extracted_text,
        'fields': cleaned_fields
    }
