import os
import json
import hashlib
from typing import Dict, List, Optional, Any


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save processing results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output JSON file
    """
    # Create the output directory before writing the JSON file.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _load_json_list_or_empty(results_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON list from disk, returning an empty list for missing, empty,
    or invalid files.

    Args:
        results_path: Path to results JSON file

    Returns:
        List of stored result dictionaries
    """
    # Missing file means there are no previously stored results.
    if not os.path.exists(results_path):
        return []

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            raw_content = f.read().strip()
            # Treat an empty file as an empty result list.
            if not raw_content:
                return []

            data = json.loads(raw_content)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        # Invalid JSON should not stop document processing.
        return []


def format_response(extracted_text: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the API response.
    
    Args:
        extracted_text: Raw extracted text
        fields: Extracted fields (can be strings or arrays)
        
    Returns:
        Formatted response dictionary
    """
    cleaned_fields = {}
    for key, value in fields.items():
        # Skip missing values so the API response stays compact.
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                cleaned_fields[key] = value
        elif isinstance(value, list):
            # Store lists only when they contain extracted values.
            if value:
                cleaned_fields[key] = value
        else:
            cleaned_fields[key] = value

    return {
        'extracted_text': extracted_text,
        'fields': cleaned_fields
    }


def build_result_record(
    filename: str,
    content: bytes,
    response: Dict[str, Any],
    content_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a normalized result record for persistent storage.

    Args:
        filename: Uploaded file name
        content: Raw uploaded file bytes
        response: API response payload
        content_type: Uploaded file content type

    Returns:
        Normalized result record
    """
    # Use a content hash so the same uploaded file can be recognized later.
    return {
        'filename': filename or '',
        'content_type': content_type or '',
        'file_hash': hashlib.sha256(content).hexdigest(),
        'response': response
    }


def append_unique_result(result: Dict[str, Any], output_path: str) -> bool:
    """
    Append a result to the JSON file only if an equivalent result is not
    already stored.

    Duplicate detection is based on the stored file identity and response.
    If the same file hash or filename is already present with the same
    response payload, the file is not written again.

    Args:
        result: Result dictionary to persist
        output_path: Path to output JSON file

    Returns:
        True if a new result was added, otherwise False
    """
    # Load existing saved responses before checking for duplicates.
    existing_results = _load_json_list_or_empty(output_path)

    new_filename = result.get('filename', '')
    new_file_hash = result.get('file_hash', '')
    new_response = result.get('response', {})

    for existing in existing_results:
        # A duplicate means the same response was already stored for the
        # same file content or the same uploaded filename.
        same_response = existing.get('response', {}) == new_response
        same_file_hash = bool(new_file_hash) and existing.get('file_hash', '') == new_file_hash
        same_filename = bool(new_filename) and existing.get('filename', '') == new_filename

        if same_response and (same_file_hash or same_filename):
            return False

    # Only append when no matching file/response pair exists yet.
    existing_results.append(result)
    save_results(existing_results, output_path)
    return True
