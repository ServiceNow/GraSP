"""
Utility for processing multimodal data (audio and images) in records.
This module orchestrates the use of audio_utils and image_utils to save base64 data URLs to files.
"""
from pathlib import Path
from typing import Any, Dict

from sygra.logger.logger_config import logger
from sygra.utils import audio_utils, image_utils


def is_multimodal_data_url(value: Any) -> bool:
    """
    Check if a value is a base64 encoded data URL for audio or image.
    
    Args:
        value: The value to check
        
    Returns:
        bool: True if the value is a multimodal data URL, False otherwise
    """
    if not isinstance(value, str):
        return False
    return image_utils.is_data_url(value) or audio_utils.is_data_url(value)


def save_multimodal_data_url(
    data_url: str,
    output_dir: Path,
    record_id: str,
    field_name: str,
    index: int = 0
) -> str:
    """
    Save a multimodal data URL (audio or image) to a file and return the file path.
    
    Args:
        data_url: The base64 data URL to save
        output_dir: Directory where the file should be saved
        record_id: ID of the record (for unique filename)
        field_name: Name of the field containing the data
        index: Index if the field contains multiple items (default: 0)
        
    Returns:
        str: Relative path to the saved file
        
    Raises:
        ValueError: If the data URL is invalid or saving fails
    """
    if image_utils.is_data_url(data_url):
        return image_utils.save_image_data_url(data_url, output_dir, record_id, field_name, index)
    elif audio_utils.is_data_url(data_url):
        return audio_utils.save_audio_data_url(data_url, output_dir, record_id, field_name, index)
    else:
        raise ValueError(f"Unsupported data URL type: {data_url[:50]}...")


def process_record_multimodal_data(
    record: Dict[str, Any],
    output_dir: Path,
    record_id: str
) -> Dict[str, Any]:
    """
    Process a record and replace all base64 data URLs with file paths.
    
    This function recursively searches through the record structure (including nested
    dicts and lists) and replaces any multimodal data URLs with file paths.
    
    Args:
        record: The record to process
        output_dir: Directory where files should be saved
        record_id: ID of the record (for unique filenames)
        
    Returns:
        Dict[str, Any]: The processed record with data URLs replaced by file paths
    """
    def process_value(value: Any, field_name: str, index: int = 0) -> Any:
        """Recursively process values in the record."""
        if is_multimodal_data_url(value):
            # Save the data URL to file and return the file path
            try:
                file_path = save_multimodal_data_url(
                    value, output_dir, record_id, field_name, index
                )
                return file_path
            except Exception as e:
                logger.warning(f"Failed to process data URL in field '{field_name}': {e}")
                return value
                
        elif isinstance(value, dict):
            return {k: process_value(v, f"{field_name}_{k}", 0) for k, v in value.items()}
            
        elif isinstance(value, list):
            return [process_value(item, field_name, idx) for idx, item in enumerate(value)]
            
        else:
            # Return value as-is if it's not a data URL, dict, or list
            return value
    
    # Process the entire record
    processed_record = {}
    for key, value in record.items():
        processed_record[key] = process_value(value, key)
    
    return processed_record


def process_batch_multimodal_data(
    records: list[Dict[str, Any]],
    output_dir: Path
) -> list[Dict[str, Any]]:
    """
    Process a batch of records and save all multimodal data to files.
    
    Args:
        records: List of records to process
        output_dir: Directory where multimodal files should be saved
        
    Returns:
        list[Dict[str, Any]]: List of processed records with data URLs replaced by file paths
    """
    if not records:
        return records
    
    # Create multimodal output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_records = []
    for record in records:
        # Use record ID if available, otherwise use index
        record_id = str(record.get('id', f"record_{len(processed_records)}"))
        
        processed_record = process_record_multimodal_data(record, output_dir, record_id)
        processed_records.append(processed_record)
    
    logger.info(f"Processed {len(records)} records, saved multimodal files to {output_dir}")
    return processed_records
