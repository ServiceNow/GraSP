import base64
import io
import os
import re
from pathlib import Path
from typing import Any, Tuple, Union, cast

import numpy as np
import requests  # type: ignore[import-untyped]

try:
    import soundfile as sf  # type: ignore[import-untyped]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "SyGra Audio requires the optional 'audio' dependencies. "
        "Install them with: pip install 'sygra[audio]'"
    )


from sygra.logger.logger_config import logger

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".aiff")


def is_data_url(val: Any) -> bool:
    """
    Check if a string is already a base64 data URL.

    Args:
        val (Any): The value to check.

    Returns:
        bool: True if the value is a data URL, False otherwise.
    """
    return isinstance(val, str) and val.startswith("data:")


def is_hf_audio_dict(val: Any) -> bool:
    """
    Detect HuggingFace AudioFeature-style dict.

    Args:
        val (Any): The value to check.

    Returns:
        bool: True if the value is a HuggingFace audio dict, False otherwise.
    """
    return (
        isinstance(val, dict)
        and isinstance(val.get("array"), np.ndarray)
        and isinstance(val.get("sampling_rate"), (int, float))
    )


def is_raw_audio_bytes(val: Any) -> bool:
    """
    Check if a value is raw audio bytes (bytes or bytearray).

    Args:
        val (Any): The value to check.

    Returns:
        bool: True if the value is raw audio bytes, False otherwise.
    """
    return isinstance(val, (bytes, bytearray))


def is_audio_path_or_url(val: Any) -> bool:
    """
    Check if a value is a local file path or a URL pointing to a supported audio file.

    Args:
        val (Any): The value to check.

    Returns:
        bool: True if the value is a valid audio file path or URL, False otherwise.
    """
    if not isinstance(val, str):
        return False

    val_lower = val.lower()

    is_url = val_lower.startswith(("http://", "https://")) and val_lower.endswith(
        SUPPORTED_AUDIO_EXTENSIONS
    )
    is_local = os.path.exists(val) and val_lower.endswith(SUPPORTED_AUDIO_EXTENSIONS)

    return is_url or is_local


def is_audio_like(val: Any) -> bool:
    """
    Unified check for any supported audio input.

    Args:
        val (Any): The value to check.

    Returns:
        bool: True if the value is any form of audio input, False otherwise.
    """
    return (
        is_raw_audio_bytes(val)
        or is_hf_audio_dict(val)
        or is_audio_path_or_url(val)
        or is_data_url(val)
    )


def load_audio(data: Any, timeout: float = 5.0) -> Union[bytes, None]:
    """
    Load audio from:
      - raw bytes
      - HF-style dict: {"array": np.ndarray, "sampling_rate": int}
      - URL
      - local file path

    Args:
        data (Any): The audio data to load.
        timeout (float): Timeout for network requests, if applicable.

    Returns:
        bytes or None: The loaded audio data as raw bytes, or None if loading fails.
    """
    if data is None:
        return None

    try:
        # 1. Raw bytes
        if is_raw_audio_bytes(data):
            return bytes(data)

        # 2. HuggingFace audio dict
        if is_hf_audio_dict(data):
            buf = io.BytesIO()
            sf.write(buf, data["array"], int(data["sampling_rate"]), format="WAV")
            return buf.getvalue()

        # 3. Remote URL
        if isinstance(data, str) and data.startswith(("http://", "https://")):
            try:
                response = requests.get(data, timeout=timeout)
                response.raise_for_status()
                return cast(bytes, response.content)
            except Exception:
                return None

        # 4. Local file path
        if isinstance(data, str) and os.path.exists(data):
            try:
                with open(data, "rb") as f:
                    return f.read()
            except Exception:
                return None
    except Exception as e:
        logger.warning(f"Failed to load audio data: {e}")
        return None

    # If none of the conditions matched and no exception occurred
    return None


def get_audio_url(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    """
    Convert raw audio bytes to a base64 data URL.

    Args:
        audio_bytes (bytes): The raw audio data.
        mime (str): The MIME type of the audio data (default is "audio/wav").

    Returns:
        str: A base64-encoded data URL representing the audio.
    """
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def get_audio_fields(sample_record: dict[str, Any]) -> list[str]:
    """
    Identify audio-like fields in a sample record.

    Args:
        sample_record (dict[str, Any]): The record to inspect.

    Returns:
        list[str]: A list of keys that likely contain audio data.
    """
    fields = []
    for k, v in sample_record.items():
        if isinstance(v, list):
            if any(is_audio_like(item) for item in v):
                fields.append(k)
        elif is_audio_like(v):
            fields.append(k)
    return fields


def expand_audio_item(item: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand an audio item with a variable URL into multiple items if necessary.

    Args:
        item (dict[str, Any]): The audio item to expand.
        state (dict[str, Any]): The current state containing variable values.

    Returns:
        list[dict[str, Any]]: A list of expanded audio items.
    """
    key_match = re.findall(r"{(.*?)}", item["audio_url"])
    expanded = []
    if key_match:
        assert len(key_match) == 1, "Only one variable is expected in audio_url"
        var_name = key_match[0]
        val = state.get(var_name)
        if isinstance(val, list):
            for audio_url in val:
                expanded.append({"type": "audio_url", "audio_url": {"url": audio_url}})
        else:
            expanded.append({"type": "audio_url", "audio_url": {"url": val}})
    else:
        expanded.append(item)
    return expanded


def parse_audio_data_url(data_url: str) -> Tuple[str, str, bytes]:
    """
    Parse an audio data URL and extract MIME type, extension, and decoded content.

    Args:
        data_url (str): The data URL string (e.g., "data:audio/wav;base64,...")

    Returns:
        Tuple[str, str, bytes]: Tuple of (mime_type, file_extension, decoded_bytes)

    Raises:
        ValueError: If the data URL format is invalid
    """
    # Pattern: data:<mime_type>;base64,<base64_data>
    pattern = r'^data:([^;]+);base64,(.+)$'
    match = re.match(pattern, data_url)

    if not match:
        raise ValueError(f"Invalid audio data URL format: {data_url[:50]}...")

    mime_type = match.group(1)
    base64_data = match.group(2)

    # Decode base64 data
    try:
        decoded_bytes = base64.b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {e}")

    # Determine file extension from MIME type
    mime_to_ext = {
        'audio/mpeg': 'mp3',
        'audio/mp3': 'mp3',
        'audio/opus': 'opus',
        'audio/aac': 'aac',
        'audio/flac': 'flac',
        'audio/wav': 'wav',
        'audio/wave': 'wav',
        'audio/pcm': 'pcm',
        'audio/ogg': 'ogg',
        'audio/m4a': 'm4a',
        'audio/aiff': 'aiff',
    }

    file_extension = mime_to_ext.get(mime_type, mime_type.split('/')[-1])

    return mime_type, file_extension, decoded_bytes


def save_audio_data_url(
    data_url: str,
    output_dir: Path,
    record_id: str,
    field_name: str,
    index: int = 0
) -> str:
    """
    Save an audio data URL to a file and return the file path.

    Args:
        data_url (str): The base64 data URL to save
        output_dir (Path): Directory where the file should be saved
        record_id (str): ID of the record (for unique filename)
        field_name (str): Name of the field containing the data
        index (int): Index if the field contains multiple items (default: 0)

    Returns:
        str: Relative path to the saved file

    Raises:
        ValueError: If the data URL is invalid or saving fails
    """
    try:
        # Parse the data URL
        mime_type, file_extension, decoded_bytes = parse_audio_data_url(data_url)

        # Create subdirectory for audio
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Create filename: record_id_fieldname_index.extension
        filename = f"{record_id}_{field_name}_{index}.{file_extension}"
        file_path = audio_dir / filename

        # Save the decoded bytes to file
        with open(file_path, 'wb') as f:
            f.write(decoded_bytes)

        # full path from root
        full_path = str(file_path.resolve())

        logger.debug(f"Saved audio file: {full_path} ({len(decoded_bytes)} bytes)")
        return full_path

    except Exception as e:
        logger.error(f"Failed to save audio data: {e}")
        raise
