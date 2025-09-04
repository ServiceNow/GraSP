import base64
import io
import os
import re
from typing import Any, Union

import numpy as np
import requests
try:
    import soundfile as sf
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "GraSP Audio requires the optional 'audio' dependencies. "
        "Install them with: pip install 'grasp[audio]'"
    )


from grasp.logger.logger_config import logger

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
                return response.content
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


def expand_audio_item(
    item: dict[str, Any], state: dict[str, Any]
) -> list[dict[str, Any]]:
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
