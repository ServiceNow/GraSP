import io
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))

from grasp.processors.data_transform import (
    CreateAudioUrlTransform,
    CreateImageUrlTransform,
    SkipRecords,
)

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def image_transform():
    return CreateImageUrlTransform()


@pytest.fixture
def audio_transform():
    return CreateAudioUrlTransform()


# =============================================================================
#                            SKIP RECORDS TRANSFORM TESTS
# =============================================================================

skip_dataset = []
for i in range(100):
    skip_dataset.append({"id": i, "name": "random_" + str(i)})


def test_skip_records_range1():
    obj = SkipRecords()
    params = {"skip_type": "range", "range": "[:10],[-10:]"}
    new_dataset = obj.transform(skip_dataset, params)
    # old dataset should be 100 records and new dataset should have 80 datasets
    # and first record is 10, skipping 0-9 and 91-99
    assert len(skip_dataset) == 100 and len(new_dataset) == 80 and new_dataset[0]["id"] == 10


def test_skip_records_range2():
    obj = SkipRecords()
    params = {"skip_type": "range", "range": "[10:20],[-20:-10]"}
    new_dataset = obj.transform(skip_dataset, params)
    # old dataset should be 100 records and new dataset should have 80 datasets
    # and first record is 0, skipping [10-20) and [80-90), last record is 99
    assert (
        len(skip_dataset) == 100
        and len(new_dataset) == 80
        and new_dataset[0]["id"] == 0
        and new_dataset[79]["id"] == 99
    )


def test_skip_records_count():
    obj = SkipRecords()
    params = {"skip_type": "count", "count": {"from_start": 10, "from_end": 10}}
    new_dataset = obj.transform(skip_dataset, params)
    # old dataset should be 100 records and new dataset should have 80 datasets
    # and first record is 10, skipping 0-9 and 91-99
    assert len(skip_dataset) == 100 and len(new_dataset) == 80 and new_dataset[0]["id"] == 10


# =============================================================================
#                            IMAGE TRANSFORM TESTS
# =============================================================================

# --- Real File Test ---


@patch("grasp.utils.image_utils.get_image_fields", return_value=["img"])
@patch("grasp.utils.image_utils.is_data_url", return_value=False)
def test_image_transform_with_generated_chart(mock_is_data_url, mock_get_fields, image_transform):
    # Generate random data and create a chart
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, size=100)

    plt.figure()
    plt.plot(x, y)
    plt.title("Random Sine Wave Chart")

    # Save plot to BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format="jpg")
    plt.close()
    buf.seek(0)

    # Convert image buffer to a temporary file (simulate file input)
    temp_image_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_image_dir, "temp_chart.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(buf.read())

    # Run the transform
    data = [{"img": temp_image_path}]
    result = image_transform.transform(data, {})

    # Assertions
    assert result[0]["img"].startswith("data:image/")

    # Clean up temp image file
    os.remove(temp_image_path)


# --- Dict with Bytes ---


@patch("grasp.utils.image_utils.get_image_fields", return_value=["img"])
@patch("grasp.utils.image_utils.is_data_url", return_value=False)
def test_image_transform_with_dict_bytes(mock_is_data_url, mock_get_fields, image_transform):
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = [{"img": {"bytes": buf.getvalue()}}]
    result = image_transform.transform(data, {})
    assert result[0]["img"].startswith("data:image/")


# --- Raw Bytes ---


@patch("grasp.utils.image_utils.get_image_fields", return_value=["img"])
@patch("grasp.utils.image_utils.is_data_url", return_value=False)
def test_image_transform_with_raw_bytes(mock_is_data_url, mock_get_fields, image_transform):
    img = Image.new("RGB", (5, 5), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = [{"img": buf.getvalue()}]
    result = image_transform.transform(data, {})
    assert result[0]["img"].startswith("data:image/")


# --- List Input (Valid and Invalid Items) ---


@patch("grasp.utils.image_utils.get_image_fields", return_value=["img"])
@patch("grasp.utils.image_utils.is_data_url", return_value=False)
@patch("grasp.utils.image_utils.load_image", side_effect=[None, None])
def test_image_transform_with_invalid_list_items(
    mock_load_image, mock_is_data_url, mock_get_fields, image_transform
):
    # create test_image1 and test_image2 as dummy images
    img1 = Image.new("RGB", (10, 10), color="green")
    img2 = Image.new("RGB", (10, 10), color="yellow")
    buf1 = io.BytesIO()
    img1.save(buf1, format="JPEG")
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")

    temp_image_dir = tempfile.gettempdir()
    temp_image_path1 = os.path.join(temp_image_dir, "temp_image1.jpg")
    temp_image_path2 = os.path.join(temp_image_dir, "temp_image2.png")
    temp_image_path3 = "random_path/not_an_image.jpg"  # This will not be processed
    with open(temp_image_path1, "wb") as f:
        f.write(buf1.getvalue())
    with open(temp_image_path2, "wb") as f:
        f.write(buf2.getvalue())
    # Run the transform with a list of images
    data = [{"img": [temp_image_path1, temp_image_path2, temp_image_path3]}]

    result = image_transform.transform(data, {})
    assert result[0]["img"][0].startswith("data:image/jpeg;base64,")
    assert result[0]["img"][1].startswith("data:image/png;base64,")
    assert result[0]["img"][2] is None  # The invalid path should return None

    os.remove(temp_image_path1)
    os.remove(temp_image_path2)


# --- Already Base64 in List ---


@patch("grasp.utils.image_utils.get_image_fields", return_value=["img"])
@patch("grasp.utils.image_utils.is_data_url", side_effect=[True, False])
@patch("grasp.utils.image_utils.load_image")
@patch("grasp.utils.image_utils.get_image_url", return_value="data:image/png;base64,abc")
@patch("os.path.exists", return_value=True)
def test_image_transform_preserves_base64_items(
    mock_exists,
    mock_get_url,
    mock_load_image,
    mock_is_data_url,
    mock_get_fields,
    image_transform,
):
    data = [{"img": ["data:image/png;base64,existing"]}]

    result = image_transform.transform(data, {})

    assert result[0]["img"][0] == "data:image/png;base64,existing"


# --- Empty or None Input ---


@patch("grasp.utils.image_utils.get_image_fields", return_value=["img"])
def test_image_transform_with_empty_string(mock_get_fields, image_transform):
    data = [{"img": ""}, {"img": None}]
    result = image_transform.transform(data, {})
    assert result[0]["img"] == ""  # because it will not be detected as an image field
    assert result[1]["img"] is None


# =============================================================================
#                            AUDIO TRANSFORM TESTS
# =============================================================================

# --- Real File Test ---


@patch("grasp.utils.audio_utils.get_audio_fields", return_value=["audio"])
@patch("grasp.utils.audio_utils.is_data_url", return_value=False)
@patch(
    "grasp.utils.audio_utils.get_audio_url",
    return_value="data:audio/wav;base64,dummybase64",
)
@patch("grasp.utils.audio_utils.load_audio")
def test_audio_transform_with_temp_audio_file(
    mock_load_audio,
    mock_get_audio_url,
    mock_is_data_url,
    mock_get_fields,
    audio_transform,
):
    # Step 1: Generate dummy waveform
    sr = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Step 2: Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, waveform, sr, format="WAV")
        audio_path = tmp.name

    # Step 3: Mock load_audio to return the waveform
    mock_load_audio.return_value = (waveform, sr)

    # Step 4: Run transform using the real file path
    data = [{"audio": audio_path}]
    result = audio_transform.transform(data)

    # Step 5: Assert
    assert result[0]["audio"].startswith("data:audio/wav;base64,")

    # Clean up temp file
    os.remove(audio_path)


# --- HuggingFace Dict ---


@patch("grasp.utils.audio_utils.get_audio_fields", return_value=["audio"])
@patch("grasp.utils.audio_utils.is_data_url", return_value=False)
def test_audio_transform_with_hf_dict(mock_is_data_url, mock_get_fields, audio_transform):
    audio_dict = {"array": np.zeros(44100), "sampling_rate": 44100}
    data = [{"audio": audio_dict}]
    result = audio_transform.transform(data)
    assert result[0]["audio"].startswith("data:audio/wav;base64,")


# --- Field Remapping ---


@patch("grasp.utils.audio_utils.get_audio_fields", return_value=["audio"])
@patch("grasp.utils.audio_utils.is_data_url", return_value=False)
@patch(
    "grasp.utils.audio_utils.get_audio_url",
    return_value="data:audio/wav;base64,dummybase64",
)
@patch("grasp.utils.audio_utils.load_audio")
def test_audio_transform_with_output_field_map(
    mock_load_audio,
    mock_get_audio_url,
    mock_is_data_url,
    mock_get_fields,
    audio_transform,
):
    # Generate dummy waveform
    sr = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Write waveform to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, waveform, sr, format="WAV")
        audio_path = tmp.name

    # Mock `load_audio` to simulate decoding
    mock_load_audio.return_value = (waveform, sr)

    # Input data with field remapping
    data = [{"audio": audio_path}]
    result = audio_transform.transform(data, {"output_fields": {"audio": "audio_base64"}})

    # Assertion: check remapped field exists
    assert "audio_base64" in result[0]
    assert result[0]["audio_base64"].startswith("data:audio/wav;base64,")
    os.remove(audio_path)
