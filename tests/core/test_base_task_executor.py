import os
import sys

# Add project root to sys.path for relative imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import hashlib
import json
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from grasp.core.base_task_executor import BaseTaskExecutor
from grasp.core.dataset.dataset_config import OutputType

# ---------------------- Fixtures ----------------------

# Mocks CLI arguments


@pytest.fixture
def mock_args():
    return Mock(
        task="test_task",
        output_dir="/tmp/test_output",
        oasst=True,  # Ensure these are True so that post_generation_tasks get picked up
        quality=True,
        resume=True,
        num_records=5,
        start_index=0,
        batch_size=2,
        checkpoint_interval=10,
        debug=False,
        run_name="",
        output_with_ts=False,
    )


# Mocks main task config


@pytest.fixture
def mock_grasp_config():
    return {
        "graph_config": {
            "nodes": {
                "build_text_node": {
                    "node_type": "lambda",
                    "output_keys": "evol_instruct_final_prompt",
                    "lambda": "grasp.recipes.evol_instruct.task_executor.EvolInstructPromptGenerator",
                },
                "evol_text_node": {
                    "node_type": "llm",
                    "output_keys": "evolved_text",
                    "prompt": [{"user": "{evol_instruct_final_prompt}"}],
                    "model": {"name": "gpt-4o", "parameters": {"temperature": 1.0}},
                },
            },
            "edges": [
                {"from": "START", "to": "build_text_node"},
                {"from": "build_text_node", "to": "evol_text_node"},
                {"from": "evol_text_node", "to": "END"},
            ],
        },
        "output_config": {"oasst_mapper": None, "data_quality": None},
        "data_config": {
            "resumable": True,
            "id_column": "some_id",
            "source": {
                "type": "disk",
                "path": "some_file.json",
                "file_path": "/tmp/fake_input.json",
            },
            "sink": {
                "type": OutputType.JSON.value,
                "file_path": "/tmp/test_output/output.json",
            },
        },
        "post_generation_tasks": {
            "oasst_mapper": {"dummy_oasst": True},
            "data_quality": {"tasks": []},
        },
    }


# Mocks model.yaml config


@pytest.fixture
def mock_model_config():
    return {
        "gpt-4o": {
            "api_key": "dummy-key",
            "api_version": "2024-02-15-preview",
            "model": "gpt-4o",
            "model_type": "azure_openai",
            "parameters": {"max_tokens": 500, "temperature": 1.0},
            "url": "https://test-url.com/",
        },
        "gpt4": {
            "api_key": "dummy-key",
            "api_version": "2024-05-01-preview",
            "model": "gpt-4-32k",
            "model_type": "azure_openai",
            "parameters": {"max_tokens": 500, "temperature": 1.0},
            "url": "https://test-url.com/",
        },
    }


# Sets up a BaseTaskExecutor instance with mocked config loading and graph
@pytest.fixture
def dummy_instance(mock_args, mock_grasp_config, mock_model_config):
    from grasp.core.base_task_executor import BaseTaskExecutor

    def mock_load_yaml_file(*args, **kwargs):
        path = args[0] if args else kwargs.get("filepath", "")
        if "models.yaml" in path:
            return mock_model_config
        return mock_grasp_config

    with (
        patch(
            "grasp.core.base_task_executor.utils.load_yaml_file",
            side_effect=mock_load_yaml_file,
        ),
        patch(
            "grasp.core.base_task_executor.utils.get_file_in_task_dir",
            return_value="dummy_path.yaml",
        ),
        patch(
            "grasp.core.dataset.file_handler.FileHandler.read",
            return_value=[{"id": "abc", "evol_instruct_final_prompt": "hello"}],
        ),
    ):
        instance = BaseTaskExecutor(mock_args)

        # Mock init_graph and compiled graph return
        mock_compiled_graph = MagicMock()
        mock_compiled_graph.get_graph.return_value.draw_ascii.return_value = "ascii_graph"
        instance.init_graph = MagicMock()
        instance.init_graph.return_value.compile.return_value = mock_compiled_graph

    return instance


# ------------------ End Fixtures ------------------


@patch("grasp.core.base_task_executor.utils.get_file_in_task_dir")
@patch("grasp.core.base_task_executor.utils.load_yaml_file")
@patch("grasp.core.base_task_executor.GraphConfig")
@patch("grasp.core.base_task_executor.BaseTaskExecutor.init_dataset")
@patch("grasp.core.base_task_executor.BaseTaskExecutor._init_output_generator")
def test_executor_initialization(
    mock_output_gen,
    mock_init_dataset,
    mock_graph_config,
    mock_load_yaml,
    mock_get_file,
    mock_args,
    mock_grasp_config,
):
    """
    Unit test to verify BaseTaskExecutor initialization flow with mocked config loading,
    dataset setup, and graph compilation. Ensures expected internal fields are set correctly.
    """
    mock_get_file.return_value = "/tests/test_task/graph_config.yaml"
    mock_load_yaml.return_value = mock_grasp_config
    mock_init_dataset.return_value = [{} for _ in range(5)]

    mock_graph_config_instance = MagicMock()
    mock_graph_config_instance.config = mock_grasp_config
    mock_graph_config.return_value = mock_graph_config_instance
    mock_output_gen.return_value = None

    executor = BaseTaskExecutor(mock_args)

    assert executor.task_name == "test_task"
    assert executor.resumable is True
    assert executor.dataset == [{} for _ in range(5)]


class DummyExecutor(BaseTaskExecutor):
    def __init__(self):
        # Minimal subclass for isolated utility method tests.
        self.id_column = None


def test_fetch_variable_value_basic():
    """Tests value extraction via variable resolution from nested structures."""
    executor = DummyExecutor()
    config = {"a": {"b": [1, 2, {"c": "value"}]}}
    result = executor._fetch_variable_value("$a.b[2].c", config)
    assert result == "value"


def test_fetch_variable_value_missing_key():
    """Ensures KeyError is raised for missing top-level key access."""
    executor = DummyExecutor()
    config = {"a": {"b": [1, 2, {"c": "value"}]}}
    with pytest.raises(KeyError):
        executor._fetch_variable_value("$a.x[2].c", config)


def test_fetch_variable_value_invalid_index():
    """Validates index parsing robustness when given an invalid string index."""
    executor = DummyExecutor()
    config = {"a": {"b": [1, 2, {"c": "value"}]}}
    with pytest.raises(ValueError):
        executor._fetch_variable_value("$a.b[abc].c", config)


def test_fetch_variable_value_dict():
    """Resolves dynamic values inside dictionary structures using config paths."""
    executor = DummyExecutor()
    config = {"a": {"b": {"c": "deep"}}}
    input_dict = {"key1": "$a.b.c", "key2": "static"}
    result = executor._fetch_variable_value(input_dict, config)
    assert result == {"key1": "deep", "key2": "static"}


def test_fetch_variable_value_list():
    """Tests resolution of a list containing both static and dynamic references."""
    executor = DummyExecutor()
    config = {"a": {"b": [1, 2, {"c": "x"}]}}
    input_list = ["$a.b[2].c", "plain"]
    result = executor._fetch_variable_value(input_list, config)
    assert result == ["x", "plain"]


def test_fetch_variable_value_literal():
    """Ensures literals (str/int) are returned unchanged from the resolution logic."""
    executor = DummyExecutor()
    config = {}
    assert executor._fetch_variable_value("plain_string", config) == "plain_string"
    assert executor._fetch_variable_value(42, config) == 42


def test_add_id_hashing_deterministic_and_presence():
    """Tests if an ID is deterministically added using hash if missing."""
    executor = DummyExecutor()
    input_record = {"x": 123, "y": "abc"}
    assert "id" not in input_record

    result = executor.add_id(input_record.copy())
    assert "id" in result and isinstance(result["id"], str)

    expected_hash = hashlib.sha256(
        json.dumps({"x": 123, "y": "abc"}, sort_keys=True).encode()
    ).hexdigest()
    assert result["id"] == expected_hash


def test_add_id_from_column_present():
    """Confirms existing ID field is retained without being modified."""
    executor = DummyExecutor()
    record = {"id": "abc123", "text": "hello"}
    result = executor.add_id(record.copy())
    assert result["id"] == "abc123"


def test_add_id_from_column_missing():
    """Verifies fallback to hash-based ID assignment when field is absent."""
    executor = DummyExecutor()
    record = {"text": "hello"}
    result = executor.add_id(record.copy())

    expected_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    assert result["id"] == expected_hash


def test_assign_ids_flat_list():
    """Checks ID assignment behavior for flat list of dictionaries."""
    executor = DummyExecutor()
    data = [{"a": 1}, {"b": 2}]
    result = executor.assign_ids(data)
    assert all("id" in r for r in result)


def test_process_static_variables():
    """Tests static variable resolution and mapping inside output configuration."""

    class DummyExecutor(BaseTaskExecutor):
        def __init__(self):
            pass

        def _fetch_variable_value(self, val, config):
            return "resolved_value"

    executor = DummyExecutor()
    output_config = {"output_map": {"field": {"value": "$data_config.source.path"}}}
    updated = executor._process_static_variables(output_config, {})
    assert updated["output_map"]["field"]["value"] == "resolved_value"


def test_input_record_generator():
    """Verifies that the default record generator returns the input unmodified."""

    class DummyExecutor(BaseTaskExecutor):
        def __init__(self):
            pass

    executor = DummyExecutor()
    record = {"text": "sample"}
    assert executor.input_record_generator(record) == record


def test_output_record_generator_with_no_output_gen():
    """Ensures output generator fallback returns original state if not configured."""

    class DummyExecutor(BaseTaskExecutor):
        def __init__(self):
            self.output_generator = None

    executor = DummyExecutor()
    state = Mock()
    assert executor.output_record_generator(state) == state


def test_output_record_generator_with_output_gen():
    """Tests if configured output generator is invoked with proper input state."""

    class DummyExecutor(BaseTaskExecutor):
        def __init__(self):
            self.output_generator = Mock()
            self.output_generator.generate.return_value = "processed"

    executor = DummyExecutor()
    state = Mock()
    assert executor.output_record_generator(state) == "processed"


@patch("grasp.core.dataset.file_handler.FileHandler.read", return_value=[{"id": 1}])
@patch("grasp.core.base_task_executor.logger.warning")
@patch("grasp.core.base_task_executor.logger.info")
@patch("grasp.core.base_task_executor.utils.delete_file")
@patch(
    "grasp.core.base_task_executor.utils.get_file_in_task_dir",
    return_value="/tmp/fake_input.json",
)
@patch("grasp.core.base_task_executor.os.path.join", return_value="/tmp/test/metadata.json")
@patch("grasp.core.base_task_executor.os.makedirs")
@patch("grasp.core.base_task_executor.os.path.exists", return_value=False)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"task_name": "dummy_task", "output_file": "out.json"}',
)
def test_execute_basic_flow(
    mock_open_file,
    mock_exists,
    mock_makedirs,
    mock_path_join,
    mock_get_file,
    mock_delete,
    mock_info,
    mock_warning,
    mock_read,
    dummy_instance,
):
    """
    Validates the complete execution path of BaseTaskExecutor, ensuring that:
    - Graph initialization is invoked.
    - Dataset processing completes successfully.
    - Metadata files are created and managed correctly.
    """
    dummy_processor = MagicMock()
    dummy_instance.graph_config.config = {"output_config": {}}

    with patch("grasp.core.base_task_executor.DatasetProcessor", return_value=dummy_processor):
        dummy_instance.execute()

    dummy_instance.init_graph.assert_called_once()
    dummy_processor.process_and_store_results.assert_called_once()


@patch("grasp.core.base_task_executor.logger.info")
@patch("grasp.core.base_task_executor.os.path.splitext", return_value=("existing", ".json"))
@patch(
    "grasp.core.base_task_executor.json.load",
    return_value={"task_name": "test_task", "output_file": "existing.json"},
)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"task_name": "test_task", "output_file": "existing.json"}',
)
@patch("grasp.core.base_task_executor.os.path.exists", side_effect=[True, True, True])
def test_resume_existing_file(
    mock_exists,
    mock_open_file,
    mock_json_load,
    mock_splitext,
    mock_info,
    dummy_instance,
):
    """
    Ensures the executor resumes correctly using a pre-existing output file.
    Confirms log message and processing path are aligned with resumable logic.
    """
    dummy_instance.resumable = True

    with (
        patch(
            "grasp.core.base_task_executor.DatasetProcessor", return_value=MagicMock()
        ) as mock_processor,
        patch("grasp.core.base_task_executor.DataQuality", return_value=MagicMock()),
    ):
        dummy_instance.execute()
        mock_processor.return_value.process_and_store_results.assert_called_once()
        mock_info.assert_any_call("Resuming with existing output file: existing.json")


@patch("grasp.core.base_task_executor.logger")
@patch("grasp.core.base_task_executor.utils.delete_file")
@patch(
    "grasp.core.base_task_executor.utils.get_file_in_task_dir",
    side_effect=["/tmp/output.json", "/tmp/metadata.json"],
)
@patch("builtins.open", new_callable=mock_open)
@patch(
    "grasp.core.base_task_executor.json.load",
    return_value={"task_name": "dummy_task", "output_file": "existing.json"},
)
@patch(
    "grasp.core.base_task_executor.os.path.exists",
    side_effect=[True, True, False, True, True],
)
def test_delete_non_resumable_files(
    mock_exists,
    mock_json_load,
    mock_open_file,
    mock_get_file,
    mock_delete_file,
    mock_logger,
    dummy_instance,
):
    """
    Validates that all associated output files are deleted when resumable is False.
    Also confirms downstream dataset processing remains unaffected by cleanup.
    """
    dummy_instance.resumable = False
    dummy_instance.output_config = None

    with (
        patch(
            "grasp.core.base_task_executor.DatasetProcessor", return_value=MagicMock()
        ) as mock_processor,
        patch("grasp.core.base_task_executor.DataQuality", return_value=MagicMock()),
    ):
        dummy_instance.execute()

        assert mock_delete_file.call_count >= 1
        mock_processor.return_value.process_and_store_results.assert_called_once()


@patch("grasp.core.base_task_executor.logger.error")
@patch("grasp.core.base_task_executor.utils.delete_file")
@patch("grasp.core.base_task_executor.os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"id": 1}\n')
def test_output_sink_error_handling(
    mock_open_file, mock_exists, mock_delete_file, mock_logger_error, dummy_instance
):
    """
    Simulates an exception during sink write and confirms error logging and resilience.
    Ensures proper logging of sink write failure while processing continues gracefully.
    """
    dummy_instance.output_config = MagicMock()
    dummy_instance.output_config.type.value = "file"
    dummy_instance.output_config.model_dump.return_value = {"type": "file"}
    dummy_instance.output_config.file_path = "bad/path"
    dummy_instance.output_config.data_quality = {}

    dummy_processor = MagicMock()
    dummy_processor.is_valid_schema = True

    with (
        patch(
            "grasp.core.base_task_executor.DatasetProcessor",
            return_value=dummy_processor,
        ),
        patch("grasp.core.base_task_executor.DataQuality", return_value=MagicMock()),
        patch(
            "grasp.core.base_task_executor.FileHandler.write",
            side_effect=Exception("Write error"),
        ),
        patch("json.loads", return_value={"id": 1}),
    ):
        dummy_instance.execute()

        mock_logger_error.assert_called_with("Error writing to sink: Write error")


@patch("grasp.core.base_task_executor.os.remove")
@patch(
    "grasp.core.base_task_executor.utils.get_file_in_task_dir",
    return_value="/tmp/test_output/output.json",
)
@patch("grasp.core.base_task_executor.logger.info")
@patch("builtins.open", new_callable=mock_open, read_data='{"id": 1}\n')
@patch("grasp.core.base_task_executor.os.path.exists", return_value=True)
def test_output_sink_success(
    mock_exists,
    mock_open_file,
    mock_logger_info,
    mock_get_file,
    mock_os_remove,
    dummy_instance,
):
    """
    Tests successful writing to disk file sink (e.g., JSON), including DataQuality post-processing.
    Validates output generator writes and cleanup operations execute as expected.
    """
    dummy_instance.output_config = MagicMock()
    dummy_instance.output_config.type = OutputType.JSON
    dummy_instance.output_config.model_dump.return_value = {"type": "disk_file"}
    dummy_instance.output_config.file_path = "/tmp/fake_output.json"
    dummy_instance.output_config.data_quality = {}
    dummy_instance.output_dir = "/tmp/test_output"

    dummy_processor = MagicMock()
    dummy_processor.is_valid_schema = True
    dummy_processor.resume_manager = None

    with (
        patch(
            "grasp.core.base_task_executor.DatasetProcessor",
            return_value=dummy_processor,
        ),
        patch("grasp.core.base_task_executor.DataQuality", return_value=MagicMock()),
        patch("grasp.core.base_task_executor.FileHandler.write") as mock_write,
        patch("json.loads", return_value={"id": 1}),
    ):
        dummy_instance.execute()
        mock_write.assert_called_once()


@patch("grasp.core.base_task_executor.os.remove")
@patch(
    "grasp.core.base_task_executor.utils.get_file_in_task_dir",
    return_value="/tmp/test_output/output.json",
)
@patch("grasp.core.base_task_executor.logger.info")
@patch("builtins.open", new_callable=mock_open, read_data='{"id": 1}\n')
@patch("grasp.core.base_task_executor.os.path.exists", return_value=True)
def test_output_sink_with_quality_success(
    mock_exists,
    mock_open_file,
    mock_logger_info,
    mock_get_file,
    mock_os_remove,
    dummy_instance,
):
    """
    Ensures a successful end-to-end output flow using a JSON output sink.
    Confirms that output and post-generation hooks are triggered appropriately.
    """
    dummy_instance.output_config = MagicMock()
    dummy_instance.output_config.type = OutputType.JSON
    dummy_instance.output_config.file_path = "/tmp/fake_output.json"
    dummy_instance.output_config.model_dump.return_value = {"type": "disk_file"}
    dummy_instance.output_config.data_quality = {}
    dummy_instance.output_dir = "/tmp/test_output"

    dummy_processor = MagicMock()
    dummy_processor.is_valid_schema = True
    dummy_processor.resume_manager = None

    with (
        patch(
            "grasp.core.base_task_executor.DatasetProcessor",
            return_value=dummy_processor,
        ),
        patch(
            "grasp.core.base_task_executor.DataQuality", return_value=MagicMock()
        ) as mock_quality,
        patch("grasp.core.base_task_executor.FileHandler.write") as mock_write,
        patch("json.loads", return_value={"id": 1}),
    ):
        dummy_instance.execute()

        mock_write.assert_called_once()
        mock_quality.assert_called_once()


@patch("grasp.core.base_task_executor.os.remove")
@patch(
    "grasp.core.base_task_executor.utils.get_file_in_task_dir",
    return_value="/tmp/test_output/output.json",
)
@patch("grasp.core.base_task_executor.logger.info")
@patch("builtins.open", new_callable=mock_open, read_data='{"id": 1}\n{"id": 2}\n')
@patch("grasp.core.base_task_executor.os.path.exists", return_value=True)
def test_output_sink_jsonl_reading(
    mock_exists, mock_open_file, mock_info, mock_get_file, mock_remove, dummy_instance
):
    """
    Verifies support for JSONL sink types, ensuring all lines are processed correctly.
    Includes check for downstream integration and post-processing hooks.
    """
    dummy_instance.output_config = MagicMock()
    dummy_instance.output_config.type = OutputType.JSONL
    dummy_instance.output_config.model_dump.return_value = {"type": "disk_file"}
    dummy_instance.output_config.file_path = "/tmp/test.jsonl"
    dummy_instance.output_config.data_quality = {}
    dummy_instance.output_dir = "/tmp/test_output"

    dummy_processor = MagicMock()
    dummy_processor.is_valid_schema = True
    dummy_processor.resume_manager = None

    with (
        patch(
            "grasp.core.base_task_executor.DatasetProcessor",
            return_value=dummy_processor,
        ),
        patch("grasp.core.base_task_executor.DataQuality", return_value=MagicMock()),
        patch("grasp.core.base_task_executor.FileHandler.write") as mock_write,
        patch("json.loads", side_effect=[{"id": 1}, {"id": 2}]),
    ):
        dummy_instance.execute()
        mock_write.assert_called_once()
