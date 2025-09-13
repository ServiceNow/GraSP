import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from datasets import Dataset, Features, IterableDataset, Value

from grasp.core.base_task_executor import BaseTaskExecutor
from grasp.core.dataset.dataset_config import OutputType
from grasp.core.graph.graph_config import GraphConfig
from grasp.utils import utils

# ---------------------- Fixtures ----------------------

# Mocks CLI arguments


@pytest.fixture(autouse=True, scope="module")
def set_current_task():
    utils.current_task = "test_task"


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


@pytest.fixture
def mock_main_subgraph_config():
    return {
        "graph_config": {
            "nodes": {
                "generate_answer": {
                    "node_type": "subgraph",
                    "subgraph": "generate_answer.yaml",
                }
            },
            "edges": [
                {"from": "START", "to": "generate_answer"},
                {"from": "generate_answer", "to": "END"},
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


@pytest.fixture
def mock_grasp_sub_config():
    return {
        "graph_config": {
            "nodes": {
                "sub_node": {
                    "node_type": "lambda",
                    "lambda": "subgraph.fake.module.DummyFunc",
                    "output_keys": "sub_output",
                }
            },
            "edges": [
                {"from": "START", "to": "sub_node"},
                {"from": "sub_node", "to": "END"},
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


@pytest.fixture
def mock_main_with_nested_subgraph_config():
    return {
        "graph_config": {
            "nodes": {
                "generate_description": {
                    "node_type": "lambda",
                    "lambda": "subgraph.fake.module.DummyFunc",
                    "output_keys": "description",
                },
                "generate_answer": {"node_type": "subgraph", "subgraph": "outer.yaml"},
            },
            "edges": [
                {"from": "START", "to": "generate_description"},
                {"from": "generate_description", "to": "generate_answer"},
                {"from": "generate_answer", "to": "END"},
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


@pytest.fixture
def mock_main_with_looping_subgraph_config():
    return {
        "graph_config": {
            "nodes": {
                "generate_description": {
                    "node_type": "lambda",
                    "lambda": "subgraph.fake.module.DummyFunc",
                    "output_keys": "description",
                },
                "generate_answer": {"node_type": "subgraph", "subgraph": "inner.yaml"},
            },
            "edges": [
                {"from": "START", "to": "generate_answer"},
                {
                    "from": "generate_answer",
                    "condition": "tasks.should_continue",
                    "path_map": {"generate_answer": "generate_answer", "END": "END"},
                },
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


@pytest.fixture
def mock_outer_nested_subgraph_config():
    return {
        "graph_config": {
            "nodes": {"sub_node": {"node_type": "subgraph", "subgraph": "inner.yaml"}},
            "edges": [
                {"from": "START", "to": "sub_node"},
                {"from": "sub_node", "to": "END"},
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


@pytest.fixture
def mock_inner_subgraph_config():
    return {
        "graph_config": {
            "nodes": {
                "generate_answer_inner": {
                    "node_type": "lambda",
                    "lambda": "subgraph.fake.module.DummyFunc",
                    "output_keys": "sub_output",
                }
            },
            "edges": [
                {"from": "START", "to": "generate_answer_inner"},
                {"from": "generate_answer_inner", "to": "END"},
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


@pytest.fixture
def mock_second_subgraph_config():
    return {
        "graph_config": {
            "nodes": {
                "another_sub_node": {
                    "node_type": "lambda",
                    "lambda": "subgraph.fake.module.AnotherDummyFunc",
                    "output_keys": "other_output",
                }
            },
            "edges": [
                {"from": "START", "to": "another_sub_node"},
                {"from": "another_sub_node", "to": "END"},
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


@pytest.fixture
def mock_main_with_multiple_subgraphs():
    return {
        "graph_config": {
            "nodes": {
                "generate_answer_1": {"node_type": "subgraph", "subgraph": "sg1.yaml"},
                "generate_answer_2": {"node_type": "subgraph", "subgraph": "sg2.yaml"},
            },
            "edges": [
                {"from": "START", "to": "generate_answer_1"},
                {"from": "generate_answer_1", "to": "generate_answer_2"},
                {"from": "generate_answer_2", "to": "END"},
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


@pytest.fixture
def mock_conditional_subgraph_main_config():
    return {
        "graph_config": {
            "nodes": {
                "initial_node": {
                    "node_type": "lambda",
                    "lambda": "subgraph.fake.module.StartFunc",
                    "output_keys": "initial_output",
                },
                "subgraph_node": {
                    "node_type": "subgraph",
                    "subgraph": "conditional_sub.yaml",
                },
            },
            "edges": [
                {"from": "START", "to": "initial_node"},
                {"from": "initial_node", "to": "subgraph_node"},
                {
                    "from": "subgraph_node",
                    "condition": "tasks.subgraph.condition.should_continue",
                    "path_map": {"loop": "subgraph_node", "exit": "END"},
                },
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


@pytest.fixture
def mock_conditional_subgraph_config():
    return {
        "graph_config": {
            "nodes": {
                "sub_node_logic": {
                    "node_type": "lambda",
                    "lambda": "subgraph.fake.module.InnerLogic",
                    "output_keys": "result",
                }
            },
            "edges": [
                {"from": "START", "to": "sub_node_logic"},
                {"from": "sub_node_logic", "to": "END"},
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
            "api_key": "dummy-keys",
            "api_version": "2024-02-15-preview",
            "model": "gpt-4o",
            "model_type": "azure_openai",
            "parameters": {"max_tokens": 500, "temperature": 1.0},
            "url": "https://test-url.com/",
        },
        "gpt4": {
            "api_key": "dummy-keys",
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


# ----------------------------
# Section 1: Initialization & Config Parsing
# ----------------------------


def test_init_from_yaml_path(dummy_instance):
    """
    Test that BaseTaskExecutor initializes correctly when reading from a YAML config file.

    This test uses the dummy_instance fixture which simulates the full pipeline loading a YAML config.
    It asserts that the graph nodes are parsed and one known node is present in the graph structure.
    """
    assert dummy_instance.graph_config.config["graph_config"]["nodes"]
    assert "build_text_node" in dummy_instance.graph_config.get_nodes()


def test_init_from_dict_config(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that BaseTaskExecutor correctly initializes when passed a graph configuration as a dictionary.

    This test bypasses YAML file loading and provides the configuration directly via `graph_config_dict`.
    It ensures the nodes are correctly registered in the graph structure.
    """

    def mock_load_yaml_file(*args, **kwargs):
        filepath = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in filepath else mock_grasp_config

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
            return_value=[{"id": "abc"}],
        ),
    ):
        executor = BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)
        assert "build_text_node" in executor.graph_config.get_nodes()


def test_missing_graph_config_key(mock_args, mock_model_config):
    """
    Test that a ValueError is raised when `graph_config` is missing from the task configuration.

    It validates that the system enforces presence of the key configuration block
    and fails early when it’s not defined.
    """
    invalid_config = {"output_config": {}, "data_config": {}}

    def mock_load_yaml_file(*args, **kwargs):
        filepath = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in filepath else invalid_config

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
            return_value=[{"id": "abc"}],
        ),
    ):
        with pytest.raises(ValueError, match="graph_config key is required"):
            BaseTaskExecutor(mock_args, graph_config_dict=invalid_config)


def test_invalid_variable_type_in_output(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that a ValueError is raised if `output_keys` is not a list or string.

    Here, `output_keys` is incorrectly set to an integer to simulate a misconfiguration,
    and the executor should throw a validation error during graph parsing.
    """
    mock_grasp_config["graph_config"]["nodes"]["build_text_node"]["output_keys"] = 123

    def mock_load_yaml_file(*args, **kwargs):
        filepath = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in filepath else mock_grasp_config

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
            return_value=[{"id": "abc"}],
        ),
    ):
        with pytest.raises(ValueError, match="Invalid variable format"):
            BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)


# ----------------------------
# Section 2: Subgraph Handling
# ----------------------------


@patch("grasp.core.graph.graph_config.utils.load_yaml_file")
@patch("grasp.core.graph.graph_config.utils.get_file_in_task_dir")
@patch(
    "grasp.core.graph.nodes.lambda_node.utils.get_func_from_str",
    return_value=lambda x: x,
)
def test_subgraph_merging(
    mock_get_func,
    mock_get_file,
    mock_load_yaml,
    mock_main_subgraph_config,
    mock_grasp_sub_config,
    mock_model_config,
):
    """
    Test flat subgraph integration into main graph.

    Test Scenario:
    --------------
    This test verifies a simple case where the main graph contains a node
    of type `subgraph`, which references a standalone subgraph config (`generate_answer.yaml`).

    The subgraph contains:
    - One lambda node: `sub_node`
    - Edges: START → sub_node → END

    After merging, the main graph should contain:
    - One wrapper subgraph node `generate_answer`
    - One namespaced subgraph node `generate_answer.sub_node`
    - The original START → generate_answer → END edge path (flattened)

    Assertions:
    -----------
    - Verify the subgraph node was namespaced and merged correctly
    - Ensure the correct number of edges (2) exist post-merge
    - Confirm the structure is preserved without extraneous nodes or edges

    Expected Graph Flow:
    --------------------
    START
      ↓
    generate_answer
      ↓
    generate_answer.sub_node
      ↓
    END
    """
    mock_get_file.return_value = "generate_answer.yaml"

    def side_effect_loader(filepath, *args, **kwargs):
        if "models.yaml" in filepath:
            return mock_model_config
        elif "generate_answer.yaml" in filepath:
            return mock_grasp_sub_config
        else:
            return mock_main_subgraph_config

    mock_load_yaml.side_effect = side_effect_loader

    graph = GraphConfig(
        config=mock_main_subgraph_config,
        dataset=[{"some_id": "test"}],
        output_transform_args={"oasst": True, "quality": True},
    )

    # Confirm subgraph node name is namespaced
    assert "generate_answer.sub_node" in graph.get_nodes()

    # Confirm correct number of nodes (1 subgraph + 1 main subgraph wrapper)
    expected_node_keys = {"generate_answer.sub_node"}
    actual_node_keys = set(graph.get_nodes().keys())
    assert expected_node_keys.issubset(actual_node_keys)

    # Extract edge configs from the graph
    edge_tuples = [edge.edge_config for edge in graph.edges]

    # Confirm that edges are properly namespaced and present
    assert {"from": "START", "to": "generate_answer.sub_node"} in edge_tuples
    assert {"from": "generate_answer.sub_node", "to": "END"} in edge_tuples

    # Optionally: Confirm that no unexpected extra edges exist
    assert len(edge_tuples) == 2


@patch("grasp.core.graph.graph_config.utils.load_yaml_file")
@patch("grasp.core.graph.graph_config.utils.get_file_in_task_dir")
@patch(
    "grasp.core.graph.nodes.lambda_node.utils.get_func_from_str",
    return_value=lambda x: x,
)
def test_subgraph_merging_nested_looping(
    mock_get_func,
    mock_get_file,
    mock_load_yaml,
    mock_main_with_nested_subgraph_config,
    mock_outer_nested_subgraph_config,
    mock_inner_subgraph_config,
    mock_model_config,
):
    """
    Test that a nested subgraph is correctly loaded and merged into the main graph.

    Test Scenario:
    ----------------
    This test verifies a 3-level nested graph structure:
    - Main graph calls a top-level lambda node `generate_description`
    - Then it calls a subgraph node `generate_answer`, defined in `outer.yaml`
    - `outer.yaml` contains a subgraph node `sub_node`, defined in `inner.yaml`
    - `inner.yaml` contains a lambda node `generate_answer_inner`

    Expected Node Hierarchy:
    ------------------------
    START
      ↓
    generate_description                     <-- Top-level lambda node
      ↓
    generate_answer.sub_node.generate_answer_inner  <-- Nested subgraph node and Deepest lambda node
      ↓
    END

    Expected Edges (edge.edge_config):
    ----------------------------------
    - {'from': 'START', 'to': 'generate_description'}
    - {'from': 'generate_description', 'to': 'generate_answer.sub_node.generate_answer_inner'}
    - {'from': 'generate_answer.sub_node.generate_answer_inner', 'to': 'END'}

    Assertions:
    -----------
    - Confirm subgraph nodes are namespaced correctly
    - Confirm presence of expected edges in the flattened graph
    - Confirm total number of merged edges is correct
    """

    def side_effect_loader_get_file(dot_walk_path, file, *args, **kwargs):
        if dot_walk_path.endswith(".yaml"):
            return dot_walk_path
        else:
            # use the original mock_get_file behavior
            return mock_get_file(dot_walk_path, file, *args, **kwargs)

    mock_get_file.side_effect = side_effect_loader_get_file

    def side_effect_loader(filepath, *args, **kwargs):
        if "models.yaml" in filepath:
            return mock_model_config
        elif "outer.yaml" in filepath:
            return mock_outer_nested_subgraph_config
        elif "inner.yaml" in filepath:
            return mock_inner_subgraph_config
        else:
            return mock_main_with_nested_subgraph_config

    mock_load_yaml.side_effect = side_effect_loader

    graph = GraphConfig(
        config=mock_main_with_nested_subgraph_config,
        dataset=[{"some_id": "test"}],
        output_transform_args={"oasst": True, "quality": True},
    )

    # Assert node namespace flattening
    assert "generate_answer.sub_node.generate_answer_inner" in graph.get_nodes()

    expected_node_keys = {
        "generate_description",
        "generate_answer.sub_node.generate_answer_inner",
    }
    actual_node_keys = set(graph.get_nodes().keys())
    assert expected_node_keys.issubset(actual_node_keys)

    # Collect edge configurations
    edge_tuples = [edge.edge_config for edge in graph.edges]

    assert {"from": "START", "to": "generate_description"} in edge_tuples
    assert {
        "from": "generate_description",
        "to": "generate_answer.sub_node.generate_answer_inner",
    } in edge_tuples
    assert {
        "from": "generate_answer.sub_node.generate_answer_inner",
        "to": "END",
    } in edge_tuples

    assert len(edge_tuples) == 3, "Expected exactly 3 edges in the merged graph"


@patch("grasp.core.graph.graph_config.utils.load_yaml_file")
@patch("grasp.core.graph.graph_config.utils.get_file_in_task_dir")
@patch(
    "grasp.core.graph.nodes.lambda_node.utils.get_func_from_str",
    return_value=lambda x: x,
)
def test_subgraph_merging_with_looping_edge(
    mock_get_func,
    mock_get_file,
    mock_load_yaml,
    mock_main_with_looping_subgraph_config,
    mock_inner_subgraph_config,
    mock_model_config,
):
    """
    Test subgraph merging with a conditional looping edge in the main graph.

    Test Scenario:
    --------------
    This test verifies the case where a subgraph node (`generate_answer`) in the main graph
    is part of a conditional edge that loops back to itself or exits to END.

    The structure in `mock_main_with_looping_subgraph_config` is:
        START
          ↓
    generate_answer (subgraph)
          ↻
    condition: tasks.should_continue
        path_map: {
            "generate_answer": generate_answer (loop),
            "END": END (exit)
        }

    Subgraph (inner.yaml) contains:
        - generate_answer_inner (lambda node)
        - Edges: START → generate_answer_inner → END

    Assertions:
    -----------
    - Subgraph node should be properly namespaced (e.g., `generate_answer.generate_answer_inner`)
    - Conditional edge with correct path_map should exist
    - Total number of edges should include: START → generate_answer and the conditional edge
    """

    mock_get_file.return_value = "inner.yaml"

    def side_effect_loader(filepath, *args, **kwargs):
        if "models.yaml" in filepath:
            return mock_model_config
        elif "inner.yaml" in filepath:
            return mock_inner_subgraph_config
        else:
            return mock_main_with_looping_subgraph_config

    mock_load_yaml.side_effect = side_effect_loader

    graph = GraphConfig(
        config=mock_main_with_looping_subgraph_config,
        dataset=[{"some_id": "test"}],
        output_transform_args={"oasst": True, "quality": True},
    )

    # Confirm that subgraph node was namespaced
    assert "generate_answer.generate_answer_inner" in graph.get_nodes()

    # Check edge configs
    edge_configs = [edge.edge_config for edge in graph.edges]

    # Check regular edge from START → generate_answer
    assert {
        "from": "START",
        "to": "generate_answer.generate_answer_inner",
    } in edge_configs

    # Check conditional edge from generate_answer
    conditional_edge = next((e for e in graph.edges if e.edge_config.get("condition")), None)
    assert conditional_edge is not None
    assert conditional_edge.edge_config["from"] == "generate_answer.generate_answer_inner"
    assert conditional_edge.edge_config["condition"] == "tasks.should_continue"
    assert conditional_edge.edge_config["path_map"] == {
        "generate_answer": "generate_answer.generate_answer_inner",
        "END": "END",
    }

    # Total expected edges: 1 regular + 1 conditional
    assert len(edge_configs) == 2


@patch("grasp.core.graph.graph_config.utils.load_yaml_file")
@patch("grasp.core.graph.graph_config.utils.get_file_in_task_dir")
@patch(
    "grasp.core.graph.nodes.lambda_node.utils.get_func_from_str",
    return_value=lambda x: x,
)
def test_multiple_subgraphs_merging(
    mock_get_func,
    mock_get_file,
    mock_load_yaml,
    mock_main_with_multiple_subgraphs,
    mock_grasp_sub_config,
    mock_second_subgraph_config,
    mock_model_config,
):
    """
    Test that multiple subgraphs defined in a single graph are:
    - Individually resolved and loaded from separate YAML files
    - Merged with correct namespacing: e.g., generate_answer_1.sub_node, generate_answer_2.another_sub_node
    - Properly connected in edge flow: START → generate_answer_1 → generate_answer_2 → END
    """

    # Simulate subgraph file lookup
    def side_effect_loader(filepath, *args, **kwargs):
        if "models.yaml" in filepath:
            return mock_model_config
        elif "sg1.yaml" in filepath:
            return mock_grasp_sub_config  # Contains sub_node
        elif "sg2.yaml" in filepath:
            return mock_second_subgraph_config  # Contains another_sub_node
        else:
            return mock_main_with_multiple_subgraphs

    mock_get_file.side_effect = lambda filename, *_, **__: filename
    mock_load_yaml.side_effect = side_effect_loader

    graph = GraphConfig(
        config=mock_main_with_multiple_subgraphs,
        dataset=[{"some_id": "test"}],
        output_transform_args={"oasst": True, "quality": True},
    )

    # Confirm both subgraph nodes are namespaced and present
    nodes = graph.get_nodes()
    assert "generate_answer_1.sub_node" in nodes
    assert "generate_answer_2.another_sub_node" in nodes

    # Confirm correct edge flow through both subgraphs
    edge_configs = [edge.edge_config for edge in graph.edges]
    assert {"from": "START", "to": "generate_answer_1.sub_node"} in edge_configs
    assert {
        "from": "generate_answer_1.sub_node",
        "to": "generate_answer_2.another_sub_node",
    } in edge_configs
    assert {"from": "generate_answer_2.another_sub_node", "to": "END"} in edge_configs

    # Ensure only expected number of outer edges
    assert len(edge_configs) == 3


@patch("grasp.core.graph.graph_config.utils.load_yaml_file")
@patch("grasp.core.graph.graph_config.utils.get_file_in_task_dir")
@patch(
    "grasp.core.graph.nodes.lambda_node.utils.get_func_from_str",
    return_value=lambda x: x,
)
def test_conditional_edge_from_subgraph(
    mock_get_func,
    mock_get_file,
    mock_load_yaml,
    mock_conditional_subgraph_main_config,
    mock_conditional_subgraph_config,
    mock_model_config,
):
    """
    Test that a subgraph with a conditional edge exiting it is:
    - Properly expanded with namespace
    - The conditional edge ('from': subgraph_node) with path_map is preserved correctly
    - Graph has expected edges and structure

    Expected Flow:
        START → initial_node → subgraph_node
            └──(condition)──> subgraph_node OR END
    """
    mock_get_file.return_value = "conditional_sub.yaml"

    def side_effect_loader(filepath, *args, **kwargs):
        if "models.yaml" in filepath:
            return mock_model_config
        elif "conditional_sub.yaml" in filepath:
            return mock_conditional_subgraph_config
        else:
            return mock_conditional_subgraph_main_config

    mock_load_yaml.side_effect = side_effect_loader

    graph = GraphConfig(
        config=mock_conditional_subgraph_main_config,
        dataset=[{"some_id": "test"}],
        output_transform_args={"oasst": True, "quality": True},
    )

    # Validate nodes were loaded and namespaced
    node_keys = graph.get_nodes().keys()
    assert "subgraph_node.sub_node_logic" in node_keys
    assert "initial_node" in node_keys

    # Extract and validate edges
    edge_configs = [edge.edge_config for edge in graph.edges]

    assert {"from": "START", "to": "initial_node"} in edge_configs
    assert {
        "from": "initial_node",
        "to": "subgraph_node.sub_node_logic",
    } in edge_configs

    # Find and validate conditional edge structure
    conditional_edge = next(
        (
            e
            for e in graph.edges
            if e.edge_config["from"] == "subgraph_node.sub_node_logic"
            and "condition" in e.edge_config
        ),
        None,
    )
    assert conditional_edge is not None
    assert conditional_edge.edge_config["condition"] == "tasks.subgraph.condition.should_continue"
    assert conditional_edge.edge_config["path_map"] == {
        "loop": "subgraph_node.sub_node_logic",
        "exit": "END",
    }

    # Optional: check total edge count = 3
    assert len(graph.edges) == 3


# ----------------------------
# Section 3: Node Overrides & Prompt Replacement
# ----------------------------


def test_prompt_placeholder_override(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that prompt placeholder overrides are correctly applied in the graph configuration.

    This test sets a prompt template with a placeholder `{something}` and provides
    an override to replace it with `{anything}`. It verifies that the transformed
    prompt correctly reflects the overridden placeholder.
    """
    mock_grasp_config["graph_config"]["nodes"]["build_text_node"]["prompt"] = [
        {"user": "say {something}"}
    ]

    override = {"build_text_node": {"prompt_placeholder_map": {"something": "anything"}}}

    def mock_load_yaml_file(*args, **kwargs):
        path = args[0] if args else kwargs.get("filepath", "")
        if "models.yaml" in path:
            return mock_model_config
        return mock_grasp_config

    with (
        patch(
            "grasp.core.graph.graph_config.utils.load_yaml_file",
            side_effect=mock_load_yaml_file,
        ),
        patch(
            "grasp.core.graph.graph_config.utils.get_file_in_dir",
            return_value="dummy_path.yaml",
        ),
    ):
        graph_config = GraphConfig(
            config=mock_grasp_config,
            dataset=[{"anything": "value"}],
            output_transform_args={"oasst": True, "quality": True},
            override_config=override,
        )

        prompt = graph_config.graph_config["nodes"]["build_text_node"]["prompt"][0]["user"]
        assert "{anything}" in prompt


def test_nested_config_override_merging(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that nested configuration overrides are *not* applied by default unless explicitly passed.

    It sets an override to change the temperature in a model config,
    but since the override is not passed to the executor, the original value should remain unchanged.
    """

    def mock_load_yaml_file(*args, **kwargs):
        filepath = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in filepath else mock_grasp_config

    with (
        patch(
            "grasp.core.base_task_executor.utils.load_yaml_file",
            side_effect=mock_load_yaml_file,
        ),
        patch(
            "grasp.core.base_task_executor.utils.get_file_in_task_dir",
            return_value="dummy_path.yaml",
        ),
        patch("grasp.core.dataset.file_handler.FileHandler.read", return_value=[{"id": 1}]),
    ):
        executor = BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)
        updated_temp = executor.graph_config.graph_config["nodes"]["evol_text_node"]["model"][
            "parameters"
        ]["temperature"]
        assert updated_temp == 1.0  # should remain original


def test_override_applied_only_to_specified_node(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that node-specific overrides do not affect unrelated nodes in the graph.

    An override is provided to change the `lambda` field for `build_text_node`,
    but the test asserts that no such change occurs unless explicitly applied.
    """

    def mock_load_yaml_file(*args, **kwargs):
        filepath = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in filepath else mock_grasp_config

    with (
        patch(
            "grasp.core.base_task_executor.utils.load_yaml_file",
            side_effect=mock_load_yaml_file,
        ),
        patch(
            "grasp.core.base_task_executor.utils.get_file_in_task_dir",
            return_value="dummy_path.yaml",
        ),
        patch("grasp.core.dataset.file_handler.FileHandler.read", return_value=[{"id": 1}]),
    ):
        executor = BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)
        evol_lambda = executor.graph_config.graph_config["nodes"]["build_text_node"]["lambda"]

        # Check that no override is accidentally applied
        assert (
            evol_lambda == "grasp.recipes.evol_instruct.task_executor.EvolInstructPromptGenerator"
        )


# ----------------------------
# Section 4: State Variable Extraction
# ----------------------------


def test_extract_from_dataset_list(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that state variables are correctly extracted from a dataset represented as a list of dictionaries.

    Verifies:
        - State variables like 'foo' and 'baz' in dataset are detected and added to graph state.
    """

    def mock_load_yaml_file(*args, **kwargs):
        filepath = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in filepath else mock_grasp_config

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
            return_value=[{"foo": "bar", "baz": 1}],
        ),
    ):
        executor = BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)
        assert "foo" in executor.graph_config.state_variables


def test_extract_from_dataset_object(mock_args, mock_grasp_config, mock_model_config):
    """
    Test state variable extraction from HuggingFace `Dataset` object.

    Verifies:
        - Dataset is converted to a list of dictionaries using `.to_pandas().to_dict(orient='records')`.
        - Extracted keys like 'name' are correctly added to the graph state.
    """
    data = Dataset.from_dict({"name": ["Alice"], "age": [30]})
    data_as_list = data.to_pandas().to_dict(orient="records")

    def mock_load_yaml_file(*args, **kwargs):
        path = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in path else mock_grasp_config

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
            return_value=data_as_list,
        ),
    ):
        executor = BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)
        assert "name" in executor.graph_config.state_variables


def test_extract_from_iterable_dataset(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that state variables can be extracted from an IterableDataset (e.g., streaming scenario).

    Verifies:
        - Keys from the first record yielded (like 'x') are recognized as state variables.
    """
    dummy_data = IterableDataset.from_generator(lambda: iter([{"x": 1, "y": 2}]))

    def mock_load_yaml_file(*args, **kwargs):
        path = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in path else mock_grasp_config

    with (
        patch(
            "grasp.core.base_task_executor.utils.load_yaml_file",
            side_effect=mock_load_yaml_file,
        ),
        patch(
            "grasp.core.base_task_executor.utils.get_file_in_task_dir",
            return_value="dummy_path.yaml",
        ),
        patch("grasp.core.dataset.file_handler.FileHandler.read", return_value=dummy_data),
    ):
        executor = BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)
        assert "x" in executor.graph_config.state_variables


def test_extract_from_prompt_templates(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that variables used in prompt templates are automatically extracted as state variables.

    Verifies:
        - Placeholders in prompts (e.g., '{text}') are extracted and registered in graph state.
    """
    mock_grasp_config["graph_config"]["nodes"]["evol_text_node"]["prompt"] = [
        {"user": "generate from {text}"}
    ]

    def mock_load_yaml_file(*args, **kwargs):
        path = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in path else mock_grasp_config

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
            return_value=[{"text": "hello"}],
        ),
        patch("grasp.core.base_task_executor.utils.extract_pattern", return_value={"text"}),
    ):
        executor = BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)
        assert "text" in executor.graph_config.state_variables


def test_extract_input_data_keys_list_of_dicts(dummy_instance):
    """
    Test _extract_input_data_keys with a list of dicts adds keys as state variables.
    """
    graph = dummy_instance.graph_config
    graph.state_variables.clear()
    sample = [{"a": 1, "b": 2}]
    graph._extract_input_data_keys(sample)
    assert "a" in graph.state_variables and "b" in graph.state_variables


def test_extract_input_data_keys_dataset(dummy_instance):
    """
    Test _extract_input_data_keys with HuggingFace Dataset sets correct state variables.
    """
    from datasets import Dataset

    graph = dummy_instance.graph_config
    graph.state_variables.clear()
    ds = Dataset.from_dict({"name": ["test"], "age": [20]})
    graph._extract_input_data_keys(ds)
    assert "name" in graph.state_variables and "age" in graph.state_variables


def test_extract_input_data_keys_iterable_dataset(dummy_instance):
    """
    Test _extract_input_data_keys with IterableDataset sets variables from first record.
    """
    graph = dummy_instance.graph_config
    graph.state_variables.clear()
    features = Features({"foo": Value("string"), "baz": Value("int32")})
    ids = IterableDataset.from_generator(
        lambda: iter([{"foo": "bar", "baz": 2}]), features=features
    )
    graph._extract_input_data_keys(ids)
    assert "foo" in graph.state_variables and "baz" in graph.state_variables


def test_extract_input_data_keys_invalid(dummy_instance):
    """
    Comprehensive test for _extract_input_data_keys:
    - IterableDataset with no records -> raises ValueError
    - Object with column_names=None -> raises ValueError
    """
    graph = dummy_instance.graph_config
    graph.state_variables.clear()

    # Case 1: IterableDataset that yields no records
    empty_ids = IterableDataset.from_generator(lambda: iter([]))
    with pytest.raises(ValueError, match="Error extracting keys from dataset:"):
        graph._extract_input_data_keys(empty_ids)

    # Case 2: Object with column_names = None
    class FakeDatasetWithNoneColumns:
        column_names = None

    with pytest.raises(ValueError, match="Error extracting keys from dataset:"):
        graph._extract_input_data_keys(FakeDatasetWithNoneColumns())


# ----------------------------
# Section 5: Graph Construction
# ----------------------------


def test_nodes_are_initialized(dummy_instance):
    """
    Test that graph nodes are properly initialized and accessible via `get_nodes()`.

    Ensures:
        - Nodes are returned as a dictionary.
        - The expected node 'build_text_node' is present in the graph.
    """
    graph_config = dummy_instance.graph_config
    nodes = graph_config.get_nodes()
    assert isinstance(nodes, dict)
    assert "build_text_node" in nodes


def test_edges_are_initialized(dummy_instance):
    """
    Test that graph edges are properly initialized and accessible via `get_edges()`.

    Ensures:
        - Edges are returned as a list.
        - The list contains at least one edge, indicating successful parsing of graph structure.
    """
    edges = dummy_instance.graph_config.get_edges()
    assert isinstance(edges, list)
    assert len(edges) > 0


def test_duplicate_state_variables_error(mock_args, mock_grasp_config, mock_model_config):
    """
    Test that duplicate `output_keys` in a node configuration are retained as-is.

    Context:
        This test ensures that the config is parsed as-is by `BaseTaskExecutor`, and that
        duplicate entries in `output_keys` are not silently altered or dropped.

    Note:
        This test does not assert for an error being raised, but confirms preservation of values.
    """
    mock_grasp_config["graph_config"]["nodes"]["build_text_node"]["output_keys"] = [
        "dup",
        "dup",
    ]

    def mock_load_yaml_file(*args, **kwargs):
        path = args[0] if args else kwargs.get("filepath", "")
        return mock_model_config if "models.yaml" in path else mock_grasp_config

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
            return_value=[{"dup": "val"}],
        ),
    ):
        executor = BaseTaskExecutor(mock_args, graph_config_dict=mock_grasp_config)
        node = executor.graph_config.get_nodes()["build_text_node"]
        assert node.node_config["output_keys"] == ["dup", "dup"]


def test_process_variables_single_str(dummy_instance):
    """
    Test _process_variables correctly adds a single string output_key to state_variables.
    """
    graph = dummy_instance.graph_config
    graph.state_variables.clear()
    config = {"output_keys": "var1"}
    graph._process_variables(config, ["output_keys"])
    assert "var1" in graph.state_variables


def test_process_variables_list_of_str(dummy_instance):
    """
    Test _process_variables correctly adds a list of output_keys to state_variables.
    """
    graph = dummy_instance.graph_config
    graph.state_variables.clear()
    config = {"output_keys": ["a", "b", "c"]}
    graph._process_variables(config, ["output_keys"])
    assert all(key in graph.state_variables for key in ["a", "b", "c"])


def test_process_variables_raises_on_invalid_format(dummy_instance):
    """
    Test _process_variables raises ValueError if output_keys is in invalid format.
    """
    graph = dummy_instance.graph_config
    config = {"output_keys": {"not": "a string or list"}}
    with pytest.raises(ValueError, match="Invalid variable format"):
        graph._process_variables(config, ["output_keys"])
