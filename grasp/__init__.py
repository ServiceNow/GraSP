"""
GraSP: GRAph-oriented Synthetic data generation Pipeline library

A powerful Python library for building and executing complex data synthesis workflows
using graph-based architectures with LLMs, agents, and custom processing nodes.
"""

import logging
from typing import Any, Callable, Union

from .configuration import ConfigLoader, load_config
from .exceptions import (
    ConfigurationError,
    DataError,
    ExecutionError,
    GraSPError,
    ModelError,
    NodeError,
    TimeoutError,
    ValidationError,
)
from .models import ModelConfigBuilder
from .workflow import Workflow, create_graph

try:
    from .core.base_task_executor import BaseTaskExecutor, DefaultTaskExecutor  # noqa: F401
    from .core.dataset.dataset_processor import DatasetProcessor  # noqa: F401
    from .core.graph.graph_config import GraphConfig  # noqa: F401
    from .core.graph.grasp_message import GraspMessage  # noqa: F401
    from .core.graph.grasp_state import GraspState  # noqa: F401
    from .core.judge_task_executor import JudgeQualityTaskExecutor  # noqa: F401
    from .core.resumable_execution import ResumableExecutionManager  # noqa: F401

    CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core modules not available: {e}")
    CORE_AVAILABLE = False

try:
    from .core.dataset.dataset_config import DataSourceConfig  # noqa: F401
    from .core.dataset.dataset_config import DataSourceType  # noqa: F401
    from .core.dataset.dataset_config import OutputConfig  # noqa: F401
    from .core.dataset.dataset_config import OutputType  # noqa: F401
    from .core.dataset.dataset_config import ShardConfig  # noqa: F401
    from .core.dataset.dataset_config import TransformConfig  # noqa: F401
    from .core.dataset.file_handler import FileHandler  # noqa: F401
    from .core.dataset.huggingface_handler import HuggingFaceHandler  # noqa: F401

    DATA_HANDLERS_AVAILABLE = True
except ImportError:
    DATA_HANDLERS_AVAILABLE = False

# Node modules
try:
    from .core.graph.nodes.agent_node import AgentNode as CoreAgentNode  # noqa: F401
    from .core.graph.nodes.base_node import BaseNode, NodeState, NodeType  # noqa: F401
    from .core.graph.nodes.llm_node import LLMNode as CoreLLMNode  # noqa: F401
    from .core.graph.nodes.multi_llm_node import MultiLLMNode as CoreMultiLLMNode  # noqa: F401
    from .core.graph.nodes.weighted_sampler_node import (  # noqa: F401
        WeightedSamplerNode as CoreWeightedSamplerNode,
    )

    NODES_AVAILABLE = True
except ImportError:
    NODES_AVAILABLE = False

# Model factory modules
try:
    from .core.models.model_factory import ModelFactory  # noqa: F401
    from .core.models.structured_output.schemas_factory import SimpleResponse  # noqa: F401
    from .core.models.structured_output.structured_output_config import (  # noqa: F401
        StructuredOutputConfig,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Utility modules
try:
    from . import utils
    from .logger.logger_config import logger  # noqa: F401
    from .logger.logger_config import reset_to_internal_logger  # noqa: F401
    from .logger.logger_config import set_external_logger  # noqa: F401
    from .utils import constants  # noqa: F401

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Import node builders
try:
    from .nodes import AgentNodeBuilder  # noqa: F401
    from .nodes import LambdaNodeBuilder  # noqa: F401
    from .nodes import LLMNodeBuilder  # noqa: F401
    from .nodes import MultiLLMNodeBuilder  # noqa: F401
    from .nodes import SubgraphNodeBuilder  # noqa: F401
    from .nodes import WeightedSamplerNodeBuilder  # noqa: F401

    NODE_BUILDERS_AVAILABLE = True
except ImportError:
    NODE_BUILDERS_AVAILABLE = False

# Import data utilities
try:
    from .data import DataSink  # noqa: F401
    from .data import DataSinkFactory  # noqa: F401
    from .data import DataSource  # noqa: F401
    from .data import DataSourceFactory  # noqa: F401
    from .data import from_file  # noqa: F401
    from .data import from_huggingface  # noqa: F401
    from .data import to_file  # noqa: F401
    from .data import to_huggingface  # noqa: F401

    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False


__version__ = "1.0.0"
__author__ = "GraSP Team"
__description__ = "GRAph-oriented Synthetic data generation Pipeline library"


# Quick utility functions
def quick_llm(model: str, prompt: str, data_source: str, output: str = "output.json"):
    """Quick LLM workflow creation."""
    return (
        Workflow(f"quick_llm_{model.replace('/', '_')}")
        .source(data_source)
        .llm(model, prompt)
        .sink(output)
    )


def quick_agent(
    model: str,
    prompt: str,
    tools: list[str],
    data_source: str,
    output: str = "output.json",
):
    """Quick agent workflow creation."""
    return (
        Workflow(f"quick_agent_{model.replace('/', '_')}")
        .source(data_source)
        .agent(model=model, tools=tools, prompt=prompt)
        .sink(output)
    )


def quick_multi_llm(
    models: dict[str, Any], prompt: str, data_source: str, output: str = "output.json"
):
    """Quick multi-LLM workflow creation."""
    return (
        Workflow("quick_multi_llm")
        .source(data_source)
        .multi_llm(models=models, prompt=prompt)
        .sink(output)
    )


def execute_task(task_name: str, **kwargs):
    """Execute an existing task configuration."""
    workflow = Workflow(task_name)
    return workflow.run(**kwargs)


def create_multimodal_workflow(name: str) -> Workflow:
    """Create workflow with multimodal capabilities enabled."""
    workflow = Workflow(name)
    workflow._supports_multimodal = True
    return workflow


def create_resumable_workflow(name: str) -> Workflow:
    """Create workflow with resumable execution enabled."""
    workflow = Workflow(name)
    workflow.resumable(True)
    return workflow


def create_quality_workflow(name: str) -> Workflow:
    """Create workflow with quality tagging enabled."""
    workflow = Workflow(name)
    workflow.quality_tagging(True)
    return workflow


def create_chat_workflow(name: str, conversation_type: str = "multiturn") -> Workflow:
    """Create workflow optimized for chat/conversation generation."""
    workflow = Workflow(name)
    workflow.chat_conversation(conversation_type)
    return workflow


def create_structured_schema(fields: dict[str, str], name: str = "CustomSchema") -> dict[str, Any]:
    """Create structured output schema configuration."""
    return {
        "enabled": True,
        "schema": {
            "name": name,
            "fields": {
                field_name: {"type": field_type} for field_name, field_type in fields.items()
            },
        },
    }


def pydantic_schema(model_class: str) -> dict[str, Any]:
    """Create structured output schema from Pydantic model class path."""
    return {"enabled": True, "schema": model_class}


def create_processor_config(processor: Union[str, Callable], **params) -> dict[str, Any]:
    """Create processor configuration."""
    if callable(processor):
        processor_path = f"{processor.__module__}.{processor.__name__}"
    else:
        processor_path = processor

    config: dict[str, Any] = {"processor": processor_path}
    if params:
        config["params"] = params

    return config


def create_transformation_config(transform: Union[str, Callable], **params) -> dict[str, Any]:
    """Create data transformation configuration."""
    if callable(transform):
        transform_path = f"{transform.__module__}.{transform.__name__}"
    else:
        transform_path = transform

    config: dict[str, Any] = {"transform": transform_path}
    if params:
        config["params"] = params

    return config


def get_version() -> str:
    """Get library version."""
    return __version__


def setup_logging(level: str = "INFO") -> None:
    """Setup logging."""
    logging.getLogger("grasp").setLevel(getattr(logging, level.upper()))


def validate_environment() -> dict[str, bool]:
    """Validate environment setup."""
    return {
        "core_available": CORE_AVAILABLE,
        "data_handlers_available": DATA_HANDLERS_AVAILABLE,
        "nodes_available": NODES_AVAILABLE,
        "models_available": MODELS_AVAILABLE,
        "utils_available": UTILS_AVAILABLE,
        "node_builders_available": NODE_BUILDERS_AVAILABLE,
        "data_utils_available": DATA_UTILS_AVAILABLE,
    }


def get_info() -> dict[str, Any]:
    """Get library information and feature availability."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "features": validate_environment(),
    }


def list_available_models() -> list[str]:
    """list available models from framework configuration."""
    if not UTILS_AVAILABLE:
        return ["Framework not available - cannot list models"]

    try:
        model_configs = utils.utils.load_model_config()
        return list(model_configs.keys())
    except Exception as e:
        return [f"Error loading models: {e}"]


def get_model_info(model_name: str) -> dict[str, Any]:
    """Get information about a specific model."""
    if not UTILS_AVAILABLE:
        return {"error": "Framework not available"}

    try:
        # Ensure the mapping type is explicit so mypy can infer correct return type
        model_configs: dict[str, dict[str, Any]] = utils.utils.load_model_config()
        return model_configs.get(model_name, {"error": f"Model {model_name} not found"})
    except Exception as e:
        return {"error": f"Error loading model info: {e}"}


# Main exports
__all__ = [
    # Main classes
    "Workflow",
    "create_graph",
    # Configuration
    "load_config",
    "ConfigLoader",
    # Model utilities
    "ModelConfigBuilder",
    # Quick functions
    "quick_llm",
    "quick_agent",
    "quick_multi_llm",
    "execute_task",
    # Advanced workflow builders
    "create_multimodal_workflow",
    "create_resumable_workflow",
    "create_quality_workflow",
    "create_chat_workflow",
    # Structured output helpers
    "create_structured_schema",
    "pydantic_schema",
    # Processing helpers
    "create_processor_config",
    "create_transformation_config",
    # Utilities
    "get_version",
    "setup_logging",
    "validate_environment",
    "get_info",
    "list_available_models",
    "get_model_info",
    # Exceptions
    "GraSPError",
    "ValidationError",
    "ExecutionError",
    "ConfigurationError",
    "NodeError",
    "DataError",
    "ModelError",
    "TimeoutError",
]

# Add conditionally available imports to __all__
if CORE_AVAILABLE:
    __all__.extend(
        [
            "BaseTaskExecutor",
            "DefaultTaskExecutor",
            "JudgeQualityTaskExecutor",
            "GraphConfig",
            "GraspState",
            "GraspMessage",
            "ResumableExecutionManager",
            "DatasetProcessor",
        ]
    )

if DATA_HANDLERS_AVAILABLE:
    __all__.extend(
        [
            "DataSourceConfig",
            "OutputConfig",
            "DataSourceType",
            "OutputType",
            "TransformConfig",
            "ShardConfig",
            "FileHandler",
            "HuggingFaceHandler",
        ]
    )

if NODES_AVAILABLE:
    __all__.extend(
        [
            "BaseNode",
            "NodeType",
            "NodeState",
            "CoreLLMNode",
            "CoreAgentNode",
            "CoreMultiLLMNode",
            "CoreWeightedSamplerNode",
        ]
    )

if MODELS_AVAILABLE:
    __all__.extend(["ModelFactory", "StructuredOutputConfig", "SimpleResponse"])

if UTILS_AVAILABLE:
    __all__.extend(
        [
            "utils",
            "constants",
            "logger",
            "set_external_logger",
            "reset_to_internal_logger",
        ]
    )

if NODE_BUILDERS_AVAILABLE:
    __all__.extend(
        [
            "LLMNodeBuilder",
            "AgentNodeBuilder",
            "MultiLLMNodeBuilder",
            "LambdaNodeBuilder",
            "WeightedSamplerNodeBuilder",
            "SubgraphNodeBuilder",
        ]
    )

if DATA_UTILS_AVAILABLE:
    __all__.extend(
        [
            "DataSource",
            "DataSink",
            "DataSourceFactory",
            "DataSinkFactory",
            "from_file",
            "from_huggingface",
            "to_file",
            "to_huggingface",
        ]
    )
