"""
GraSP: GRAph-oriented Synthetic data generation Pipeline library

A powerful Python library for building and executing complex data synthesis workflows
using graph-based architectures with LLMs, agents, and custom processing nodes.
"""

import logging
from typing import Union, Dict, Any

from .workflow import Workflow, create_graph
from .configuration import ConfigLoader, load_config
from .exceptions import (
    GraSPError,
    ValidationError,
    ExecutionError,
    ConfigurationError,
    NodeError,
    DataError,
    ModelError,
    TimeoutError,
)
from .models import ModelConfigBuilder

try:
    from .core.base_task_executor import BaseTaskExecutor, DefaultTaskExecutor
    from .core.judge_task_executor import JudgeQualityTaskExecutor
    from .core.resumable_execution import ResumableExecutionManager
    from .core.dataset.dataset_processor import DatasetProcessor
    from .core.graph.graph_config import GraphConfig
    from .core.graph.grasp_state import GraspState
    from .core.graph.grasp_message import GraspMessage

    CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core modules not available: {e}")
    CORE_AVAILABLE = False

try:
    from .core.dataset.dataset_config import (
        DataSourceConfig,
        OutputConfig,
        DataSourceType,
        OutputType,
        TransformConfig,
        ShardConfig,
    )
    from .core.dataset.file_handler import FileHandler
    from .core.dataset.huggingface_handler import HuggingFaceHandler

    DATA_HANDLERS_AVAILABLE = True
except ImportError:
    DATA_HANDLERS_AVAILABLE = False

# Node modules
try:
    from .core.graph.nodes.base_node import BaseNode, NodeType, NodeState
    from .core.graph.nodes.llm_node import LLMNode as CoreLLMNode
    from .core.graph.nodes.agent_node import AgentNode as CoreAgentNode
    from .core.graph.nodes.multi_llm_node import MultiLLMNode as CoreMultiLLMNode
    from .core.graph.nodes.weighted_sampler_node import (
        WeightedSamplerNode as CoreWeightedSamplerNode,
    )

    NODES_AVAILABLE = True
except ImportError:
    NODES_AVAILABLE = False

# Model factory modules
try:
    from .core.models.model_factory import ModelFactory
    from .core.models.structured_output.structured_output_config import (
        StructuredOutputConfig,
    )
    from .core.models.structured_output.schemas_factory import SimpleResponse

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Utility modules
try:
    from . import utils
    from .utils import utils
    from .utils import constants
    from .logger.logger_config import (
        logger,
        set_external_logger,
        reset_to_internal_logger,
    )

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Import node builders
try:
    from .nodes import (
        LLMNodeBuilder,
        AgentNodeBuilder,
        MultiLLMNodeBuilder,
        LambdaNodeBuilder,
        WeightedSamplerNodeBuilder,
        SubgraphNodeBuilder,
    )

    NODE_BUILDERS_AVAILABLE = True
except ImportError:
    NODE_BUILDERS_AVAILABLE = False

# Import data utilities
try:
    from .data import (
        DataSource,
        DataSink,
        DataSourceFactory,
        DataSinkFactory,
        from_file,
        from_huggingface,
        to_file,
        to_huggingface,
    )

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
        .agent(model, prompt, tools)
        .sink(output)
    )


def quick_multi_llm(
    models: Dict[str, str], prompt: str, data_source: str, output: str = "output.json"
):
    """Quick multi-LLM workflow creation."""
    return (
        Workflow("quick_multi_llm")
        .source(data_source)
        .multi_llm(models, prompt)
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


def create_structured_schema(
    fields: Dict[str, str], name: str = "CustomSchema"
) -> Dict[str, Any]:
    """Create structured output schema configuration."""
    return {
        "enabled": True,
        "schema": {
            "name": name,
            "fields": {
                field_name: {"type": field_type}
                for field_name, field_type in fields.items()
            },
        },
    }


def pydantic_schema(model_class: str) -> Dict[str, Any]:
    """Create structured output schema from Pydantic model class path."""
    return {"enabled": True, "schema": model_class}


def create_processor_config(
    processor: Union[str, callable], **params
) -> Dict[str, Any]:
    """Create processor configuration."""
    if callable(processor):
        processor_path = f"{processor.__module__}.{processor.__name__}"
    else:
        processor_path = processor

    config = {"processor": processor_path}
    if params:
        config["params"] = params

    return config


def create_transformation_config(
    transform: Union[str, callable], **params
) -> Dict[str, Any]:
    """Create data transformation configuration."""
    if callable(transform):
        transform_path = f"{transform.__module__}.{transform.__name__}"
    else:
        transform_path = transform

    config = {"transform": transform_path}
    if params:
        config["params"] = params

    return config


def get_version() -> str:
    """Get library version."""
    return __version__


def setup_logging(level: str = "INFO") -> None:
    """Setup logging."""
    logging.getLogger("grasp").setLevel(getattr(logging, level.upper()))


def validate_environment() -> Dict[str, bool]:
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


def get_info() -> Dict[str, Any]:
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
        model_configs = utils.load_model_config()
        return list(model_configs.keys())
    except Exception as e:
        return [f"Error loading models: {e}"]


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    if not UTILS_AVAILABLE:
        return {"error": "Framework not available"}

    try:
        model_configs = utils.load_model_config()
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
