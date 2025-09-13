import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml

from grasp.processors.output_record_generator import BaseOutputGenerator

try:
    from argparse import Namespace

    from grasp.core.base_task_executor import BaseTaskExecutor, DefaultTaskExecutor
    from grasp.core.dataset.dataset_config import DataSourceConfig  # noqa: F401
    from grasp.core.dataset.dataset_config import DataSourceType  # noqa: F401
    from grasp.core.dataset.dataset_config import OutputConfig  # noqa: F401
    from grasp.core.dataset.dataset_config import OutputType  # noqa: F401
    from grasp.core.graph.functions.node_processor import (
        NodePostProcessor,
        NodePostProcessorWithState,
        NodePreProcessor,
    )
    from grasp.core.graph.grasp_message import GraspMessage  # noqa: F401
    from grasp.core.graph.grasp_state import GraspState  # noqa: F401
    from grasp.core.judge_task_executor import JudgeQualityTaskExecutor
    from grasp.logger.logger_config import logger
    from grasp.utils import constants  # noqa: F401
    from grasp.utils import utils as utils

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    from grasp.logger.logger_config import logger

from grasp.exceptions import ConfigurationError, ExecutionError, GraSPError
from grasp.models import ModelConfigBuilder

# Import node builders with conditional availability
try:
    from grasp.nodes import (
        AgentNodeBuilder,
        LambdaNodeBuilder,
        LLMNodeBuilder,
        MultiLLMNodeBuilder,
        SubgraphNodeBuilder,
        WeightedSamplerNodeBuilder,
    )

    NODE_BUILDERS_AVAILABLE = True
except ImportError:
    NODE_BUILDERS_AVAILABLE = False

# Import data utilities with conditional availability
try:
    from grasp.data import DataSink, DataSource

    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False


class Workflow:
    """
    Unified workflow builder supporting all GraSP paradigms and use cases.

    Supports:
    - Fluent API building with custom processors
    - Existing YAML task execution
    - Explicit graph building with node builders
    - Direct function references and lambda processing
    - Complex conditional workflows and edge routing

    Examples:
        # Fluent API with custom processors
        >>> workflow = Workflow("my_workflow")
        >>> result = workflow.source(data) \\
        >>>         .llm("gpt-4", "Create haiku: {text}",
        >>>              post_process=TextExtractionPostProcessor) \\
        >>>         .sink("output.json") \\
        >>>         .run()

        # Execute existing YAML task
        >>> workflow = Workflow("tasks/examples/glaive_code_assistant")
        >>> workflow.run(num_records=1)

        # Explicit graph building
        >>> graph = create_graph("complex_workflow")
        >>> summarizer = graph.add_llm_node("summarizer", "gpt-4o") \\
        >>>                   .system_message("Create summaries") \\
        >>>                   .user_message("Summarize: {text}")
        >>> graph.sequence("START", "summarizer", "END")
        >>> results = graph.run()

        # Direct function references
        >>> workflow.source(data) \\
        >>>         .lambda_func(TextCleanerLambda, output="cleaned") \\
        >>>         .llm("gpt-4", "Process: {cleaned}",
        >>>              pre_process=PromptTemplatePreProcessor) \\
        >>>         .run()
    """

    def __init__(self, name: Optional[str] = None):
        self.name: str = name or f"workflow_{uuid.uuid4().hex[:8]}"
        self._config: dict[str, Any] = {
            "graph_config": {"nodes": {}, "edges": [], "graph_properties": {}},
            "data_config": {},
            "output_config": {},
        }
        self._node_counter: int = 0
        self._last_node: Optional[str] = None
        self._temp_files: list[str] = []
        self._overrides: dict[str, Any] = {}
        self._node_builders: dict[str, Any] = {}
        self._messages: list[str] = []

        # Feature support flags
        self._supports_subgraphs = True
        self._supports_multimodal = True
        self._supports_resumable = True
        self._supports_quality = True
        self._supports_oasst = True

    def source(
        self, source: Union[str, Path, dict[str, Any], list[dict[str, Any]], DataSource]
    ) -> "Workflow":
        """Add data source with full framework support."""
        if isinstance(source, (str, Path)):
            source_config = {
                "type": "disk",
                "file_path": str(source),
                "file_format": self._detect_file_format(str(source)),
            }
        elif isinstance(source, list):
            temp_file = self._create_temp_file(source)
            source_config = {
                "type": "disk",
                "file_path": temp_file,
                "file_format": "json",
            }
        elif isinstance(source, dict):
            if "data" in source:
                temp_file = self._create_temp_file(source["data"])
                source_config = {
                    "type": "disk",
                    "file_path": temp_file,
                    "file_format": "json",
                }
            else:
                source_config = source
        elif DATA_UTILS_AVAILABLE and isinstance(source, DataSource):
            source_config = source.to_config()
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        self._config["data_config"]["source"] = source_config
        return self

    def sink(self, sink: Union[str, Path, dict[str, Any], DataSink]) -> "Workflow":
        """Add data sink with full framework support."""
        if isinstance(sink, (str, Path)):
            output_path = str(sink)
            path_dir = os.path.dirname(output_path)
            if path_dir == "":
                path_dir = os.getcwd()
            os.makedirs(path_dir, exist_ok=True)
            sink_config = {
                "type": "json" if output_path.endswith(".json") else "jsonl",
                "file_path": output_path,
            }
        elif isinstance(sink, dict):
            sink_config = sink
        elif DATA_UTILS_AVAILABLE and isinstance(sink, DataSink):
            sink_config = sink.to_config()
        else:
            raise ValueError(f"Unsupported sink type: {type(sink)}")

        self._config["data_config"]["sink"] = sink_config
        return self

    def llm(
        self,
        model: Union[str, dict[str, Any]],
        prompt: Union[str, list[dict[str, str]]],
        output: str = "messages",
        pre_process: Optional[Union[str, Callable, NodePreProcessor]] = None,
        post_process: Optional[
            Union[str, Callable, NodePostProcessor, NodePostProcessorWithState]
        ] = None,
        **kwargs,
    ) -> "Workflow":
        """Add LLM node with full feature support including custom processors."""
        node_name: str = self._generate_node_name("llm")

        if isinstance(model, str):
            model_config = ModelConfigBuilder.from_name(model, **kwargs)
        else:
            model_config = ModelConfigBuilder.validate_config(model)

        if isinstance(prompt, str):
            prompt_config = [{"user": prompt}]
        else:
            prompt_config = prompt

        node_config: dict[str, Any] = {
            "node_type": "llm",
            "model": model_config,
            "prompt": prompt_config,
        }

        if output != "messages":
            node_config["output_keys"] = output

        if kwargs.get("chat_history"):
            node_config["chat_history"] = True

        # Handle pre-processors
        if pre_process:
            if isinstance(pre_process, type) and issubclass(pre_process, NodePreProcessor):
                # Class reference - convert to string path
                node_config["pre_process"] = f"{pre_process.__module__}.{pre_process.__name__}"
            elif callable(pre_process):
                node_config["pre_process"] = self._callable_to_string_path(pre_process)
            else:
                node_config["pre_process"] = pre_process

        # Handle post-processors
        if post_process:
            if isinstance(post_process, type) and (
                issubclass(post_process, NodePostProcessor)
                or issubclass(post_process, NodePostProcessorWithState)
            ):
                # Class reference - convert to string path
                node_config["post_process"] = f"{post_process.__module__}.{post_process.__name__}"
            elif callable(post_process):
                node_config["post_process"] = self._callable_to_string_path(post_process)
            else:
                node_config["post_process"] = post_process

        if kwargs.get("structured_output"):
            node_config["structured_output"] = kwargs["structured_output"]

        self._add_node_with_edge(node_name, node_config)
        return self

    def multi_llm(
        self,
        models: dict[str, Any],
        prompt: Union[str, list[dict[str, str]]],
        **kwargs,
    ) -> "Workflow":
        """Add Multi-LLM node."""
        node_name = self._generate_node_name("multi_llm")

        models_config = {}
        for label, model in models.items():
            if isinstance(model, str):
                models_config[label] = ModelConfigBuilder.from_name(model)
            else:
                models_config[label] = ModelConfigBuilder.validate_config(model)

        if isinstance(prompt, str):
            prompt_config = [{"user": prompt}]
        else:
            prompt_config = prompt

        node_config = {
            "node_type": "multi_llm",
            "models": models_config,
            "prompt": prompt_config,
        }

        if kwargs.get("output_keys"):
            node_config["output_keys"] = kwargs["output_keys"]

        if kwargs.get("multi_llm_post_process"):
            node_config["multi_llm_post_process"] = kwargs["multi_llm_post_process"]

        self._add_node_with_edge(node_name, node_config)
        return self

    def agent(
        self,
        model: Union[str, dict[str, Any]],
        tools: list[Any],
        prompt: Union[str, list[dict[str, str]]],
        **kwargs,
    ) -> "Workflow":
        """Add agent node with full feature support."""
        node_name = self._generate_node_name("agent")

        if isinstance(model, str):
            model_config = ModelConfigBuilder.from_name(model, **kwargs)
        else:
            model_config = ModelConfigBuilder.validate_config(model)

        node_config = {
            "node_type": "agent",
            "model": model_config,
            "tools": tools,
            "prompt": [{"user": prompt}] if isinstance(prompt, str) else prompt,
        }

        if kwargs.get("inject_system_messages"):
            node_config["inject_system_messages"] = kwargs["inject_system_messages"]

        if kwargs.get("chat_history"):
            node_config["chat_history"] = True  # type: ignore[assignment]

        self._add_node_with_edge(node_name, node_config)
        return self

    def lambda_func(
        self, func: Union[str, Callable, type], output: str = "result", **kwargs
    ) -> "Workflow":
        """Add lambda function node supporting direct function/class references."""
        node_name = self._generate_node_name("lambda")

        # Handle different types of function references
        if isinstance(func, type):
            # Class reference
            func_path = f"{func.__module__}.{func.__name__}"
        elif callable(func):
            func_path = self._callable_to_string_path(func)
        else:
            func_path = func

        node_config = {
            "node_type": "lambda",
            "lambda": func_path,
            "output_keys": output,
        }

        self._add_node_with_edge(node_name, node_config)
        return self

    def weighted_sampler(self, attributes: dict[str, dict[str, Any]], **kwargs) -> "Workflow":
        """Add weighted sampler node."""
        node_name = self._generate_node_name("weighted_sampler")

        node_config = {"node_type": "weighted_sampler", "attributes": attributes}

        self._add_node_with_edge(node_name, node_config)
        return self

    def subgraph(
        self, subgraph_name: str, node_config_map: Optional[dict[str, Any]] = None
    ) -> "Workflow":
        """Add subgraph node."""
        node_name = self._generate_node_name("subgraph")

        node_config: dict[str, Any] = {"node_type": "subgraph", "subgraph": subgraph_name}

        if node_config_map:
            node_config["node_config_map"] = node_config_map

        self._add_node_with_edge(node_name, node_config)
        return self

    def add_node(self, name: str, node_config: dict[str, Any]) -> "Workflow":
        """Add node with explicit configuration."""
        self._config["graph_config"]["nodes"][name] = node_config
        return self

    def add_llm_node(self, name: str, model: Union[str, dict[str, Any]]) -> "LLMNodeBuilder":
        """Add LLM node and return builder for chaining."""
        if not NODE_BUILDERS_AVAILABLE:
            raise GraSPError("Node builders not available.")

        builder = LLMNodeBuilder(name, model)
        self._node_builders[name] = builder
        return builder

    def add_agent_node(self, name: str, model: Union[str, dict[str, Any]]) -> "AgentNodeBuilder":
        """Add agent node and return builder for chaining."""
        if not NODE_BUILDERS_AVAILABLE:
            raise GraSPError("Node builders not available.")

        builder = AgentNodeBuilder(name, model)
        self._node_builders[name] = builder
        return builder

    def add_multi_llm_node(self, name: str) -> "MultiLLMNodeBuilder":
        """Add multi-LLM node and return builder for chaining."""
        if not NODE_BUILDERS_AVAILABLE:
            raise GraSPError("Node builders not available.")

        builder = MultiLLMNodeBuilder(name)
        self._node_builders[name] = builder
        return builder

    def add_lambda_node(self, name: str, func: Union[str, Callable]) -> "LambdaNodeBuilder":
        """Add lambda node and return builder for chaining."""
        if not NODE_BUILDERS_AVAILABLE:
            raise GraSPError("Node builders not available.")

        builder = LambdaNodeBuilder(name, func)
        self._node_builders[name] = builder
        return builder

    def add_weighted_sampler_node(self, name: str) -> "WeightedSamplerNodeBuilder":
        """Add weighted sampler node and return builder for chaining."""
        if not NODE_BUILDERS_AVAILABLE:
            raise GraSPError("Node builders not available.")

        builder = WeightedSamplerNodeBuilder(name)
        self._node_builders[name] = builder
        return builder

    def add_subgraph_node(self, name: str, subgraph: str) -> "SubgraphNodeBuilder":
        """Add subgraph node and return builder for chaining."""
        if not NODE_BUILDERS_AVAILABLE:
            raise GraSPError("Node builders not available.")

        builder = SubgraphNodeBuilder(name, subgraph)
        self._node_builders[name] = builder
        return builder

    def add_edge(self, from_node: str, to_node: str) -> "Workflow":
        """Add simple edge between nodes."""
        edge_config = {"from": from_node, "to": to_node}
        self._config["graph_config"]["edges"].append(edge_config)
        return self

    def add_conditional_edge(
        self, from_node: str, condition: Union[str, Callable], path_map: dict[str, str]
    ) -> "Workflow":
        """Add conditional edge with path mapping."""
        if callable(condition):
            condition_path = self._callable_to_string_path(condition)
        else:
            condition_path = condition

        edge_config = {
            "from": from_node,
            "condition": condition_path,
            "path_map": path_map,
        }
        self._config["graph_config"]["edges"].append(edge_config)
        return self

    def sequence(self, *nodes: str) -> "Workflow":
        """Connect nodes in sequence. Adds START and END nodes if not already added."""
        if nodes[0] != "START":
            self.add_edge("START", nodes[0])
        if nodes[-1] != "END":
            self.add_edge(nodes[-1], "END")
        for i in range(len(nodes) - 1):
            self.add_edge(nodes[i], nodes[i + 1])
        return self

    def build(self) -> "Workflow":
        """Build workflow by applying all node builders and return self for execution."""
        # Apply all node builders to finalize configuration
        for name, builder in self._node_builders.items():
            node_config = builder.build()
            self._config["graph_config"]["nodes"][name] = node_config

        # Clear builders since they're now applied
        self._node_builders.clear()
        return self

    def resumable(self, enabled: bool = True) -> "Workflow":
        """Enable/disable resumable execution."""
        if "data_config" not in self._config:
            self._config["data_config"] = {}
        self._config["data_config"]["resumable"] = enabled
        return self

    def quality_tagging(
        self, enabled: bool = True, config: Optional[dict[str, Any]] = None
    ) -> "Workflow":
        """Enable quality tagging."""
        if "output_config" not in self._config:
            self._config["output_config"] = {}

        if enabled:
            try:
                from grasp.utils import utils

                post_generation_tasks = utils.load_yaml_file("config/grasp.yaml")[
                    "post_generation_tasks"
                ]
                quality_config = config or post_generation_tasks["data_quality"]
                self._config["output_config"]["data_quality"] = quality_config
            except Exception:
                self._config["output_config"]["data_quality"] = config or {
                    "enabled": True,
                    "metrics": ["coherence", "relevance", "factuality"],
                }
        return self

    def oasst_mapping(
        self, enabled: bool = True, config: Optional[dict[str, Any]] = None
    ) -> "Workflow":
        """Enable OASST mapping."""
        if "output_config" not in self._config:
            self._config["output_config"] = {}

        if enabled:
            try:
                from grasp.utils import utils

                post_generation_tasks = utils.load_yaml_file("config/grasp.yaml")[
                    "post_generation_tasks"
                ]
                oasst_config = config or post_generation_tasks["oasst_mapper"]
                self._config["output_config"]["oasst_mapper"] = oasst_config
            except Exception:
                # Fallback configuration
                self._config["output_config"]["oasst_mapper"] = config or {
                    "required": "yes",
                    "intermediate_writing": "no",
                }
        return self

    def id_column(self, column: str) -> "Workflow":
        """Set ID column for data."""
        if "data_config" not in self._config:
            self._config["data_config"] = {}
        self._config["data_config"]["id_column"] = column
        return self

    def transformations(self, transforms: list[dict[str, Any]]) -> "Workflow":
        """Add data transformations."""
        if "data_config" not in self._config:
            self._config["data_config"] = {}
        if "source" not in self._config["data_config"]:
            self._config["data_config"]["source"] = {}

        transformations: list = []

        for transform in transforms:
            if callable(transform):
                transformations.append(self._callable_to_string_path(transform))
            else:
                transformations.append(transform)

        self._config["data_config"]["source"]["transformations"] = transformations
        return self

    def graph_properties(self, properties: dict[str, Any]) -> "Workflow":
        """Set graph properties like chat conversation type."""
        if "graph_config" not in self._config:
            self._config["graph_config"] = {}
        if "graph_properties" not in self._config["graph_config"]:
            self._config["graph_config"]["graph_properties"] = {}

        self._config["graph_config"]["graph_properties"].update(properties)
        return self

    def chat_conversation(self, conv_type: str = "multiturn", window_size: int = 5) -> "Workflow":
        """Configure chat conversation settings."""
        return self.graph_properties(
            {"chat_conversation": conv_type, "chat_history_window_size": window_size}
        )

    def disable_default_transforms(self) -> "Workflow":
        """Disable default transformations."""
        if "data_config" not in self._config:
            self._config["data_config"] = {}
        if "source" not in self._config["data_config"]:
            self._config["data_config"]["source"] = {}

        self._config["data_config"]["source"]["transformations"] = []
        return self

    def enable_resumable(self, enabled: bool = True) -> "Workflow":
        """Enable resumable execution (backward compatibility)."""
        return self.resumable(enabled)

    def enable_quality_tagging(
        self, enabled: bool = True, config: Optional[dict[str, Any]] = None
    ) -> "Workflow":
        """Enable quality tagging (backward compatibility)."""
        return self.quality_tagging(enabled, config)

    def enable_oasst_mapping(
        self, enabled: bool = True, config: Optional[dict[str, Any]] = None
    ) -> "Workflow":
        """Enable OASST mapping (backward compatibility)."""
        return self.oasst_mapping(enabled, config)

    def set_output_generator(self, generator: Union[str, type, BaseOutputGenerator]) -> "Workflow":
        """Set output generator (backward compatibility)."""
        return self.output_generator(generator)

    def set_chat_conversation(
        self, conv_type: str = "multiturn", window_size: int = 5
    ) -> "Workflow":
        """Configure chat conversation settings (backward compatibility)."""
        return self.chat_conversation(conv_type, window_size)

    def override(self, path: str, value: Any) -> "Workflow":
        """
        Universal override method using dot notation paths.

        Examples:
            .override("graph_config.nodes.llm_1.model.parameters.temperature", 0.9)
            .override("graph_config.nodes.llm_1.prompt.0.user", "New prompt: {text}")
            .override("graph_config.nodes.llm_1.model.name", "gpt-4o")
            .override("data_config.source.repo_id", "new/dataset")
        """
        if not hasattr(self, "_overrides"):
            self._overrides = {}

        self._overrides[path] = value
        return self

    def override_model(
        self, node_name: str, model_name: Optional[str] = None, **params
    ) -> "Workflow":
        """Convenient method for model overrides."""
        if model_name:
            self.override(f"graph_config.nodes.{node_name}.model.name", model_name)

        for param, value in params.items():
            self.override(f"graph_config.nodes.{node_name}.model.parameters.{param}", value)

        return self

    def output_generator(self, generator: Union[str, type, BaseOutputGenerator]) -> "Workflow":
        """Set output record generator cleanly."""
        if "output_config" not in self._config:
            self._config["output_config"] = {}

        if isinstance(generator, str):
            self._config["output_config"]["generator"] = generator
        elif isinstance(generator, type):
            self._config["output_config"][
                "generator"
            ] = f"{generator.__module__}.{generator.__name__}"

        return self

    def preserve_fields(self, *field_names: str) -> "Workflow":
        """Preserve input fields in output."""
        if "output_config" not in self._config:
            self._config["output_config"] = {}
        if "output_map" not in self._config["output_config"]:
            self._config["output_config"]["output_map"] = {}

        for field in field_names:
            self._config["output_config"]["output_map"][field] = {"from": field}

        return self

    def output_field(
        self,
        name: str,
        from_key: Optional[str] = None,
        value: Optional[Any] = None,
        transform: Optional[str] = None,
    ) -> "Workflow":
        """Add custom output field."""
        if "output_config" not in self._config:
            self._config["output_config"] = {}
        if "output_map" not in self._config["output_config"]:
            self._config["output_config"]["output_map"] = {}

        field_config = {}
        if from_key:
            field_config["from"] = from_key
        elif value is not None:
            field_config["value"] = value

        if transform:
            field_config["transform"] = transform

        self._config["output_config"]["output_map"][name] = field_config
        return self

    def output_metadata(self, metadata: dict[str, Any]) -> "Workflow":
        """
        Add static metadata to all output records.

        Args:
            metadata: Dictionary of metadata to add

        Returns:
            Workflow: Self for method chaining

        Example:
            >>> workflow.output_metadata({
            ...     "workflow_version": "1.0",
            ...     "processing_date": "2024-01-01"
            ... })
        """
        for key, value in metadata.items():
            self.output_field(key, value=value)

        return self

    def override_prompt(
        self, node_name: str, role: str, content: str, index: int = 0
    ) -> "Workflow":
        """Convenient method for prompt overrides."""
        self.override(f"graph_config.nodes.{node_name}.prompt.{index}.{role}", content)
        return self

    def run(
        self,
        num_records: Optional[int] = None,
        start_index: int = 0,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Execute workflow with full framework feature support."""
        if not CORE_AVAILABLE:
            raise GraSPError("Core framework not available")

        # Apply node builders before execution
        self.build()

        has_source = "source" in self._config.get("data_config", {})
        has_nodes = len(self._config["graph_config"]["nodes"]) > 0

        if has_source and has_nodes:
            return self._execute_programmatic_workflow(
                num_records, start_index, output_dir, **kwargs
            )
        elif not has_source and not has_nodes:
            return self._execute_existing_task(num_records, start_index, output_dir, **kwargs)
        else:
            raise ConfigurationError("Incomplete workflow. Add both source and processing nodes.")

    def save_config(self, path: Union[str, Path]) -> None:
        """Save workflow configuration as YAML file."""
        # Apply all node builders first
        self.build()

        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def _execute_existing_task(
        self,
        num_records: Optional[int],
        start_index: int,
        output_dir: Optional[str],
        **kwargs,
    ) -> Any:
        """Execute existing YAML-based task with full feature support."""
        task_name: str = self.name
        current_task: str = task_name
        utils.current_task = task_name

        config_file = f"{task_name}/graph_config.yaml"

        if not os.path.isabs(task_name):
            config_file = os.path.join(os.getcwd(), config_file)

        if not os.path.exists(config_file):
            raise ConfigurationError(
                f"Task '{task_name}' not found.\n"
                f"Expected config file: {config_file}\n"
                f"Or create a programmatic workflow: workflow.source(...).llm(...).run()"
            )

        logger.info(f"Executing existing YAML task with full features: {task_name}")

        import yaml

        with open(config_file, "r") as f:
            original_config = yaml.safe_load(f)

        modified_config = self._apply_overrides(original_config)

        args = Namespace(
            task=current_task,
            num_records=num_records,
            start_index=start_index,
            output_dir=output_dir,
            batch_size=kwargs.get("batch_size", 50),
            checkpoint_interval=kwargs.get("checkpoint_interval", 100),
            debug=kwargs.get("debug", False),
            resume=kwargs.get("resume", False),
            output_with_ts=kwargs.get("output_with_ts", False),
            run_name=kwargs.get("run_name"),
            oasst=kwargs.get("oasst", False),
            quality=kwargs.get("quality", False),
        )

        try:
            executor: BaseTaskExecutor
            if kwargs.get("quality_only", False):
                executor = JudgeQualityTaskExecutor(args, kwargs.get("quality_config"))
            else:
                executor = DefaultTaskExecutor(args)

            executor.config = modified_config

            result = executor.execute()
            logger.info(f"Successfully executed task with all features: {task_name}")
            return result
        except Exception as e:
            if "model_type" in str(e).lower() or "model" in str(e).lower():
                logger.error(f"Model configuration error in {config_file}")
            elif "current task name is not initialized" in str(e).lower():
                logger.error(f"Task context initialization failed for '{task_name}'")
            raise ExecutionError(f"Failed to execute task '{task_name}': {e}")

    def _execute_programmatic_workflow(
        self,
        num_records: Optional[int],
        start_index: int,
        output_dir: Optional[str],
        **kwargs,
    ) -> Any:
        """Execute programmatic workflow with full framework features."""
        try:
            task_name = self.name
            utils.current_task = self.name
            os.makedirs(task_name, exist_ok=True)
            self._temp_files.append(task_name)

            if not kwargs.get("enable_default_transforms", False):
                self.disable_default_transforms()

            config_file = f"{task_name}/graph_config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False)

            utils.current_task = self.name

            args = Namespace(
                task=self.name,
                num_records=num_records,
                start_index=start_index,
                output_dir=output_dir or task_name,
                batch_size=kwargs.get("batch_size", 50),
                checkpoint_interval=kwargs.get("checkpoint_interval", 100),
                debug=kwargs.get("debug", False),
                resume=kwargs.get(
                    "resume",
                    self._config.get("data_config", {}).get("resumable", False),
                ),
                output_with_ts=kwargs.get("output_with_ts", False),
                run_name=kwargs.get("run_name"),
                oasst=kwargs.get(
                    "oasst",
                    bool(self._config.get("output_config", {}).get("oasst_mapper")),
                ),
                quality=kwargs.get(
                    "quality",
                    bool(self._config.get("output_config", {}).get("data_quality")),
                ),
            )

            executor: BaseTaskExecutor
            if kwargs.get("quality_only", False):
                executor = JudgeQualityTaskExecutor(args, kwargs.get("quality_config"))
            else:
                executor = DefaultTaskExecutor(args)

            result = executor.execute()

            output_file = None
            if self._config.get("data_config", {}).get("sink", {}).get("file_path"):
                output_file = self._config["data_config"]["sink"]["file_path"]
            elif args.output_dir:
                import glob

                pattern = os.path.join(args.output_dir, "*output*.json*")
                output_files = glob.glob(pattern)
                if output_files:
                    output_file = output_files[0]

            if output_file and os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        if output_file.endswith(".jsonl"):
                            return [json.loads(line) for line in f if line.strip()]
                        else:
                            return json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read output file {output_file}: {e}")

            return result

        except Exception as e:
            raise ExecutionError(f"Programmatic workflow execution failed: {e}")
        finally:
            self._cleanup()

    def _apply_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply all overrides to the loaded configuration."""
        if not hasattr(self, "_overrides") or not self._overrides:
            return config

        import copy

        modified_config = copy.deepcopy(config)

        for path, value in self._overrides.items():
            self._set_nested_value(modified_config, path, value)

        logger.info(f"Applied {len(self._overrides)} configuration overrides")
        return modified_config

    def _set_nested_value(self, config: dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value using dot notation path."""
        keys = path.split(".")
        current = config

        # Navigate through all keys except the last one
        for i, key in enumerate(keys[:-1]):
            if key.isdigit():
                key_index = int(key)
                if not isinstance(current, list):
                    # Build the current path for better error reporting
                    current_path = ".".join(keys[: i + 1])
                    raise ValueError(f"Expected list at path {current_path}, got {type(current)}")
                # Extend list if needed
                while key_index >= len(current):
                    current.append({})
                current = current[key_index]
            else:
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], (dict, list)):
                    # If the key exists but is not a container, replace it
                    current[key] = {}
                current = current[key]

        # Set the final value
        final_key = keys[-1]
        if final_key.isdigit():
            final_key_index = int(final_key)
            if not isinstance(current, list):
                raise ValueError(f"Expected list for index {final_key} at path {path}")
            # Extend list if needed
            while final_key_index >= len(current):
                current.append(None)
            current[final_key_index] = value
        else:
            current[final_key] = value

    def _callable_to_string_path(self, func: Union[str, Callable]) -> str:
        """Convert callable to string path for YAML serialization."""
        if callable(func):
            if hasattr(func, "__module__") and hasattr(func, "__name__"):
                return f"{func.__module__}.{func.__name__}"
            elif hasattr(func, "__class__"):
                return f"{func.__class__.__module__}.{func.__class__.__name__}"
            else:
                class_name = func.__class__.__name__
                return f"__main__.{class_name}"
        return func

    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        ext = Path(file_path).suffix.lower()
        return ext[1:] if ext in [".json", ".jsonl", ".csv", ".parquet"] else "json"

    def _create_temp_file(self, data: list[dict[str, Any]]) -> str:
        """Create temporary file for data."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with temp_file as f:
            json.dump(data, f, indent=2)
        self._temp_files.append(temp_file.name)
        return temp_file.name

    def _generate_node_name(self, node_type: str) -> str:
        """Generate unique node name."""
        self._node_counter += 1
        return f"{node_type}_{self._node_counter}"

    def _add_node_with_edge(self, node_name: str, node_config: dict[str, Any]) -> None:
        """Add node and create edges."""
        self._config["graph_config"]["nodes"][node_name] = node_config

        # Create edges
        if self._last_node is None:
            edge_config = {"from": "START", "to": node_name}
        else:
            edge_config = {"from": self._last_node, "to": node_name}

        self._config["graph_config"]["edges"].append(edge_config)
        self._last_node = node_name

        # Add edge to END
        end_edge = {"from": node_name, "to": "END"}
        self._config["graph_config"]["edges"] = [
            e for e in self._config["graph_config"]["edges"] if e.get("to") != "END"
        ]
        self._config["graph_config"]["edges"].append(end_edge)

    def _cleanup(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
            except Exception as e:
                logger.warning(f"Could not clean up {temp_file}: {e}")
        self._temp_files.clear()

    def __del__(self):
        """Cleanup on destruction."""
        self._cleanup()


def create_graph(name: str) -> Workflow:
    """Factory function to create a new workflow builder (backward compatibility)."""
    return Workflow(name)


__all__ = ["Workflow", "create_graph"]
