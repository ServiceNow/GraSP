from typing import Any, Callable, Optional, Union

from ..models import ModelConfigBuilder


class BaseNodeBuilder:
    """Base class for all node builders."""

    def __init__(self, name: str, node_type: str):
        self.name: str = name
        self.node_type: str = node_type
        self._config: dict[str, Any] = {"node_type": node_type}
        self._messages: list[dict[str, Union[str, list[dict[str, Any]]]]] = []

    def build(self) -> dict[str, Any]:
        """Build the final node configuration."""
        if self._messages:
            self._config["prompt"] = self._messages
        return self._config.copy()

    def _callable_to_string_path(self, func: Callable) -> str:
        """Convert a callable to its string path."""
        if hasattr(func, "__module__") and hasattr(func, "__name__"):
            return f"{func.__module__}.{func.__name__}"

        elif hasattr(func, "__class__"):
            return f"{func.__class__.__module__}.{func.__class__.__name__}"

        elif hasattr(func, "__module__") and hasattr(func, "__qualname__"):
            return f"{func.__module__}.{func.__qualname__}"

        else:
            raise ValueError(f"Cannot determine string path for callable: {func}")


class LLMNodeBuilder(BaseNodeBuilder):
    """Builder for LLM nodes with full feature support."""

    def __init__(self, name: str, model: Union[str, dict[str, Any]]):
        super().__init__(name, "llm")

        if isinstance(model, str):
            self._config["model"] = ModelConfigBuilder.from_name(model)
        else:
            self._config["model"] = ModelConfigBuilder.validate_config(model)

    def prompt(self, prompt: Union[str, list[dict[str, str]]]) -> "LLMNodeBuilder":
        """Set prompt configuration directly."""
        if isinstance(prompt, str):
            self._config["prompt"] = [{"user": prompt}]
        else:
            self._config["prompt"] = prompt
        return self

    def system_message(self, message: str) -> "LLMNodeBuilder":
        """Add system message."""
        # Remove any existing system message
        self._messages = [msg for msg in self._messages if "system" not in msg]
        # Add system message at the beginning
        self._messages.insert(0, {"system": message})
        return self

    def user_message(self, message: str) -> "LLMNodeBuilder":
        """Add user message."""
        self._messages.append({"user": message})
        return self

    def assistant_message(self, message: str) -> "LLMNodeBuilder":
        """Add assistant message."""
        self._messages.append({"assistant": message})
        return self

    def add_message(self, role: str, content: str) -> "LLMNodeBuilder":
        """Add message with custom role."""
        self._messages.append({role: content})
        return self

    def temperature(self, temp: float) -> "LLMNodeBuilder":
        """Set temperature parameter for the model."""
        if "model" not in self._config:
            self._config["model"] = {}
        if "parameters" not in self._config["model"]:
            self._config["model"]["parameters"] = {}

        self._config["model"]["parameters"]["temperature"] = temp
        return self

    def max_tokens(self, tokens: int) -> "LLMNodeBuilder":
        """Set max_tokens parameter for the model."""
        if "model" not in self._config:
            self._config["model"] = {}
        if "parameters" not in self._config["model"]:
            self._config["model"]["parameters"] = {}

        self._config["model"]["parameters"]["max_tokens"] = tokens
        return self

    def model_parameter(self, param_name: str, value: Any) -> "LLMNodeBuilder":
        """Set arbitrary model parameter."""
        if "model" not in self._config:
            self._config["model"] = {}
        if "parameters" not in self._config["model"]:
            self._config["model"]["parameters"] = {}

        self._config["model"]["parameters"][param_name] = value
        return self

    def multimodal_message(self, role: str, content: list[dict[str, Any]]) -> "LLMNodeBuilder":
        """Add multimodal message with complex content."""
        self._messages.append({role: content})
        return self

    def output_keys(self, output: str) -> "LLMNodeBuilder":
        """Set output keys."""
        self._config["output_keys"] = output
        return self

    def chat_history(self, enabled: bool = True) -> "LLMNodeBuilder":
        """Enable/disable chat history."""
        self._config["chat_history"] = enabled
        return self

    def input_key(self, key: str) -> "LLMNodeBuilder":
        """Set input key."""
        self._config["input_key"] = key
        return self

    def output_role(self, role: str) -> "LLMNodeBuilder":
        """Set output role."""
        self._config["output_role"] = role
        return self

    def pre_process(self, processor: Union[str, Callable]) -> "LLMNodeBuilder":
        """Set pre-processor."""
        if callable(processor):
            self._config["pre_process"] = self._callable_to_string_path(processor)
        else:
            self._config["pre_process"] = processor
        return self

    def post_process(self, processor: Union[str, Callable]) -> "LLMNodeBuilder":
        """Set post-processor."""
        if callable(processor):
            self._config["post_process"] = self._callable_to_string_path(processor)
        else:
            self._config["post_process"] = processor
        return self

    def structured_output(self, schema: Union[str, dict[str, Any]], **kwargs) -> "LLMNodeBuilder":
        """Configure structured output."""
        structured_config = {
            "enabled": True,
            "schema": schema,
            "fallback_strategy": kwargs.get("fallback_strategy", "instruction"),
            "retry_on_parse_error": kwargs.get("retry_on_parse_error", True),
            "max_parse_retries": kwargs.get("max_parse_retries", 2),
        }
        self._config["structured_output"] = structured_config
        return self


class AgentNodeBuilder(BaseNodeBuilder):
    """Builder for Agent nodes with full feature support."""

    def __init__(self, name: str, model: Union[str, dict[str, Any]]):
        super().__init__(name, "agent")

        if isinstance(model, str):
            self._config["model"] = ModelConfigBuilder.from_name(model)
        else:
            self._config["model"] = ModelConfigBuilder.validate_config(model)

    def prompt(self, prompt: Union[str, list[dict[str, str]]]) -> "AgentNodeBuilder":
        """Set prompt configuration directly."""
        if isinstance(prompt, str):
            self._config["prompt"] = [{"user": prompt}]
        else:
            self._config["prompt"] = prompt
        return self

    def system_message(self, message: str) -> "AgentNodeBuilder":
        """Add system message."""
        # Remove any existing system message
        self._messages = [msg for msg in self._messages if "system" not in msg]
        # Add system message at the beginning
        self._messages.insert(0, {"system": message})
        return self

    def user_message(self, message: str) -> "AgentNodeBuilder":
        """Add user message."""
        self._messages.append({"user": message})
        return self

    def assistant_message(self, message: str) -> "AgentNodeBuilder":
        """Add assistant message."""
        self._messages.append({"assistant": message})
        return self

    def add_message(self, role: str, content: str) -> "AgentNodeBuilder":
        """Add message with custom role."""
        self._messages.append({role: content})
        return self

    def tools(self, tools: list[Any]) -> "AgentNodeBuilder":
        """Set agent tools."""
        self._config["tools"] = tools
        return self

    def inject_system_messages(self, messages: dict[int, str]) -> "AgentNodeBuilder":
        """Set system message injection configuration."""
        self._config["inject_system_messages"] = messages
        return self

    def chat_history(self, enabled: bool = True) -> "AgentNodeBuilder":
        """Enable/disable chat history."""
        self._config["chat_history"] = enabled
        return self

    def pre_process(self, processor: Union[str, Callable]) -> "AgentNodeBuilder":
        """Set pre-processor."""
        if callable(processor):
            self._config["pre_process"] = self._callable_to_string_path(processor)
        else:
            self._config["pre_process"] = processor
        return self

    def post_process(self, processor: Union[str, Callable]) -> "AgentNodeBuilder":
        """Set post-processor."""
        if callable(processor):
            self._config["post_process"] = self._callable_to_string_path(processor)
        else:
            self._config["post_process"] = processor
        return self


class MultiLLMNodeBuilder(BaseNodeBuilder):
    """Builder for Multi-LLM nodes."""

    def __init__(self, name: str):
        super().__init__(name, "multi_llm")
        self._config["models"] = {}

    def add_model(self, label: str, model: Union[str, dict[str, Any]]) -> "MultiLLMNodeBuilder":
        """Add a model to the multi-LLM configuration."""
        if isinstance(model, str):
            self._config["models"][label] = ModelConfigBuilder.from_name(model)
        else:
            self._config["models"][label] = ModelConfigBuilder.validate_config(model)
        return self

    def prompt(self, prompt: Union[str, list[dict[str, str]]]) -> "MultiLLMNodeBuilder":
        """Set prompt configuration directly."""
        if isinstance(prompt, str):
            self._config["prompt"] = [{"user": prompt}]
        else:
            self._config["prompt"] = prompt
        return self

    def system_message(self, message: str) -> "MultiLLMNodeBuilder":
        """Add system message."""
        # Remove any existing system message
        self._messages = [msg for msg in self._messages if "system" not in msg]
        # Add system message at the beginning
        self._messages.insert(0, {"system": message})
        return self

    def user_message(self, message: str) -> "MultiLLMNodeBuilder":
        """Add user message."""
        self._messages.append({"user": message})
        return self

    def assistant_message(self, message: str) -> "MultiLLMNodeBuilder":
        """Add assistant message."""
        self._messages.append({"assistant": message})
        return self

    def output_keys(self, output: str) -> "MultiLLMNodeBuilder":
        """Set output keys."""
        self._config["output_keys"] = output
        return self

    def multi_llm_post_process(self, processor: Union[str, Callable]) -> "MultiLLMNodeBuilder":
        """Set multi-LLM post-processor."""
        if callable(processor):
            self._config["multi_llm_post_process"] = self._callable_to_string_path(processor)
        else:
            self._config["multi_llm_post_process"] = processor
        return self


class LambdaNodeBuilder(BaseNodeBuilder):
    """Builder for Lambda nodes."""

    def __init__(self, name: str, func: Union[str, Callable]):
        super().__init__(name, "lambda")

        if callable(func):
            self._config["lambda"] = self._callable_to_string_path(func)
        else:
            self._config["lambda"] = func

    def output_keys(self, output: str) -> "LambdaNodeBuilder":
        """Set output keys."""
        self._config["output_keys"] = output
        return self


class WeightedSamplerNodeBuilder(BaseNodeBuilder):
    """Builder for Weighted Sampler nodes."""

    def __init__(self, name: str):
        super().__init__(name, "weighted_sampler")
        self._config["attributes"] = {}

    def add_attribute(
        self,
        name: str,
        values: Union[list[Any], dict[str, Any]],
        weights: Optional[list[float]] = None,
    ) -> "WeightedSamplerNodeBuilder":
        """Add an attribute to sample."""
        attr_config = {"values": values}
        if weights and isinstance(values, list):
            attr_config["weights"] = weights
        self._config["attributes"][name] = attr_config
        return self


class SubgraphNodeBuilder(BaseNodeBuilder):
    """Builder for Subgraph nodes."""

    def __init__(self, name: str, subgraph: str):
        super().__init__(name, "subgraph")
        self._config["subgraph"] = subgraph

    def node_config_map(self, config_map: dict[str, Any]) -> "SubgraphNodeBuilder":
        """Set node configuration overrides."""
        self._config["node_config_map"] = config_map
        return self


__all__ = [
    "LLMNodeBuilder",
    "AgentNodeBuilder",
    "MultiLLMNodeBuilder",
    "LambdaNodeBuilder",
    "WeightedSamplerNodeBuilder",
    "SubgraphNodeBuilder",
]
