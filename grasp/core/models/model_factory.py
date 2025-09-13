from typing import Any, Dict, Type

from grasp.core.models.custom_models import (
    CustomAzure,
    CustomMistralAPI,
    CustomOllama,
    CustomOpenAI,
    CustomTGI,
    CustomTriton,
    CustomVLLM,
)
from grasp.core.models.langgraph.openai_chat_model import CustomOpenAIChatModel
from grasp.core.models.langgraph.vllm_chat_model import CustomVLLMChatModel
from grasp.logger.logger_config import logger
from grasp.utils import utils


class ModelFactory:
    """
    Factory class for creating and initializing custom model instances.
    This factory handles the creation of appropriate model types based on configuration,
    with special handling for agent nodes that require models extended from BaseChatModel.
    """

    # Mapping of model types to their respective implementation classes
    MODEL_TYPE_MAP: Dict[str, Dict[str, Type[Any]]] = {
        "default": {
            "vllm": CustomVLLM,
            "mistralai": CustomMistralAPI,
            "tgi": CustomTGI,
            "azure": CustomAzure,
            "azure_openai": CustomOpenAI,
            "ollama": CustomOllama,
            "triton": CustomTriton,
        },
        "langgraph": {
            "vllm": CustomVLLMChatModel,
            "azure_openai": CustomOpenAIChatModel,
        },
    }

    @classmethod
    def create_model(cls, model_config: Dict[str, Any], backend: str = "default") -> Any:
        """
        Create and return an appropriate model instance based on the provided configuration.

        Args:
            model_config: Dictionary containing model configuration parameters
            backend: The backend to use for model creation

        Returns:
            An instance of a custom model class

        Raises:
            ValueError: If required configuration keys are missing
            NotImplementedError: If the specified model type is not supported
        """
        # Validate required keys
        utils.validate_required_keys(["name"], model_config, "model")

        # Update model config with global settings
        model_config = cls._update_model_config(model_config)

        # Validate model type is present after update
        utils.validate_required_keys(["model_type"], model_config, "model")

        model_type = model_config["model_type"]

        try:
            return cls.MODEL_TYPE_MAP[backend][model_type](model_config)
        except KeyError:
            logger.error(
                f"No specialized model implementation for {model_type} found for backend {backend}."
            )
            # If we get here, the model type is not supported
            raise NotImplementedError(f"Model type {model_type} for {backend} is not implemented")

    @staticmethod
    def _update_model_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update model configuration with global settings from the model config file.

        Args:
            model_config: Dictionary containing model configuration parameters

        Returns:
            Updated model configuration dictionary
        """
        global_model_configs = utils.load_model_config()
        global_model_config: dict[str, Any] = global_model_configs.get(model_config["name"], {})

        for param, value in model_config.items():
            if not isinstance(value, dict):
                global_model_config[param] = value
            else:
                # If it's a dictionary, update keys which are passed, do not remove other keys
                if param not in global_model_config:
                    global_model_config[param] = {}
                global_model_config[param].update(value)

        return global_model_config

    @classmethod
    def get_model(cls, model_config: Dict[str, Any], backend: str = "default") -> Any:
        """
        Get a model instance wrapped in a Runnable for use in LLM nodes.
        This method returns a Langgraph RunnableLambda instance.

        Args:
            model_config: Dictionary containing model configuration parameters
            backend: The backend to use for model creation

        Returns:
            A Runnable-wrapped model instance
        """
        from langchain_core.runnables import RunnableLambda

        model = cls.create_model(model_config, backend)

        # Wrap the model in a RunnableLambda for compatibility with LangChain
        return RunnableLambda(lambda x: x, afunc=model)
