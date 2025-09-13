from enum import Enum
from typing import Any, Dict, Optional

import httpx
from httpx import Timeout
from mistralai_azure import MistralAzure
from mistralai_azure.utils.retries import BackoffStrategy, RetryConfig

from grasp.core.models.client.http_client import HttpClient
from grasp.core.models.client.ollama_client import OllamaClient
from grasp.core.models.client.openai_azure_client import OpenAIAzureClient
from grasp.core.models.client.openai_client import OpenAIClient
from grasp.logger.logger_config import logger
from grasp.utils import constants, utils


class ModelType(Enum):
    """Enum representing the supported model types for client creation."""

    VLLM = "vllm"
    OPENAI = "azure_openai"
    AZURE = "azure"
    MISTRALAI = "mistralai"
    TGI = "tgi"
    OLLAMA = "ollama"
    TRITON = "triton"


# Define which model types do not require a AUTH_TOKEN
NO_AUTH_TOKEN_NEEDED_MODEL_TYPES = [ModelType.OLLAMA.value]


class ClientFactory:
    """
    Factory class for creating and initializing different client instances for API calls.
    This factory handles the creation of appropriate client types based on the model configuration,
    abstracting away the details of client initialization.
    """

    @classmethod
    def create_client(
        cls,
        model_config: Dict[str, Any],
        url: str,
        auth_token: Optional[str] = None,
        async_client: bool = False,
    ) -> Any:
        """
        Create and return an appropriate client instance based on the provided model configuration.

        Args:
            model_config: Dictionary containing model configuration parameters
            url: The base URL for the API
            auth_token: The authentication token for the API
            async_client: Whether to create an async client or a synchronous one

        Returns:
            A client instance for the specified model type

        Raises:
            ValueError: If required configuration keys are missing
            NotImplementedError: If the specified model type is not supported
        """
        # Validate model_type is present
        utils.validate_required_keys(["model_type"], model_config, "model")

        model_type = model_config["model_type"].lower()

        # Validate if url is present
        if url is None:
            logger.error("URL is required for client creation.")
            raise ValueError("URL is required for client creation.")

        # Validate if auth_token is present
        if auth_token is None and model_type not in NO_AUTH_TOKEN_NEEDED_MODEL_TYPES:
            logger.error("Auth token/API key is required for client creation.")
            raise ValueError("Auth token/API key is required for client creation.")

        # Validate that the model_type is supported
        supported_types = [type_enum.value for type_enum in ModelType]
        if model_type not in supported_types:
            supported_types_str = ", ".join(supported_types)
            logger.error(
                f"Unsupported model type: {model_type}. Supported types: {supported_types_str}"
            )
            raise ValueError(
                f"Unsupported model type: {model_type}. Must be one of: {supported_types_str}"
            )

        # Create client based on model type
        if model_type == ModelType.VLLM.value:
            # Initialize the client with default chat_completions_api
            return cls._create_openai_client(
                model_config,
                url,
                auth_token,
                async_client,
                not model_config.get("completions_api", False),
            )
        elif model_type == ModelType.OPENAI.value:
            return cls._create_openai_azure_client(
                model_config,
                url,
                auth_token,
                async_client,
                not model_config.get("completions_api", False),
            )
        elif (
            model_type == ModelType.AZURE.value
            or model_type == ModelType.TGI.value
            or model_type == ModelType.TRITON.value
        ):
            return cls._create_http_client(model_config, url, auth_token)
        elif model_type == ModelType.MISTRALAI.value:
            return cls._create_mistral_client(model_config, url, auth_token, async_client)
        elif model_type == ModelType.OLLAMA.value:
            return cls._create_ollama_client(
                model_config,
                url,
                None,
                async_client,
                not model_config.get("completions_api", False),
            )
        else:
            # This should never be reached due to the validation above, but included for completeness
            logger.error(f"No client implementation available for model type: {model_type}")
            raise NotImplementedError(f"Client for model type {model_type} is not implemented")

    @staticmethod
    def _create_openai_client(
        model_config: Dict[str, Any],
        url: str,
        auth_token: Optional[str] = None,
        async_client: bool = True,
        chat_completions_api: bool = True,
    ) -> OpenAIClient:
        """
        Create a client for VLLM models.

        Args:
            model_config: Dictionary containing model configuration parameters
            url: The base URL for the API
            auth_token: The authentication token for the API
            async_client: Whether to create an async client

        Returns:
            An OpenAI-compatible client instance configured for VLLM
        """
        model_config = utils.get_updated_model_config(model_config)
        utils.validate_required_keys(["url", "auth_token"], model_config, "model")
        ssl_verify: bool = bool(model_config.get("ssl_verify", True))
        ssl_cert = model_config.get("ssl_cert")
        httpx_client = (
            httpx.AsyncClient(http1=True, verify=ssl_verify, cert=ssl_cert)
            if async_client
            else httpx.Client(http1=True, verify=ssl_verify, cert=ssl_cert)
        )

        client_kwargs = {
            "base_url": url,
            "api_key": auth_token,
            "timeout": model_config.get("timeout", constants.DEFAULT_TIMEOUT),
            "max_retries": model_config.get("max_retries", 3),
            "http_client": httpx_client,
        }

        return OpenAIClient(async_client, chat_completions_api, **client_kwargs)

    @staticmethod
    def _create_openai_azure_client(
        model_config: Dict[str, Any],
        url: str,
        auth_token: Optional[str] = None,
        async_client: bool = True,
        chat_completions_api: bool = True,
    ) -> OpenAIAzureClient:
        """
        Create a client for OpenAI models.

        Args:
            model_config: Dictionary containing model configuration parameters
            url: The base URL for the API
            auth_token: The authentication token for the API
            async_client: Whether to create an async client

        Returns:
            An AzureOpenAI client instance
        """
        model_config = utils.get_updated_model_config(model_config)
        utils.validate_required_keys(
            ["url", "api_key", "api_version", "model"], model_config, "model"
        )
        ssl_verify: bool = bool(model_config.get("ssl_verify", True))
        ssl_cert = model_config.get("ssl_cert")
        httpx_client = (
            httpx.AsyncClient(http1=True, verify=ssl_verify, cert=ssl_cert)
            if async_client
            else httpx.Client(http1=True, verify=ssl_verify, cert=ssl_cert)
        )

        client_kwargs = {
            "azure_deployment": model_config.get("model"),
            "azure_endpoint": url,
            "api_version": model_config.get("api_version"),
            "default_headers": {"Connection": "close"},
            "api_key": auth_token,
            "timeout": model_config.get("timeout", constants.DEFAULT_TIMEOUT),
            "max_retries": model_config.get("max_retries", 3),
            "http_client": httpx_client,
        }

        return OpenAIAzureClient(async_client, chat_completions_api, **client_kwargs)

    @staticmethod
    def _create_mistral_client(
        model_config: Dict[str, Any],
        url: str,
        auth_token: Optional[str] = None,
        async_client: bool = True,
    ) -> MistralAzure:
        """
        Create a client for MistralAI models.

        Args:
            model_config: Dictionary containing model configuration parameters
            url: The base URL for the API
            auth_token: The authentication token for the API
            async_client: Whether to create an async client (note: MistralAI may not support async client)

        Returns:
            A MistralAzure client instance
        """
        model_config = utils.get_updated_model_config(model_config)
        utils.validate_required_keys(["url", "auth_token"], model_config, "model")
        ssl_verify: bool = bool(model_config.get("ssl_verify", True))
        ssl_cert = model_config.get("ssl_cert")
        httpx_client = (
            httpx.AsyncClient(
                http1=True,
                verify=ssl_verify,
                cert=ssl_cert,
                timeout=Timeout(timeout=model_config.get("timeout", constants.DEFAULT_TIMEOUT)),
            )
            if async_client
            else httpx.Client(
                http1=True,
                verify=ssl_verify,
                cert=ssl_cert,
                timeout=Timeout(timeout=model_config.get("timeout", constants.DEFAULT_TIMEOUT)),
            )
        )
        # Configure retry settings
        retry_config = RetryConfig(
            strategy="backoff",
            retry_connection_errors=True,
            backoff=BackoffStrategy(
                initial_interval=1000,
                max_interval=1000,
                exponent=1.5,
                max_elapsed_time=10,
            ),
        )

        client_kwargs: Dict[str, Any] = {
            "azure_api_key": auth_token,
            "azure_endpoint": url,
            "retry_config": retry_config,
        }
        (
            client_kwargs.update({"async_client": httpx_client})
            if async_client
            else client_kwargs.update({"client": httpx_client})
        )
        client = MistralAzure(**client_kwargs)
        return client

    @staticmethod
    def _create_http_client(
        model_config: Dict[str, Any], url: str, auth_token: Optional[str] = None
    ) -> HttpClient:
        """
        Create an HTTP client.

        Args:
            model_config: Dictionary containing model configuration parameters
            url: The base URL for the API
            auth_token: The authentication token for the API

        Returns:
            An HttpClient instance
        """
        model_config = utils.get_updated_model_config(model_config)
        utils.validate_required_keys(["url", "auth_token"], model_config, "model")
        ssl_verify = bool(model_config.get("ssl_verify", True))
        ssl_cert = model_config.get("ssl_cert")
        if auth_token is None:
            # Should be validated by create_client for these model types, but keep a safe check here
            raise ValueError("Auth token/API key is required for HTTP client creation.")
        auth_token = auth_token.replace("Bearer ", "")

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        timeout = model_config.get("timeout", constants.DEFAULT_TIMEOUT)
        max_retries = model_config.get("max_retries", 3)

        # Create and return HttpClient instance
        return HttpClient(
            base_url=url,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            ssl_verify=ssl_verify,
            ssl_cert=ssl_cert,
        )

    @staticmethod
    def _create_ollama_client(
        model_config: Dict[str, Any],
        url: str,
        auth_token: Optional[str] = None,
        async_client: bool = False,
        chat_completions_api: bool = True,
    ) -> OllamaClient:
        """
        Create a client for Ollama models.

        Args:
            model_config: Dictionary containing model configuration parameters
            url: The base URL for the API (default: http://localhost:11434)
            auth_token: Not used for Ollama, kept for API consistency
            async_client: Whether to create an async client
            chat_completions_api: Whether to use the chat completions API

        Returns:
            An OllamaClient instance
        """
        # Ollama doesn't need auth token, so we don't validate it

        client_kwargs = {
            "timeout": model_config.get("timeout", constants.DEFAULT_TIMEOUT),
        }
        if url is not None:
            client_kwargs.update({"host": url})

        stop = model_config.get("stop", None)

        return OllamaClient(async_client, chat_completions_api, stop, **client_kwargs)
