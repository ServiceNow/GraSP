from typing import Any, Dict, List, Optional, Sequence, Union, cast

import httpx
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models.base import _convert_message_to_dict
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ConfigDict, Field

from grasp.core.models.client.base_client import BaseClient
from grasp.logger.logger_config import logger
from grasp.utils import constants


class OpenAIClientConfig(BaseModel):
    base_url: str = Field(..., description="Base URL for the OpenAI API")
    api_key: str = Field(..., description="API key for authentication")
    http_client: Union[httpx.AsyncClient, httpx.Client] = Field(
        ..., description="HTTP client to use"
    )
    # default_headers: Dict[str, str] = Field(default={"Connection": "close"},
    #                                         description="Default headers for API requests")
    timeout: int = Field(
        default=constants.DEFAULT_TIMEOUT, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )


class OpenAIClient(BaseClient):
    def __init__(
        self,
        async_client=True,
        chat_completions_api=True,
        stop: Optional[List[str]] = None,
        **client_kwargs,
    ):
        """
        Initialize a OpenAI client.

        Args:
        - async_client (bool, optional): Whether to use an async client. Defaults to False.
        - chat_completions_api (bool, optional): Whether to use the chat completions API. Defaults to True.
        - stop (Optional[List[str]], optional): List of strings indicating when to stop generating text. Defaults to None.
        - **client_kwargs: Additional keyword arguments to pass to the OpenAI or AsyncOpenAI constructor.
        """
        super().__init__(**client_kwargs)

        # Validate client_kwargs using Pydantic model
        validated_config = OpenAIClientConfig(**client_kwargs)
        validated_client_kwargs = validated_config.model_dump()

        self.client: Any = (
            AsyncOpenAI(**validated_client_kwargs)
            if async_client
            else OpenAI(**validated_client_kwargs)
        )
        self.async_client = async_client
        self.chat_completions_api = chat_completions_api
        self.stop = stop

    def build_request(
        self,
        messages: Optional[Sequence[BaseMessage]] = None,
        formatted_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Build a request payload for the model.

        If the model is using the chat completions API, the messages will be converted to a list of dictionaries
        and added to the payload under the key "messages". If the model is using the completions API, the formatted
        prompt will be added to the payload under the key "prompt". If the messages or formatted prompt are invalid,
        a ValueError will be raised.

        Args:
            messages (List[BaseMessage]): The messages to pass to the model. This is necessary for chat completions API.
            formatted_prompt (str): The formatted prompt to pass to the model. This is necessary for completions API.
            stop (Optional[List[str]], optional): List of stop sequences that indicate when text generation should halt. If None, the client-level default set during initialization will be used.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The request payload.

        Raises:
            ValueError: If the messages or formatted prompt are invalid.
        """
        # Prefer explicit stop passed to this call; otherwise use client default
        effective_stop = stop if stop is not None else self.stop
        if effective_stop is not None:
            kwargs["stop"] = effective_stop
        payload = {**kwargs}
        if self.chat_completions_api:
            if messages is not None and len(messages) > 0:
                messages = self._convert_input(messages).to_messages()
                payload["messages"] = [_convert_message_to_dict(m) for m in messages]
                return payload
            else:
                logger.error(
                    "messages passed is None or empty. Please provide valid messages to build request with chat completions API."
                )
                raise ValueError(
                    "messages passed is None or empty. Please provide valid messages to build request with chat completions API."
                )
        else:
            if formatted_prompt is not None and len(formatted_prompt) > 0:
                payload["prompt"] = formatted_prompt
                return payload
            else:
                logger.error(
                    "formatted_prompt passed is None. Please provide a valid formatted prompt to build request with completion API."
                )
                raise ValueError(
                    "formatted_prompt passed is None. Please provide a valid formatted prompt to build request with completion API."
                )

    def send_request(
        self,
        payload: Any,
        model_name: str,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Send a request to the Vllm hosted model.

        This method takes in a payload dictionary, a model name, and generation parameters.
        It sends a request to the Vllm hosted model with the given payload and generation parameters.
        If the chat completions API is being used, it will call the `chat.completions.create` method.
        Otherwise, it will call the `completions.create` method.

        Args:
            payload (dict): The payload to send to the model.
            model_name (str): The name of the model to send the request to.
            generation_params (Optional[Dict[str, Any]], optional): Additional generation parameters to pass to the model. Defaults to None.

        Returns:
            Any: The response from the model.
        """
        generation_params = generation_params or {}

        # Check for vLLM-specific parameters
        additional_extensions = {
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
        }

        additional_params = {
            k: v for k, v in generation_params.items() if k in additional_extensions
        }
        standard_params = {
            k: v for k, v in generation_params.items() if k not in additional_extensions
        }

        client = cast(Any, self.client)
        if not additional_params:
            if self.chat_completions_api:
                return client.chat.completions.create(
                    **payload, model=model_name, **generation_params
                )
            else:
                return client.completions.create(**payload, model=model_name, **generation_params)
        else:
            logger.info(f"Detected vLLM-specific parameters: {additional_params}")
            # Use extra_body to pass vLLM-specific parameters
            if self.chat_completions_api:
                return client.chat.completions.create(
                    **payload,
                    model=model_name,
                    extra_body=additional_params,
                    **standard_params,
                )
            else:
                return client.completions.create(
                    **payload,
                    model=model_name,
                    extra_body=additional_params,
                    **standard_params,
                )
