from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_openai.chat_models.base import _convert_message_to_dict
from ollama import AsyncClient, Client
from pydantic import BaseModel, ConfigDict, Field

from grasp.core.models.client.base_client import BaseClient
from grasp.logger.logger_config import logger
from grasp.utils import constants


class OllamaClientConfig(BaseModel):
    host: str = Field(default="http://localhost:11434", description="Base URL for the Ollama API")
    timeout: int = Field(
        default=constants.DEFAULT_TIMEOUT, description="Request timeout in seconds"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )


class OllamaClient(BaseClient):
    def __init__(
        self,
        async_client=False,
        chat_completions_api=True,
        stop: Optional[List[str]] = None,
        **client_kwargs,
    ):
        """
        Initialize an Ollama client.

        Args:
        - async_client (bool, optional): Whether to use an async client. Defaults to False.
        - chat_completions_api (bool, optional): Whether to use the chat completions API. Defaults to True.
        - stop (Optional[List[str]], optional): List of strings indicating when to stop generating text. Defaults to None.
        - **client_kwargs: Additional keyword arguments to pass to the Ollama API.
        """
        super().__init__(**client_kwargs)

        # Validate client_kwargs using Pydantic model
        validated_config = OllamaClientConfig(**client_kwargs)
        validated_client_kwargs = validated_config.model_dump()

        self.client: Any = (
            AsyncClient(**validated_client_kwargs)
            if async_client
            else Client(**validated_client_kwargs)
        )
        self.async_client = async_client
        self.chat_completions_api = chat_completions_api
        self.tools = None

    def build_request(
        self,
        messages: Optional[Sequence[BaseMessage]] = None,
        formatted_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Build a request payload for the Ollama model.

        Args:
            messages (List[BaseMessage]): The messages to pass to the model. This is necessary for chat completions API.
            formatted_prompt (str): The formatted prompt to pass to the model. This is necessary for completions API.
            stop (Optional[List[str]], optional): List of strings to stop generating at. Defaults to None.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The request payload.

        Raises:
            ValueError: If the messages or formatted prompt are invalid.
        """
        payload = {**kwargs}
        if "tools" in payload:
            self.tools = payload["tools"]
            del payload["tools"]

        if self.chat_completions_api:
            # Convert to Ollama message format
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
            if formatted_prompt is not None:
                payload["prompt"] = formatted_prompt
                return payload
            else:
                logger.error(
                    "Formatted prompt passed is None. Please provide a valid formatted prompt for completion API."
                )
                raise ValueError(
                    "Formatted prompt passed is None. Please provide a valid formatted prompt for completion API."
                )

    def send_request(
        self,
        payload,
        model_name: str,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Send a request to the Ollama API.

        Args:
            payload (dict): The payload to send to the model.
            model_name (str): The name of the model to send the request to.
            generation_params (Optional[Dict[str, Any]]): Additional parameters to pass to the model.

        Returns:
            Any: The response from the model.
        """
        # Normalize generation_params and extract optional 'format'
        generation_params = dict(generation_params or {})
        format = generation_params.pop("format", None)

        if self.chat_completions_api:
            return self.client.chat(
                model=model_name,
                messages=payload["messages"],
                options=generation_params,
                tools=self.tools,
                format=format,
            )
        else:
            return self.client.generate(
                model=model_name,
                prompt=payload["prompt"],
                options=generation_params,
                format=format,
            )
