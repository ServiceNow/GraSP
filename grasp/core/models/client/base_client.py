from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, convert_to_messages
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue


class BaseClient(ABC):
    def __init__(self, **kwargs):
        self.client = None

    @staticmethod
    def _convert_input(model_input: LanguageModelInput) -> PromptValue:
        """
        Convert the input to a PromptValue.

        This method takes in a LanguageModelInput, which can be a PromptValue,
        a str, or a list of BaseMessages, and returns a PromptValue.

        If the input is already a PromptValue, it is simply returned.
        If the input is a str, it is converted to a StringPromptValue.
        If the input is a list of BaseMessages, it is converted to a ChatPromptValue.
        Otherwise, a ValueError is raised.

        Args:
            model_input (LanguageModelInput): The input to convert.

        Returns:
            PromptValue: The converted input.

        Raises:
            ValueError: If the input is not a PromptValue, str, or list of BaseMessages.
        """
        if isinstance(model_input, PromptValue):
            return model_input
        if isinstance(model_input, str):
            return StringPromptValue(text=model_input)
        if isinstance(model_input, Sequence):
            return ChatPromptValue(messages=convert_to_messages(model_input))
        msg = (
            f"Invalid input type {type(model_input)}. "
            "Must be a PromptValue, str, or list of BaseMessages."
        )
        raise ValueError(msg)

    @abstractmethod
    def build_request(
        self,
        messages: Optional[Sequence[BaseMessage]] = None,
        formatted_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Build a request payload for the model.

        Args:
            messages (Sequence[BaseMessage]): The input messages to the model.
            formatted_prompt (str): The formatted prompt to pass to the model.
            stop (Optional[List[str]], optional): List of strings to stop generating at. Defaults to None.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The request payload.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Build request must be implemented")

    def build_request_with_payload(self, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Convenience method to construct a request directly from a payload dict.

        This is an optional helper. Subclasses may override to customize behavior,
        but are not required to implement it.

        Args:
            payload: Base payload dictionary.
            **kwargs: Additional fields to merge into the payload.

        Returns:
            The merged payload dictionary.
        """
        payload.update(kwargs)
        return payload

    @abstractmethod
    def send_request(
        self,
        payload: Any,
        model_name: str,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Send the request payload to the model and return the response.

        Args:
            payload (dict): The request payload to send.
            model_name (str): The name of the model to send the request to.
            generation_params: Optional[Dict[str, Any]]: Additional keyword arguments to include in the request.

        Returns:
            dict: The response from the model.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Send request must be implemented")

    async def async_send_request(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Optional async helper to send a request. By default, this delegates to the
        synchronous send_request in a thread, so subclasses don't have to implement
        an async variant unless they want custom behavior.

        Returns:
            The response from the model.
        """
        pass
