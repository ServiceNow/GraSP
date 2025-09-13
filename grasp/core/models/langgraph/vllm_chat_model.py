from typing import Any, List

import openai
from langchain_core.messages import BaseMessage

from grasp.core.models.custom_models import ModelParams
from grasp.core.models.langgraph.grasp_base_chat_model import GraspBaseChatModel
from grasp.logger.logger_config import logger
from grasp.utils import constants


class CustomVLLMChatModel(GraspBaseChatModel):
    """
    A custom VLLM chat model implementation that interfaces with VLLM-served models through the OpenAI API format.

    This class extends LangChain's BaseChatModel to work with VLLM-deployed models, allowing them to be used
    seamlessly within the GraSP framework. It handles both synchronous and asynchronous model invocations,
    and supports tool/function calling compatible with LangChain's agent interfaces.

    Usage:
    ------
    1. Deploy a VLLM server with your model
    2. Configure the model in your graph_config.yaml:
        ```yaml
        nodes:
          your_agent_node:
            node_type: agent
            model:
              name: vllm_model_name # As registered in models.yaml
              parameters:
                temperature: 0.7                        # Optional: Model temperature
                max_tokens: 1024                        # Optional: Max output tokens
            # Other agent node configuration...
        ```

    3. The model can then be used with GraSP's AgentNode to do tool calls and Agentic data generation

    This implementation handles:
    - Message formatting and conversion between LangChain and OpenAI formats
    - Tool/function calling for agents
    - Both sync and async interfaces using _generate() and _agenerate() methods respectively
    - bind_tools() method to bind Agent specific tools to the model

    Follow the guide for more information on how to implement a custom Chat Model:
    [Guide](https://python.langchain.com/docs/how_to/custom_chat_model/).
    """

    def __init__(self, model_config: dict[str, Any]) -> None:
        """
        Initializes the CustomVLLMChatModel with configuration parameters.

        Parameters
        ----------
        model_config : dict[str, Any]
            A dictionary containing model configuration parameters.

        Returns
        -------
        None
        """
        super().__init__(model_config)
        self._model_serving_name = model_config.get("model_serving_name", self._get_name())

    @property
    def _llm_type(self) -> str:
        return "custom-vllm-chatmodel-wrapper"

    async def _generate_response(
        self,
        messages: List[BaseMessage],
        model_params: ModelParams,
        async_client: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """
        Generates a response to a given prompt asynchronously.

        Parameters
        ----------
        messages : list[AnyMessage]
            The input prompt to generate a response for.
        model_params : ModelParams
            The model parameters containing the URL and authentication token.
        async_client : bool
            Whether to use an async client to send the request. Defaults to True.

        Returns
        -------
        tuple[Any, int]
            A tuple containing the generated response string and a status code.
        """
        try:
            response_code = 200
            model_url = model_params.url
            model_auth_token = model_params.auth_token
            self._set_client(model_url, model_auth_token)
            # Build request and send it
            payload = self._client.build_request(messages=messages, **kwargs)
            response = await self._client.send_request(
                payload, self._model_serving_name, self._generation_params
            )
        except openai.RateLimitError as e:
            logger.warn(f"VLLM api request exceeded rate limit: {e}")
            response = f"{constants.ERROR_PREFIX} Http request failed {e}"
            response_code = 429
        except Exception as x:
            response = f"{constants.ERROR_PREFIX} Http request failed {x}"
            logger.error(response)
            rcode = self._get_status_from_body(x)
            response_code = rcode if rcode is not None else 999
        return response, response_code

    def _sync_generate_response(
        self,
        messages: list[BaseMessage],
        model_params: ModelParams,
        async_client: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """
        Generates a response to a given prompt synchronously.

        Parameters
        ----------
        messages : list[AnyMessage]
            The input prompt to generate a response for.
        model_params : ModelParams
            The model parameters containing the URL and authentication token.
        async_client : bool
            Whether to use an async client to send the request. Defaults to True.

        Returns
        -------
        tuple[Any, int]
            A tuple containing the generated response string and a status code.
        """
        try:
            response_code = 200
            model_url = model_params.url
            model_auth_token = model_params.auth_token
            self._set_client(model_url, model_auth_token, async_client=async_client)
            # Build request and send it
            payload = self._client.build_request(messages=messages, **kwargs)
            response = self._client.send_request(
                payload, self._model_serving_name, self._generation_params
            )
        except openai.RateLimitError as e:
            logger.warn(f"VLLM api request exceeded rate limit: {e}")
            response = f"{constants.ERROR_PREFIX} Http request failed {e}"
            response_code = 429
        except Exception as x:
            response = f"{constants.ERROR_PREFIX} Http request failed {x}"
            logger.error(response)
            rcode = self._get_status_from_body(x)
            response_code = rcode if rcode is not None else 999
        return response, response_code
