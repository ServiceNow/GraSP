import asyncio
import collections
import json
import os
import random
import sys
import time
from abc import abstractmethod
from typing import Any, Callable, DefaultDict, Dict, List, Literal, Optional, Sequence, Tuple, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, run_in_executor
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
    _create_usage_metadata,
)
from openai.types import Completion
from openai.types.chat import ChatCompletion
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import AutoTokenizer

from grasp.core.models.client.base_client import BaseClient
from grasp.core.models.client.client_factory import ClientFactory
from grasp.core.models.custom_models import ModelParams
from grasp.logger.logger_config import logger
from grasp.utils import constants, utils


class GraspBaseChatModel(BaseChatModel):
    def __init__(self, model_config: dict[str, Any]) -> None:
        """
        Initialize a GraspBaseChatModel with configuration parameters.

        Parameters
        ----------
        model_config : dict[str, Any]
            A dictionary containing model configuration parameters.

        Returns
        -------
        None
        """
        super().__init__()
        utils.validate_required_keys(["name", "parameters"], model_config, "model")
        self._config = model_config
        # sleep before every call - in ms
        self._delay = model_config.get("delay", 100)
        # max_wait for 8 attempts = 2^(8-1) = 128 secs
        self._retry_attempts = model_config.get("retry_attempts", 8)
        self._generation_params = model_config.get("parameters")
        self._hf_chat_template_model_id = model_config.get("hf_chat_template_model_id")
        if self._hf_chat_template_model_id:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._hf_chat_template_model_id,
                token=os.environ.get(constants.HF_TOKEN),
            )
            if model_config.get("modify_tokenizer", False):
                self._set_chat_template()
        self._model_stats: Dict[str, Any] = {"resp_code_dist": {}, "errors": {}}
        # track total count for round_robin load balancing; see "_get_model_url"
        self._call_count = 0
        # track the number of requests per url for least_requests load balancing; see "_get_model_url"
        self._url_reqs_count: DefaultDict[str, int] = collections.defaultdict(int)
        # store the timestamps to check if server is down
        self._model_failed_response_timestamp: list[float] = []
        if self._get_name() in constants.COMPLETION_ONLY_MODELS:
            self._config["completions_api"] = True
        self._validate_completions_api_support()
        self._client: BaseClient

    def _validate_completions_api_support(self) -> None:
        """
        Validates that if completions_api is set to True, raises an error that chat model does not support completion API.

        Raises
        ------
        ValueError
            Chat model does not support completion API.
        """
        if self._config.get("completions_api", False):
            raise ValueError(
                f"Chat Model {self._get_name()} does not support completion API. "
                f"Please set completions_api to False or remove completions_api from {self._get_name()} config in models.yaml"
            )

    def _get_name(self) -> str:
        """
        Return the name of the model, as specified in the model configuration.

        Returns
        -------
        str
            The name of the model.
        """
        return str(self._config["name"])

    def _set_chat_template(self):
        """
        Set the chat template for the tokenizer from the environment variable.
        Raises
        -------
        EnvironmentError
            If the environment variable for the chat template is not set and override_tokenizer is True.
        """
        env_name = utils.get_env_name(self._config["name"])
        env_var = f"GRASP_{env_name}_CHAT_TEMPLATE"

        template = os.environ.get(env_var)
        if template:
            self._tokenizer.chat_template = template
        else:
            raise EnvironmentError(
                f"Environment variable {env_var} not set, but override_tokenizer is True."
            )

    def _get_model_params(self) -> ModelParams:
        """
        Get the model parameters.

        Returns the model parameters based on the `url` and `auth_token` configuration.
        If `url` is a string, it returns it directly. If `url` is a list, it implements
        load balancing based on the `load_balancing` configuration. If it is "round_robin",
        it returns the model parameters for the current index in the list. If it is "least_requests",
        it returns the model parameters for the URL with the least number of requests so far.

        Args:
            None

        Returns:
            ModelParams: The model parameters.
        """
        url = self._config["url"]
        if "auth_token" in self._config:
            auth_token = self._config["auth_token"]
        else:
            auth_token = self._config["api_key"]

        return_url = None
        return_auth_token = None
        if isinstance(url, str):
            return_url = url
            return_auth_token = auth_token
        elif isinstance(url, list):
            load_balancing = self._config.get("load_balancing", "least_requests")
            if load_balancing == "round_robin":
                idx = self._call_count % len(url)
                return_url = url[idx]
                return_auth_token = auth_token[idx] if isinstance(auth_token, list) else auth_token
            elif load_balancing == "least_requests":
                # initialize the count for each url if it is not already done
                if not self._url_reqs_count:
                    self._url_reqs_count = collections.defaultdict(int, {u: 0 for u in url})
                # find the url with least requests
                min_value = min(self._url_reqs_count.values())
                min_keys = [k for k, v in self._url_reqs_count.items() if v == min_value]
                # get random url if all have same number of requests
                return_url = random.choice(min_keys)
                return_auth_token = (
                    auth_token[url.index(return_url)]
                    if isinstance(auth_token, list)
                    else auth_token
                )
                self._url_reqs_count[return_url] += 1
            else:
                raise ValueError(
                    f"Invalid load balancing type: {load_balancing}. Supported types are round_robin and least_requests"
                )
        else:
            raise ValueError("Model URL should be a string or a list of strings")

        self._call_count += 1
        return ModelParams(return_url, return_auth_token)

    def _update_model_stats(
        self, response: Union[Completion, ChatCompletion], resp_status: int
    ) -> None:
        """
        Update the model statistics with the given response.

        Args:
            response: The response from the model.
            resp_status: The status code of the response.

        Returns:
            None
        """
        resp_text = ""
        content = response.choices[0].model_dump()["message"]["content"]
        if content:
            resp_text = content.strip()
        code_count = self._model_stats["resp_code_dist"].get(resp_status, 0)
        self._model_stats["resp_code_dist"][resp_status] = code_count + 1
        if resp_status != 200:
            # TODO: Right now the error messages are based on vllm; need to generalize for all models
            resp_text = resp_text.lower()
            if "timed out" in resp_text:
                error_type = "timeout"
            elif "maximum context length is" in resp_text:
                error_type = "tokens_exceeded"
            elif "connection error" in resp_text:
                error_type = "connection_error"
            else:
                error_type = "other"
            error_count = self._model_stats["errors"].get(error_type, 0)
            self._model_stats["errors"][error_type] = error_count + 1

        # log model stats after every model_stats_interval
        total_requests = sum(self._model_stats["resp_code_dist"].values())
        model_stats_interval = self._config.get("stats_interval", 10000)
        if total_requests % model_stats_interval == 0:
            # convert stats to percentage
            temp_model_stats = {"total_requests": total_requests}
            for key_for_percent in ["resp_code_dist", "errors"]:
                temp_model_stats[key_for_percent] = {
                    k: f"{(v / total_requests): 0.3f}"
                    for k, v in self._model_stats[key_for_percent].items()
                }

            logger.info(f"[{self._get_name()}] Model Stats: {temp_model_stats}")

    def _handle_server_down(self, resp_status: int):
        """
        Handle server down situation.

        If the server is down, we check if we receive server down status(404, 500-503)
        if the failure count is 10(MAX_FAILED_ERROR) within 30(MODEL_FAILURE_WINDOW_IN_SEC) seconds
        shutdown the process, we need to fix the model first

        Args:
            resp_status (int): The status code of the response.

        Returns:
            None
        """
        if not constants.HANDLE_SERVER_DOWN:
            # no need to handle this, user has disabled the feature
            return
        if resp_status in constants.SERVER_DOWN_ERROR_CODE:
            # append the current timestamp for this error
            self._model_failed_response_timestamp.append(time.time())
            # if storage is full, pop the first in
            if len(self._model_failed_response_timestamp) > constants.MAX_FAILED_ERROR:
                self._model_failed_response_timestamp.pop(0)

            total_in_queue = len(self._model_failed_response_timestamp)
            # if total count is more than maximum error to handle, than only do the validation
            if total_in_queue >= constants.MAX_FAILED_ERROR:
                # when MAX_FAILED_ERROR = 10
                # if 100 in total, check time diff of 91st(old) and 100th(new)
                # if 10 in total, check time diff of 1st(old) and 10th(new)
                oldest_timestamp = self._model_failed_response_timestamp[
                    total_in_queue - constants.MAX_FAILED_ERROR
                ]
                newest_timestamp = self._model_failed_response_timestamp[total_in_queue - 1]
                time_gap_in_sec = newest_timestamp - oldest_timestamp
                logger.warning(
                    f"Server failure count: {constants.MAX_FAILED_ERROR} in {time_gap_in_sec} seconds."
                )
                # last n(MAX_FAILED_ERROR) failures within t(MODEL_FAILURE_WINDOW_IN_SEC) seconds
                if time_gap_in_sec < constants.MODEL_FAILURE_WINDOW_IN_SEC:
                    logger.error(
                        f"SYSTEM EXITING as the dependant model({self._get_name()}) is down for longer period."
                    )
                    sys.exit()

    def _is_retryable_error(self, result: tuple[Any, int]) -> bool:
        """
        Check if the error is a rate limit error by checking response code.

        Currently retrying for too many requests error(429)
        and APIConnectionError(599) returned by OpenAI intermittently
        and 444 = Blocked by azure content filter

        Args:
            result (tuple[Any, int]): The result from the API call, which contains the response body and status code.

        Returns:
            bool: True if the error is retryable, False otherwise.
        """
        return len(result) == 2 and result[1] in constants.RETRYABLE_HTTP_ERROR

    def _log_before_retry(self, retry_state):
        """
        Log a message before retrying a failed request.

        This function is a callback that is called by the retrying library
        before each retry attempt. It logs a message warning the user that
        the request is being retried and the reason for the retry.

        Args:
            retry_state: The retry state object, which contains the outcome
                of the previous attempt and the next action to take.
        """
        resp_code = retry_state.outcome.result()[1]
        logger.warning(
            f"[{self._get_name()}] Retrying the request in {retry_state.next_action.sleep} seconds as it returned"
            f" {resp_code} code"
        )
        resp_code = retry_state.outcome.result()[1]
        logger.warning(
            f"[{self._get_name()}] Retrying the request in {retry_state.next_action.sleep} seconds as it returned"
            f" {resp_code} code"
        )

    def _get_status_from_body(self, response: Any) -> Optional[int]:
        """
        Extract http error status code from body.

        This function takes a string response body and
        returns the http error status code if it can be extracted.
        If the extraction fails, it returns None.

        The function first checks if the response body is a dict
        and if so, it tries to extract the statusCode from it.
        If not, it tries to parse the response body as a json string
        and then extract the statusCode from it. If the statusCode is
        not present in the body, it looks for a code field and returns
        that instead.

        Args:
            response (str): The response body as a string.

        Returns:
            int: The http error status code if it can be extracted, None otherwise.
        """
        try:
            # Attempt to normalize to a dict body
            body: Optional[dict[str, Any]] = None
            # Some SDK exceptions have a `.body` attribute (object with dict or JSON string)
            if hasattr(response, "body"):
                resp_body = getattr(response, "body")
                if isinstance(resp_body, dict):
                    body = resp_body
                elif isinstance(resp_body, str):
                    body = json.loads(resp_body)
            # If not found via attribute, the response itself might be a dict or JSON string
            if body is None:
                if isinstance(response, dict):
                    body = response
                elif isinstance(response, str):
                    # Try load as JSON string
                    body = json.loads(response)
                else:
                    return None

            status_code = body.get("statusCode")
            # for openai api it is in code
            if status_code is None:
                code = body.get("code")
                if code is not None:
                    return int(code)
                return None
            else:
                return int(status_code)
        except Exception:
            return None

    def _get_chat_formatted_text(self, messages: list[AnyMessage]) -> str:
        """
        Converts a list of messages to a string formatted according to the
        chat template. The string is formatted by applying the chat template
        to the list of messages after converting each message to a dictionary.
        The `tokenize` parameter is set to `False` and `add_generation_prompt` is
        set to `True` when calling `apply_chat_template`.

        Args:
            messages: A list of messages to be formatted.

        Returns:
            formatted_text: A string formatted according to the chat template.
        """
        formatted_text = str(
            self._tokenizer.apply_chat_template(
                [_convert_message_to_dict(message) for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        logger.debug(f"Chat formatted text: {formatted_text}")
        return formatted_text

    async def _generate_response_with_retry(
        self,
        messages: List[BaseMessage],
        model_params: ModelParams,
        async_client: bool = True,
        **kwargs: Any,
    ):
        """
        Retry text generation with model with exponential backoff and random jitter.
        Total number of retry attempts and delay between each attempt can be configured
        via "retry_attempts" and "delay" properties in eval/config/models.json
        :param messages: List of messages sent to the chat model
        :param model_params: Model parameters containing url and auth_token
        :param async_client: Whether to use an async client to send requests
        :param kwargs: Additional keyword arguments for the _generate_text method
        :return: The response text and status code
        """
        result = None
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_result(self._is_retryable_error),
                wait=wait_random_exponential(multiplier=1),
                stop=stop_after_attempt(self._retry_attempts),
                before_sleep=self._log_before_retry,
            ):  # Configure retry logic
                with attempt:
                    # Initial delay
                    await asyncio.sleep(self._delay / 1000)
                    response, response_code = await self._generate_response(
                        messages, model_params, async_client, **kwargs
                    )
                    # Process response with post-processing
                    result = self._invoke_post_process(response), response_code
                # outcome may be None; guard before accessing attributes
                outcome = attempt.retry_state.outcome
                if outcome is not None and not outcome.failed:
                    attempt.retry_state.set_result(result)

        except RetryError:
            logger.error(f"{self._get_name()} Request failed after {self._retry_attempts} attempts")

        return result

    def _sync_generate_response_with_retry(
        self,
        messages: List[BaseMessage],
        model_params: ModelParams,
        async_client: bool = True,
        **kwargs: Any,
    ):
        """
        Retry text generation with model with exponential backoff and random jitter.
        Total number of retry attempts and delay between each attempt can be configured
        via "retry_attempts" and "delay" properties in eval/config/models.json
        :param messages: List of messages sent to the chat model
        :param model_params: Model parameters containing url and auth_token
        :param async_client: Whether to use an async client to send requests
        :param kwargs: Additional keyword arguments for the _generate_text method
        :return: The response text and status code
        """
        result = None
        try:
            for attempt in Retrying(
                retry=retry_if_result(self._is_retryable_error),
                wait=wait_random_exponential(multiplier=1),
                stop=stop_after_attempt(self._retry_attempts),
                before_sleep=self._log_before_retry,
            ):  # Configure retry logic
                with attempt:
                    # Initial delay
                    time.sleep(self._delay / 1000)
                    response, response_code = self._sync_generate_response(
                        messages, model_params, async_client, **kwargs
                    )
                    # Process response with post-processing
                    result = self._invoke_post_process(response), response_code
                outcome = attempt.retry_state.outcome
                if outcome is not None and not outcome.failed:
                    attempt.retry_state.set_result(result)

        except RetryError:
            logger.error(f"{self._get_name()} Request failed after {self._retry_attempts} attempts")

        return result

    @abstractmethod
    async def _generate_response(
        self,
        messages: List[BaseMessage],
        model_params: ModelParams,
        async_client: bool = True,
        **kwargs: Any,
    ) -> Tuple[Any, int]:
        """
        Abstract method to generate text with the model.

        Args:
            messages (List[BaseMessage]): List of messages to send to the model
            model_params (ModelParams): Model parameters containing url and auth_token
            async_client (bool, optional): Whether to use an async client to send requests. Defaults to True.
            **kwargs (Any): Additional keyword arguments for the generation method

        Returns:
            Tuple[Any, int]: The response text and status code
        """
        raise NotImplementedError("_generate_response() must be implemented")

    @abstractmethod
    def _sync_generate_response(
        self,
        messages: List[BaseMessage],
        model_params: ModelParams,
        async_client: bool = True,
        **kwargs: Any,
    ) -> Tuple[Any, int]:
        """
        Abstract method to generate text with the model.

        Args:
            messages (List[BaseMessage]): List of messages to send to the model
            model_params (ModelParams): Model parameters containing url and auth_token
            async_client (bool, optional): Whether to use an async client to send requests. Defaults to True.
            **kwargs (Any): Additional keyword arguments for the generation method

        Returns:
            Tuple[Any, int]: The response text and status code
        """
        pass

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of Grasp chat model."""

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Asynchronous implementation of the `_generate` method.

        Args:
            messages (List[BaseMessage]): List of messages to send to the model
            stop (Optional[List[str]], optional): List of strings to stop generating at. Defaults to None.
            run_manager (Optional, optional): Run manager to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments for the generation method

        Returns:
            ChatResult: The generated chat result
        """
        generation_info = None
        model_params = self._get_model_params()
        model_url = model_params.url
        logger.debug(
            f"[{self._get_name()}][{model_url}] REQUEST: {[_convert_message_to_dict(m) for m in messages]}"
        )

        response, response_code = await self._generate_response_with_retry(
            messages, model_params, **kwargs
        )
        self._update_model_stats(response, response_code)
        self._handle_server_down(response_code)
        # reduce the count of requests for the url to handle least_requests load balancing
        self._url_reqs_count[model_url] -= 1
        return await run_in_executor(None, self._create_chat_result, response, generation_info)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Synchronous implementation of the `_generate` method.

        Args:
            messages (List[BaseMessage]): List of messages to send to the model
            stop (Optional[List[str]], optional): List of strings to stop generating at. Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun], optional): Run manager to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments for the generation method

        Returns:
            ChatResult: The response text and status code
        """
        generation_info = None
        model_params = self._get_model_params()
        model_url = model_params.url
        logger.debug(
            f"[{self._get_name()}][{model_url}] REQUEST: {[_convert_message_to_dict(m) for m in messages]}"
        )

        response, response_code = self._sync_generate_response_with_retry(
            messages=messages, model_params=model_params, async_client=False, **kwargs
        )
        self._update_model_stats(response, response_code)
        self._handle_server_down(response_code)
        # reduce the count of requests for the url to handle least_requests load balancing
        self._url_reqs_count[model_url] -= 1
        return self._create_chat_result(response, generation_info)

    def _invoke_post_process(self, response: ChatCompletion) -> ChatCompletion:
        post_proc = self._get_post_processor()
        # if post_process is defined at models.yaml, process the output text
        if post_proc is None:
            return response
        response.choices[0].message.content = post_proc().apply(response.choices[0].message.content)
        return response

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict[str, str], Literal["any", "auto"], str]] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [convert_to_openai_tool(tool, strict=True) for tool in tools]
        if not tool_choice:
            pass
        elif isinstance(tool_choice, dict):
            kwargs["tool_choice"] = tool_choice
        elif isinstance(tool_choice, str) and tool_choice in ("any", "auto"):
            kwargs["tool_choice"] = {"type": tool_choice}
        elif isinstance(tool_choice, str):
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        else:
            raise ValueError(
                f"Unrecognized 'tool_choice' type {tool_choice=}. Expected dict, " f"str, or None."
            )

        if parallel_tool_calls is not None:
            disable_parallel_tool_use = not parallel_tool_calls
            if "tool_choice" in kwargs:
                kwargs["tool_choice"]["disable_parallel_tool_use"] = disable_parallel_tool_use
            else:
                kwargs["tool_choice"] = {
                    "type": "auto",
                    "disable_parallel_tool_use": disable_parallel_tool_use,
                }

        return self.bind(tools=formatted_tools, **kwargs)

    def _set_client(self, url: str, auth_token: str, async_client: bool = True):
        """
        Get or create the client instance on demand.

        Args:
            url (str): The URL of the model.
            auth_token (str): The authentication token for the model.
            async_client (bool, optional): Whether to create an async client. Defaults to True.

        Returns:
            None
        """
        self._client = ClientFactory.create_client(self._config, url, auth_token, async_client)

    def _create_chat_result(
        self,
        response: ChatCompletion,
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        """
        Create a ChatResult object from the model response.

        Args:
            response ChatCompletion: The response from the model.
            generation_info (Optional[dict], optional): Any additional information
                about the generation. Defaults to None.

        Returns:
            ChatResult: The ChatResult object.
        """
        generations = []

        response_dict = response if isinstance(response, dict) else response.model_dump()
        # Sometimes the AI Model calling will get error, we should raise it.
        # Otherwise, the next code 'choices.extend(response["choices"])'
        # will throw a "TypeError: 'NoneType' object is not iterable" error
        # to mask the true error. Because 'response["choices"]' is None.
        if response_dict.get("error"):
            raise ValueError(response_dict.get("error"))

        token_usage = response_dict.get("usage")
        for res in response_dict["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)
            generation_info = generation_info or {}
            generation_info["finish_reason"] = (
                res.get("finish_reason")
                if res.get("finish_reason") is not None
                else generation_info.get("finish_reason")
            )
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": response_dict.get("model", response.model),
            "system_fingerprint": response_dict.get("system_fingerprint", ""),
        }
        if "id" in response_dict:
            llm_output["id"] = response_dict["id"]
        if "service_tier" in response_dict:
            llm_output["service_tier"] = response_dict["service_tier"]

        if isinstance(response, ChatCompletion) and getattr(response, "choices", None):
            chat_completion_message = response.choices[0].message
            if hasattr(chat_completion_message, "parsed"):
                generations[0].message.additional_kwargs["parsed"] = chat_completion_message.parsed
            if hasattr(chat_completion_message, "refusal"):
                generations[0].message.additional_kwargs[
                    "refusal"
                ] = chat_completion_message.refusal

        return ChatResult(generations=generations, llm_output=llm_output)

    # get post processor if available, returns none if not defined
    def _get_post_processor(self):
        """
        Get post-processor function from config if available

        Returns:
            Callable[[str], str]: Post processor function if available, None otherwise
        """
        post_proc = self._config.get("post_process")
        return utils.get_func_from_str(post_proc) if post_proc else None

    def ping(self) -> int:
        """
        Ping the model with a hello message and return http code
        if returns 200, its success
        """
        user_message = HumanMessage(content="hello")
        status = 200
        try:
            _ = self._generate([user_message])
        except Exception as e:
            logger.error(f"Error: {e}")
            status = 501
            return status
        return status
