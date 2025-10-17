from __future__ import annotations

import asyncio
import base64
import collections
import json
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    DefaultDict,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    cast,
)

import openai
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import AutoTokenizer

import sygra.utils.constants as constants
from sygra.core.models.client.base_client import BaseClient
from sygra.core.models.client.client_factory import ClientFactory
from sygra.core.models.client.http_client import HttpClient
from sygra.core.models.client.openai_client import OpenAIClient
from sygra.core.models.structured_output.structured_output_config import (
    StructuredOutputConfig,
)
from sygra.logger.logger_config import logger
from sygra.utils import utils


class ModelParams:
    def __init__(self, url: str, auth_token: str):
        self.url = url
        self.auth_token = auth_token


class BaseCustomModel(ABC):
    def __init__(self, model_config: dict[str, Any]) -> None:
        utils.validate_required_keys(["name", "parameters"], model_config, "model")
        self.model_config = model_config
        self._structured_output_lock: Optional[asyncio.Lock] = None

        # Initialize structured output configuration
        structured_output_raw = model_config.get("structured_output")
        if structured_output_raw is None:
            self.structured_output_config: Dict[str, Any] = {}
            key_present = False
        else:
            self.structured_output_config = structured_output_raw or {}
            key_present = True

        self.structured_output = StructuredOutputConfig(self.structured_output_config, key_present)

        # sleep before every call - in ms
        self.delay = model_config.get("delay", 100)
        # max_wait for 8 attempts = 2^(8-1) = 128 secs
        self.retry_attempts = model_config.get("retry_attempts", 8)
        self.generation_params: dict[Any, Any] = model_config.get("parameters") or {}
        self.chat_template_params: dict[Any, Any] = model_config.get("chat_template_params") or {}
        self.hf_chat_template_model_id = model_config.get("hf_chat_template_model_id")
        if self.hf_chat_template_model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_chat_template_model_id, token=os.environ.get(constants.HF_TOKEN)
            )
            if model_config.get("modify_tokenizer", False):
                self._set_chat_template()
        self.model_stats: Dict[str, Any] = {"resp_code_dist": {}, "errors": {}}
        # track total count for round_robin load balancing; see "_get_model_url"
        self.call_count = 0
        # track the number of requests per url for least_requests load balancing; see "_get_model_url"
        self.url_reqs_count: DefaultDict[str, int] = collections.defaultdict(int)
        # store the timestamps to check if server is down
        self.model_failed_response_timestamp: list[float] = []
        self._validate_completions_api_support()
        self._client: BaseClient

    def _set_client(self, url: str, auth_token: Optional[str] = None, async_client: bool = True):
        """Get or create the client instance on demand."""
        self._client = ClientFactory.create_client(self.model_config, url, auth_token, async_client)

    def _validate_completions_api_support(self) -> None:
        """
        Validates that if completions_api is set to True, raises an error that model does not support completion API.

        Raises
        ------
        ValueError
            Model does not support completion API.
        """
        if self.model_config.get("completions_api", False):
            raise ValueError(
                f"Model {self.name()} does not support completion API. "
                f"Please set completions_api to False or remove completions_api from {self.name()} config in models.yaml"
            )

    def _set_chat_template(self):
        """
        Set the chat template for the tokenizer from the environment variable.
        Raises
        -------
        EnvironmentError
            If the environment variable for the chat template is not set and override_tokenizer is True.
        """
        env_name = utils.get_env_name(self.name())
        env_var = f"SYGRA_{env_name}_CHAT_TEMPLATE"

        template = os.environ.get(env_var)
        if template:
            self.tokenizer.chat_template = template
        else:
            raise EnvironmentError(
                f"Environment variable {env_var} not set, but override_tokenizer is True."
            )

    async def __call__(self, input: ChatPromptValue, **kwargs: Any) -> Any:
        # model_url = self._get_model_url()
        model_params = self._get_model_params()
        model_url = model_params.url

        # Handle structured output
        use_structured_output = (
            self.structured_output_config is not None and self.structured_output.enabled
        )

        logger.debug(
            f"[{self.name()}][{model_url}] REQUEST: {utils.convert_messages_from_langchain_to_chat_format(input.messages)}"
        )
        resp_text, resp_status = await self._call_with_retry(
            input, model_params, use_structured_output, **kwargs
        )

        # Apply common finalization logic
        return self._finalize_response(resp_text, resp_status, model_url)

    def _finalize_response(self, resp_text: str, resp_status: int, model_url: str) -> AIMessage:
        """Common response finalization logic"""
        self._update_model_stats(resp_text, resp_status)
        self._handle_server_down(resp_status)
        # reduce the count of requests for the url to handle least_requests load balancing
        self.url_reqs_count[model_url] -= 1
        logger.debug(f"[{self.name()}][{model_url}] RESPONSE: {resp_text}")
        resp_text = self._replace_special_tokens(resp_text)
        resp_text = self._post_process_for_model(resp_text)
        return AIMessage(resp_text)

    async def _get_lock(self) -> asyncio.Lock:
        if self._structured_output_lock is None:
            self._structured_output_lock = asyncio.Lock()
        return self._structured_output_lock

    async def _handle_structured_output(
        self, input: ChatPromptValue, model_params: ModelParams, **kwargs: Any
    ) -> Optional[Tuple[str, int]]:
        """Handle structured output generation"""
        lock = await self._get_lock()
        # Re-entry prevention using asyncio locking
        async with lock:
            pydantic_model = self.structured_output.get_pydantic_model()
            if not pydantic_model:
                logger.warning(
                    "Structured output enabled but no valid schema found, falling back to regular generation"
                )
                # Return a flag to signal that regular generation should be used
                return None

            # Check if model supports native structured output
            if self._supports_native_structured_output():
                logger.info(f"Using native structured output for {self.name()}")
                return await self._generate_native_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )
            else:
                logger.info(f"Using fallback structured output for {self.name()}")
                # Get response from fallback method
                (
                    resp_text,
                    resp_status,
                ) = await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )
                return resp_text, resp_status

    def _supports_native_structured_output(self) -> bool:
        """Check if the model supports native structured output"""
        # OpenAI and vLLM support native structured output
        return isinstance(self, (CustomOpenAI, CustomVLLM, CustomTGI, CustomOllama))

    async def _generate_native_structured_output(
        self,
        input: ChatPromptValue,
        model_params: ModelParams,
        pydantic_model: Type[BaseModel],
        **kwargs: Any,
    ) -> Tuple[str, int]:
        """Generate structured output using native model support"""
        # This will be implemented in specific model classes
        raise NotImplementedError("Native structured output not implemented for this model")

    async def _generate_fallback_structured_output(
        self,
        input: ChatPromptValue,
        model_params: ModelParams,
        pydantic_model: Type[BaseModel],
        **kwargs: Any,
    ) -> Tuple[str, int]:
        """Generate structured output using instruction-based fallback"""
        logger.info("Generating fallback structured output")
        parser = PydanticOutputParser(pydantic_object=pydantic_model)
        format_instructions = parser.get_format_instructions()

        # Modify the prompt to include format instructions
        modified_messages = list(input.messages)
        if modified_messages and modified_messages[-1].content:
            modified_messages[-1].content = (
                str(modified_messages[-1].content) + f"\n\n{format_instructions}"
            )

        modified_input = ChatPromptValue(messages=modified_messages)

        # Generate the text with retry (uses our centralized retry logic)
        resp_text, resp_status = await self._generate_response_with_retry(
            modified_input, model_params, **kwargs
        )

        if resp_status != 200:
            logger.error(
                f"[{self.name()}] Failed to generate fallback structured output: Status {resp_status}"
            )
            return resp_text, resp_status

        # Try to parse the response to validate it's proper JSON
        try:
            parsed_output = parser.parse(resp_text)
            logger.info(f"[{self.name()}] Structured output parsed successfully")
            # Return the validated JSON string
            return parsed_output.model_dump_json(), 200
        except Exception as e:
            logger.warning(f"[{self.name()}] Failed to parse structured output: {e}")
            logger.error(f"[{self.name()}] Returning unparsed response")
            # Return the original response text with status code 200
            return resp_text, 200

    def name(self) -> str:
        return cast(str, self.model_config["name"])

    def _get_model_params(self) -> ModelParams:
        url = self.model_config.get("url", "")
        auth_token = self.model_config.get("auth_token", "")

        return_url = None
        return_auth_token = None
        if isinstance(url, str):
            return_url = url
            return_auth_token = auth_token
        elif isinstance(url, list):
            load_balancing = self.model_config.get("load_balancing", "least_requests")
            if load_balancing == "round_robin":
                idx = self.call_count % len(url)
                return_url = url[idx]
                return_auth_token = auth_token[idx] if isinstance(auth_token, list) else auth_token
            elif load_balancing == "least_requests":
                # initialize the count for each url if it is not already done
                if not self.url_reqs_count:
                    self.url_reqs_count = collections.defaultdict(int, {u: 0 for u in url})
                # find the url with least requests
                min_value = min(self.url_reqs_count.values())
                min_keys = [k for k, v in self.url_reqs_count.items() if v == min_value]
                # get random url if all have same number of requests
                return_url = random.choice(min_keys)
                return_auth_token = (
                    auth_token[url.index(return_url)]
                    if isinstance(auth_token, list)
                    else auth_token
                )
                self.url_reqs_count[return_url] += 1
            else:
                raise ValueError(
                    f"Invalid load balancing type: {load_balancing}. Supported types are round_robin and least_requests"
                )
        else:
            raise ValueError("Model URL should be a string or a list of strings")

        self.call_count += 1
        return ModelParams(return_url, return_auth_token)

    # return the configured url if it is a string or
    # return the url based on the call count if it is a list to distribute the load
    def _get_model_url(self) -> str:
        url = self.model_config["url"]
        return_url = None
        if isinstance(url, str):
            return_url = url
        elif isinstance(url, list):
            load_balancing = self.model_config.get("load_balancing", "least_requests")
            if load_balancing == "round_robin":
                return_url = url[self.call_count % len(url)]
            elif load_balancing == "least_requests":
                # initialize the count for each url if it is not already done
                if not self.url_reqs_count:
                    self.url_reqs_count = collections.defaultdict(int, {u: 0 for u in url})
                # find the url with the least requests
                min_value = min(self.url_reqs_count.values())
                min_keys = [k for k, v in self.url_reqs_count.items() if v == min_value]
                # get random url if all have same number of requests
                return_url = random.choice(min_keys)
                self.url_reqs_count[return_url] += 1
            else:
                raise ValueError(
                    f"Invalid load balancing type: {load_balancing}. Supported types are round_robin and least_requests"
                )
        else:
            raise ValueError("Model URL should be a string or a list of strings")

        self.call_count += 1
        return return_url

    def _update_model_stats(self, resp_text: str, resp_status: int) -> None:
        code_count = self.model_stats["resp_code_dist"].get(resp_status, 0)
        self.model_stats["resp_code_dist"][resp_status] = code_count + 1
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
            error_count = self.model_stats["errors"].get(error_type, 0)
            self.model_stats["errors"][error_type] = error_count + 1

        # log model stats after every model_stats_interval
        total_requests = sum(self.model_stats["resp_code_dist"].values())
        model_stats_interval = self.model_config.get("stats_interval", 10000)
        if total_requests % model_stats_interval == 0:
            # convert stats to percentage
            temp_model_stats = {"total_requests": total_requests}
            for key_for_percent in ["resp_code_dist", "errors"]:
                temp_model_stats[key_for_percent] = {
                    k: f"{(v / total_requests): 0.3f}"
                    for k, v in self.model_stats[key_for_percent].items()
                }

            logger.info(f"[{self.name()}] Model Stats: {temp_model_stats}")

    @abstractmethod
    async def _generate_response(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        pass

    def _ping_model(self, url, auth_token) -> int:
        """
        Ping a single model
        Args:
            url: single url to ping
            auth_token: auth token for the url
        """
        # hello message
        msg = utils.backend_factory.get_test_message()
        # build parameters
        model_param = ModelParams(url=url, auth_token=auth_token)
        _, status = asyncio.run(self._generate_response(msg, model_param))
        return status

    def ping(self) -> int:
        """
        Ping the model with a hello message and return http code
        if returns 200, its success
        """
        url_obj = self.model_config.get("url")
        auth_token = self.model_config.get("auth_token")
        if isinstance(url_obj, list):
            for i, url in enumerate(url_obj):
                token = auth_token[i] if isinstance(auth_token, list) else auth_token
                status = self._ping_model(url=url, auth_token=token)
                if status != 200:
                    logger.error(f"Server({url}) responded with {status}")
                    return status
            return 200
        else:
            return self._ping_model(url=self.model_config.get("url"), auth_token=auth_token)

    def get_chat_formatted_text(
        self, chat_format_object: Sequence[BaseMessage], **chat_template_params
    ) -> str:
        chat_formatted_text = str(
            self.tokenizer.apply_chat_template(
                utils.convert_messages_from_langchain_to_chat_format(chat_format_object),
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_params,
            )
        )
        logger.debug(f"Chat formatted text: {chat_formatted_text}")
        return chat_formatted_text

    def _replace_special_tokens(self, text: str) -> str:
        for token in self.model_config.get("special_tokens", []):
            text = text.replace(token, "")
        return text.strip()

    def _post_process_for_model(self, text: str) -> str:
        if self.name() == "mixtral8x7b":
            # handle 8x7b generation of underscore with backslash
            text = text.replace("\\_", "_")
        elif self.name() == "mixtral_instruct_8x22b":
            # very specific pattern observed in 8x22b. The value within the details tag is what we need
            pattern1 = re.compile("<details><summary>.*?</summary>(.*?)</details>", re.DOTALL)
            res = re.findall(pattern1, text)
            if len(res) == 1:
                text = res[0]

            # to handle cases where mixtral adds additional tags around the text. e.g. [ANS]....[/ANS].
            # we are currently handling only cases where we observe one occurrence of the tag to reduce false positives
            pattern2 = re.compile("^\[([A-Z]+)\](.*?)\[/\\1\]", re.DOTALL)
            res = re.findall(pattern2, text)
            if len(res) == 1:
                text = res[0][1]
            # 8x22b adds intermittently adds begin{align*} and end{align*} tags. Probably coming from latex training
            # data.
            text = text.replace("begin{align*}", "").replace("end{align*}", "").strip()
        return text

    def _is_retryable_error(self, result):
        """check if the error is a rate limit error by checking response code"""
        # currently retrying for too many requests error(429)
        # and APIConnectionError(599) returned by OpenAI intermittently
        # and 444 = Blocked by azure content filter
        return len(result) == 2 and result[1] in constants.RETRYABLE_HTTP_ERROR

    def _log_before_retry(self, retry_state):
        """log retry attempt"""
        resp_code = retry_state.outcome.result()[1]
        logger.warning(
            f"[{self.name()}] Retrying the request in {retry_state.next_action.sleep} seconds as it returned"
            f" {resp_code} code"
        )

    async def _call_with_retry(
        self,
        input: ChatPromptValue,
        model_params: ModelParams,
        use_structured_output: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, int]:
        """
        Centralized retry method that delegates to either regular text generation
        or structured output handling based on the flag.
        """
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_result(self._is_retryable_error),
                wait=wait_random_exponential(multiplier=1),
                stop=stop_after_attempt(self.retry_attempts),
                before_sleep=self._log_before_retry,
            ):
                with attempt:
                    # initial delay for each call (in ms)
                    await asyncio.sleep(self.delay / 1000)

                    # Call the appropriate method based on the flag
                    if use_structured_output:
                        # Call the structured output handling
                        so_result = await self._handle_structured_output(
                            input, model_params, **kwargs
                        )

                        # If _handle_structured_output returns None, it means we should fall back to regular generation
                        if so_result is None:
                            logger.info(
                                "Structured output not configured, falling back to regular generation"
                            )
                            result = await self._generate_response(input, model_params, **kwargs)
                        else:
                            result = so_result
                    else:
                        # Regular text generation
                        result = await self._generate_response(input, model_params, **kwargs)

                    # Apply post-processing if defined
                    post_proc = self._get_post_processor()
                    if post_proc is not None:
                        resp_str = post_proc().apply(result[0])
                        resp_code = result[1]
                        # IMPORTANT: add more items if needed in future
                        result = (resp_str, resp_code)
                if (
                    attempt.retry_state.outcome is not None
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError:
            logger.error(f"[{self.name()}] Request failed after {self.retry_attempts} attempts")
        return result

    async def _generate_response_with_retry(
        self, input: ChatPromptValue, model_params: ModelParams, **kwargs: Any
    ) -> Tuple[str, int]:
        """
        Backward compatibility method that uses the centralized _call_with_retry.
        Retry text generation with model with exponential backoff and random jitter.
        Total number of retry attempts and delay between each attempt can be configured
        via "retry_attempts" and "delay" properties in eval/config/models.json
        """
        return await self._call_with_retry(
            input, model_params, use_structured_output=False, **kwargs
        )

    # get post processor if available, returns none if not defined
    def _get_post_processor(self):
        post_proc = self.model_config.get("post_process")
        return utils.get_func_from_str(post_proc) if post_proc else None

    def _handle_server_down(self, resp_status: int):
        """
        When the server is down, we check if we receive server down status(404, 500-503)
        if the failure count is 10(MAX_FAILED_ERROR) within 30(MODEL_FAILURE_WINDOW_IN_SEC) seconds
        shutdown the process, we need to fix the model first
        """
        if not constants.HANDLE_SERVER_DOWN:
            # no need to handle this, user has disabled the feature
            return
        if resp_status in constants.SERVER_DOWN_ERROR_CODE:
            # append the current timestamp for this error
            self.model_failed_response_timestamp.append(time.time())
            # if storage is full, pop the first in
            if len(self.model_failed_response_timestamp) > constants.MAX_FAILED_ERROR:
                self.model_failed_response_timestamp.pop(0)

            total_in_queue = len(self.model_failed_response_timestamp)
            # if total count is more than maximum error to handle, than only do the validation
            if total_in_queue >= constants.MAX_FAILED_ERROR:
                # when MAX_FAILED_ERROR = 10
                # if 100 in total, check time diff of 91st(old) and 100th(new)
                # if 10 in total, check time diff of 1st(old) and 10th(new)
                oldest_timestamp = self.model_failed_response_timestamp[
                    total_in_queue - constants.MAX_FAILED_ERROR
                ]
                newest_timestamp = self.model_failed_response_timestamp[total_in_queue - 1]
                time_gap_in_sec = newest_timestamp - oldest_timestamp
                logger.warning(
                    f"Server failure count: {constants.MAX_FAILED_ERROR} in {time_gap_in_sec} seconds."
                )
                # last n(MAX_FAILED_ERROR) failures within t(MODEL_FAILURE_WINDOW_IN_SEC) seconds
                if time_gap_in_sec < constants.MODEL_FAILURE_WINDOW_IN_SEC:
                    logger.error(
                        f"SYSTEM EXITING as the dependant model({self.name()}) is down for longer period."
                    )
                    sys.exit()

    def _get_status_from_body(self, response: Any) -> Optional[int]:
        """
        Extract http error status code from body
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


class CustomTGI(BaseCustomModel):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        utils.validate_required_keys(["url", "auth_token"], model_config, "model")
        self.model_config = model_config
        self.auth_token = cast(str, model_config.get("auth_token")).replace("Bearer ", "")

    async def _generate_native_structured_output(
        self,
        input: ChatPromptValue,
        model_params: ModelParams,
        pydantic_model: Type[BaseModel],
        **kwargs: Any,
    ) -> Tuple[str, int]:
        """Generate structured output using TGI's native support"""
        logger.info(f"[{self.name()}] Attempting native structured output generation")
        model_url = model_params.url
        try:
            # Set Client
            self._set_client(model_params.url, model_params.auth_token)
            client = cast(HttpClient, self._client)

            # Get JSON schema from the Pydantic model
            json_schema = pydantic_model.model_json_schema()

            # Build Request
            payload = {
                "inputs": self.get_chat_formatted_text(
                    input.messages, **(self.chat_template_params or {})
                )
            }

            # Prepare generation parameters with guidance
            generation_params_with_guidance = {
                **(self.generation_params or {}),
                "parameters": {"grammar": {"type": "json", "value": json_schema}},
            }

            payload = client.build_request_with_payload(payload=payload)

            # Send Request with guidance parameters
            resp = await client.async_send_request(
                payload, generation_params=generation_params_with_guidance
            )

            resp_text = resp.text
            resp_status = resp.status_code
            logger.info(f"[{self.name()}][{model_url}] RESPONSE: Native support call successful")
            logger.debug(f"[{self.name()}] Native structured output response: {resp_text}")

            if resp_status != 200:
                logger.error(
                    f"[{self.name()}] Native structured output HTTP request failed with code: {resp_status}"
                )
                # Fall back to instruction-based approach
                logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
                return await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )

            # Parse the response text - TGI returns the guided JSON directly
            resp_text = json.loads(resp_text)["generated_text"]

            # Validate the response against the schema
            try:
                resp_text = json.loads(resp_text) if isinstance(resp_text, str) else resp_text
                # Attempt to parse with the pydantic model to validate
                logger.debug(f"[{self.name()}] Validating response against schema")
                pydantic_model.model_validate(resp_text)
                # If validation succeeds, return the validated JSON
                logger.info(f"[{self.name()}] Native structured output generation succeeded")
                return json.dumps(resp_text), resp_status
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"[{self.name()}] Native structured output validation failed: {e}")
                logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
                # Fall back to instruction-based approach
                return await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )

        except Exception as e:
            logger.error(f"[{self.name()}] Native structured output generation failed: {e}")
            logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
            # Fall back to instruction-based approach
            return await self._generate_fallback_structured_output(
                input, model_params, pydantic_model, **kwargs
            )

    async def _generate_response(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        try:
            # Set Client
            self._set_client(model_params.url, model_params.auth_token)
            client = cast(HttpClient, self._client)
            # Build Request
            payload = {
                "inputs": self.get_chat_formatted_text(
                    input.messages, **(self.chat_template_params or {})
                )
            }
            payload = client.build_request_with_payload(payload=payload)
            # Send Request
            resp = await client.async_send_request(
                payload, generation_params=self.generation_params
            )

            resp_text = resp.text
            if resp.status_code != 200:
                logger.error(
                    f"HTTP request failed with code: {resp.status_code} and error: {resp_text}"
                )
                resp_text = f"{constants.ERROR_PREFIX} {resp_text}"
                if (
                    constants.ELEMAI_JOB_DOWN in resp_text
                    or constants.CONNECTION_ERROR in resp_text
                ):
                    # server down
                    resp.status = 503
            else:
                resp_text = json.loads(resp_text)["generated_text"]
        except Exception as x:
            resp_text = f"{constants.ERROR_PREFIX} Http request failed {x}"
            logger.error(resp_text)
            rcode = self._get_status_from_body(x)
            ret_code = rcode if rcode else 999
            return resp_text, ret_code
        return resp_text, resp.status_code


class CustomAzure(BaseCustomModel):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        utils.validate_required_keys(["url", "auth_token"], model_config, "model")
        self.model_config = model_config
        auth_token_value = model_config["auth_token"]  # already validated key exists

        if isinstance(auth_token_value, str):
            self.auth_token = auth_token_value.replace("Bearer ", "")
        elif isinstance(auth_token_value, list) and auth_token_value:
            # take first element if non-empty list
            first_item = auth_token_value[0]
            if not isinstance(first_item, str):
                raise TypeError("auth_token list must contain strings")
            self.auth_token = first_item.replace("Bearer ", "")
        else:
            raise ValueError("auth_token must be a string or non-empty list of strings")

    async def _generate_response(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        model_url = model_params.url
        try:
            # Set Client
            self._set_client(model_url, model_params.auth_token)
            # Build Request
            payload = {
                "messages": utils.convert_messages_from_langchain_to_chat_format(input.messages)
            }
            payload = self._client.build_request_with_payload(payload=payload)
            # Send Request
            resp = await self._client.async_send_request(
                payload, generation_params=self.generation_params
            )

            logger.debug(f"[{self.name()}]\n[{model_url}] \n REQUEST DATA: {payload}")

            resp_text = resp.text
            if resp.status_code != 200:
                logger.error(
                    f"[{self.name()}] HTTP request failed with code: {resp.status_code} and error: {resp_text}"
                )
                resp_text = ""
            else:
                result = json.loads(resp_text)
                if result["choices"][0]["finish_reason"] == "content_filter":
                    return "Blocked by azure content filter", 444
                resp_text = result["choices"][0]["message"]["content"]
        except Exception as x:
            resp_text = f"Http request failed {x}"
            logger.error(resp_text)
            rcode = self._get_status_from_body(x)
            ret_code = rcode if rcode else 999
            return resp_text, ret_code
        return resp_text, resp.status_code


class CustomMistralAPI(BaseCustomModel):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)

    async def _generate_response(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        ret_code = 200
        model_url = model_params.url
        try:
            chat_format_messages = utils.convert_messages_from_langchain_to_chat_format(
                input.messages
            )
            messages = [{"role": m["role"], "content": m["content"]} for m in chat_format_messages]
            self._set_client(model_url, model_params.auth_token)
            chat_response = await self._client.chat.complete_async(  # type: ignore
                model=self.model_config.get("model"),
                messages=messages,
                **self.generation_params,
            )
            resp_text = chat_response.choices[0].message.content
        except Exception as x:
            resp_text = f"{constants.ERROR_PREFIX} Http request failed {x}"
            logger.error(resp_text)
            lower_resp_text = resp_text.lower()
            rcode = self._get_status_from_body(x)
            if (
                constants.MIXTRAL_API_RATE_LIMIT_ERROR.lower() in lower_resp_text
                or constants.MIXTRAL_API_MODEL_OVERLOAD_ERROR.lower() in lower_resp_text
            ):
                ret_code = 429
            elif rcode is not None:
                ret_code = rcode
            else:
                # for other cases, return 999, dont retry
                ret_code = 999
        return resp_text, ret_code


class CustomVLLM(BaseCustomModel):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        utils.validate_required_keys(["url", "auth_token"], model_config, "model")
        self.model_config = model_config
        self.auth_token = str(model_config.get("auth_token")).replace("Bearer ", "")
        self.model_serving_name = model_config.get("model_serving_name", self.name())

    def _validate_completions_api_support(self) -> None:
        if self.model_config.get("completions_api", False):
            logger.info(f"Model {self.name()} supports completion API.")

    async def _generate_native_structured_output(
        self,
        input: ChatPromptValue,
        model_params: ModelParams,
        pydantic_model: Type[BaseModel],
        **kwargs: Any,
    ) -> Tuple[str, int]:
        """Generate structured output using vLLM's guided generation"""
        logger.info(f"[{self.name()}] Attempting native structured output generation")
        model_url = model_params.url
        try:
            self._set_client(model_url, model_params.auth_token)
            client = cast(OpenAIClient, self._client)

            # Create JSON schema for guided generation
            json_schema = pydantic_model.model_json_schema()

            # Prepare payload using the client
            if self.model_config.get("completions_api", False):
                formatted_prompt = self.get_chat_formatted_text(
                    input.messages, **(self.chat_template_params or {})
                )
                payload = client.build_request(formatted_prompt=formatted_prompt)
            else:
                payload = client.build_request(messages=input.messages)

            # Use vLLM's native guided generation
            extra_params = {**(self.generation_params or {}), "guided_json": json_schema}

            # Send the request using the client
            completion = await client.send_request(payload, self.model_serving_name, extra_params)

            # Check if the request was successful based on the response status
            resp_status = getattr(completion, "status_code", 200)  # Default to 200 if not present

            if resp_status != 200:
                logger.error(
                    f"[{self.name()}] Native structured output request failed with code: {resp_status}"
                )
                # Fall back to instruction-based approach
                logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
                return await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )

            # Extract response text based on API type
            if self.model_config.get("completions_api", False):
                resp_text = completion.choices[0].model_dump()["text"].strip()
            else:
                resp_text = completion.choices[0].model_dump()["message"]["content"].strip()

            logger.info(f"[{self.name()}][{model_url}] RESPONSE: Native support call successful")
            logger.debug(f"[{self.name()}] Native structured output response: {resp_text}")

            # Now validate and format the JSON output
            try:
                parsed_data = json.loads(resp_text)
                # Validate with pydantic model
                pydantic_model(**parsed_data)
                # Return JSON string representation
                logger.info(f"[{self.name()}] Native structured output generation succeeded")
                return json.dumps(parsed_data), resp_status
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"[{self.name()}] Native structured output validation failed: {e}")
                logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
                # Fall back to instruction-based approach
                return await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )

        except Exception as e:
            logger.error(f"[{self.name()}] Native structured output generation failed: {e}")
            logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
            # Fall back to instruction-based approach
            return await self._generate_fallback_structured_output(
                input, model_params, pydantic_model, **kwargs
            )

    async def _generate_response(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        ret_code = 200
        model_url = model_params.url
        try:
            # create vllm client for every request otherwise it starts failing set header to close connection
            # otherwise spurious event loop errors show up -
            # https://github.com/encode/httpx/discussions/2959#discussioncomment-7665278
            self._set_client(model_url, model_params.auth_token)
            if self.model_config.get("completions_api", False):
                formatted_prompt = self.get_chat_formatted_text(
                    input.messages, **(self.chat_template_params or {})
                )
                payload = self._client.build_request(formatted_prompt=formatted_prompt)
            else:
                payload = self._client.build_request(messages=input.messages)
            completion = await self._client.send_request(
                payload, self.model_serving_name, self.generation_params
            )
            if self.model_config.get("completions_api", False):
                resp_text = completion.choices[0].model_dump()["text"].strip()
            else:
                resp_text = completion.choices[0].model_dump()["message"]["content"].strip()
            # TODO: Test rate limit handling for vllm
        except openai.RateLimitError as e:
            logger.warn(f"vLLM api request exceeded rate limit: {e}")
            resp_text = f"{constants.ERROR_PREFIX} Http request failed {e}"
            ret_code = 429
        except Exception as x:
            resp_text = f"{constants.ERROR_PREFIX} Http request failed {x}"
            logger.error(resp_text)
            rcode = self._get_status_from_body(x)
            if constants.ELEMAI_JOB_DOWN in resp_text or constants.CONNECTION_ERROR in resp_text:
                # inference server is down
                ret_code = 503
            elif rcode is not None:
                ret_code = rcode
            else:
                # for other cases, return 999, dont retry
                ret_code = 999
        return resp_text, ret_code


class CustomOpenAI(BaseCustomModel):

    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        utils.validate_required_keys(
            ["url", "auth_token", "api_version", "model"], model_config, "model"
        )
        self.model_config = model_config

    def _validate_completions_api_support(self) -> None:
        if self.model_config.get("completions_api", False):
            logger.info(f"Model {self.name()} supports completion API.")

    async def _generate_native_structured_output(
        self,
        input: ChatPromptValue,
        model_params: ModelParams,
        pydantic_model: Type[BaseModel],
        **kwargs: Any,
    ) -> Tuple[str, int]:
        """Generate structured output using OpenAI's native support"""
        logger.info(f"[{self.name()}] Attempting native structured output generation")
        model_url = model_params.url
        try:
            self._set_client(model_url, model_params.auth_token)

            # Prepare payload using the client
            if self.model_config.get("completions_api", False):
                formatted_prompt = self.get_chat_formatted_text(
                    input.messages, **(self.chat_template_params or {})
                )
                payload = self._client.build_request(formatted_prompt=formatted_prompt)
            else:
                payload = self._client.build_request(messages=input.messages)

            # Add pydantic_model to generation params
            all_params = {
                **(self.generation_params or {}),
                "pydantic_model": pydantic_model,
            }
            # Send the request using the client
            completion = await self._client.send_request(
                payload, str(self.model_config.get("model")), all_params
            )

            # Check if the request was successful based on the response status
            resp_status = getattr(completion, "status_code", 200)  # Default to 200 if not present

            if resp_status != 200:
                logger.error(
                    f"[{self.name()}] Native structured output request failed with code: {resp_status}"
                )
                # Fall back to instruction-based approach
                logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
                return await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )

            # Extract response text based on API type
            if self.model_config.get("completions_api", False):
                resp_text = completion.choices[0].model_dump()["text"].strip()
            else:
                resp_text = completion.choices[0].model_dump()["message"]["content"].strip()

            logger.info(f"[{self.name()}][{model_url}] RESPONSE: Native support call successful")
            logger.debug(f"[{self.name()}] Native structured output response: {resp_text}")

            # Try to parse and validate the response
            try:
                json_data = json.loads(resp_text)
                # Try to validate with pydantic model
                logger.debug(f"[{self.name()}] Validating response against schema")
                pydantic_model.model_validate(json_data)
                logger.info(f"[{self.name()}] Native structured output generation succeeded")
                return resp_text, resp_status
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"[{self.name()}] Native structured output validation failed: {e}")
                logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
                # Fall back to instruction-based approach
                return await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )

        except Exception as e:
            logger.error(f"[{self.name()}] Native structured output generation failed: {e}")
            logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
            # Fall back to instruction-based approach
            return await self._generate_fallback_structured_output(
                input, model_params, pydantic_model, **kwargs
            )

    async def _generate_response(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        # Check if this is a TTS request based on output_type
        if self.model_config.get("output_type") == "audio":
            return await self._generate_speech(input, model_params)
        else:
            return await self._generate_text(input, model_params)
        

    async def _generate_text(
            self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        """
        Generate text using OpenAI/Azure OpenAI Chat or Completions API.
        This method is called when output_type is 'text' or not specified in model config.
        Args:
            input: ChatPromptValue containing the messages for chat completion
            model_params: Model parameters including URL and auth token
        Returns:
            Tuple of (response_text, status_code)
            - On success: returns generated text and 200
            - On error: returns error message and error code
        """
        ret_code = 200
        model_url = model_params.url
        try:
            # create azure openai client for every request otherwise it starts failing
            # set header to close connection otherwise spurious event loop errors show up - https://github.com/encode/httpx/discussions/2959#discussioncomment-7665278
            self._set_client(model_url, model_params.auth_token)
            if self.model_config.get("completions_api", False):
                formatted_prompt = self.get_chat_formatted_text(
                    input.messages, **(self.chat_template_params or {})
                )
                payload = self._client.build_request(formatted_prompt=formatted_prompt)
            else:
                payload = self._client.build_request(messages=input.messages)
            completion = await self._client.send_request(
                payload, str(self.model_config.get("model")), self.generation_params
            )
            if self.model_config.get("completions_api", False):
                resp_text = completion.choices[0].model_dump()["text"].strip()
            else:
                resp_text = completion.choices[0].model_dump()["message"]["content"].strip()
        except openai.RateLimitError as e:
            logger.warn(f"AzureOpenAI api request exceeded rate limit: {e}")
            resp_text = f"{constants.ERROR_PREFIX} Http request failed {e}"
            ret_code = 429
        except Exception as x:
            resp_text = f"{constants.ERROR_PREFIX} Http request failed {x}"
            logger.error(resp_text)
            rcode = self._get_status_from_body(x)
            ret_code = rcode if rcode else 999
        return resp_text, ret_code

    async def _generate_speech(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        """
        Generate speech from text using OpenAI/Azure OpenAI TTS API.
        This method is called when output_type is 'audio' in model config.

        Args:
            input: ChatPromptValue containing the text to convert to speech
            model_params: Model parameters including URL and auth token

        Returns:
            Tuple of (response_text, status_code)
            - On success: returns base64 encoded audio and 200
            - On error: returns error message and error code
        """
        ret_code = 200
        model_url = model_params.url

        try:
            # Extract text from messages
            text_to_speak = ""
            for message in input.messages:
                if hasattr(message, 'content'):
                    text_to_speak += str(message.content) + " "
            text_to_speak = text_to_speak.strip()

            if not text_to_speak:
                logger.error(f"[{self.name()}] No text provided for TTS conversion")
                return f"{constants.ERROR_PREFIX} No text provided for TTS conversion", 400

            # Validate text length (OpenAI TTS limit is 4096 characters)
            if len(text_to_speak) > 4096:
                logger.warn(f"[{self.name()}] Text exceeds 4096 character limit: {len(text_to_speak)} characters")

            # Set up the OpenAI client
            self._set_client(model_url, model_params.auth_token)

            # Get TTS-specific parameters from generation_params or model_config
            voice = self.generation_params.get('voice', self.model_config.get('voice', None))
            response_format = self.generation_params.get('response_format', self.model_config.get('response_format', 'wav'))
            speed = self.generation_params.get('speed', self.model_config.get('speed', 1.0))

            # Validate speed
            speed = max(0.25, min(4.0, float(speed)))

            logger.debug(f"[{self.name()}] TTS parameters - voice: {voice}, format: {response_format}, speed: {speed}")

            # Prepare TTS request parameters
            tts_params = {
                'input': text_to_speak,
                'voice': voice,
                'response_format': response_format,
                'speed': speed
            }

            # Make the TTS API call
            audio_response = await self._client.create_speech(
                model=str(self.model_config.get("model")),
                **tts_params
            )

            # Map response format to MIME type
            mime_types = {
                'mp3': 'audio/mpeg',
                'opus': 'audio/opus',
                'aac': 'audio/aac',
                'flac': 'audio/flac',
                'wav': 'audio/wav',
                'pcm': 'audio/pcm'
            }
            mime_type = mime_types.get(response_format, 'audio/wav')

            # Create base64 encoded data URL
            audio_base64 = base64.b64encode(audio_response.content).decode('utf-8')
            data_url = f"data:{mime_type};base64,{audio_base64}"
            resp_text = data_url

        except openai.RateLimitError as e:
            logger.warning(f"[{self.name()}] OpenAI TTS API request exceeded rate limit: {e}")
            resp_text = f"{constants.ERROR_PREFIX} Rate limit exceeded: {e}"
            ret_code = 429
        except openai.APIError as e:
            logger.error(f"[{self.name()}] OpenAI TTS API error: {e}")
            resp_text = f"{constants.ERROR_PREFIX} API error: {e}"
            ret_code = getattr(e, 'status_code', 500)
        except Exception as x:
            resp_text = f"{constants.ERROR_PREFIX} TTS request failed: {x}"
            logger.error(f"[{self.name()}] {resp_text}")
            rcode = self._get_status_from_body(x)
            ret_code = rcode if rcode else 999

        return resp_text, ret_code


class CustomOllama(BaseCustomModel):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        self.model_config = model_config

    def _validate_completions_api_support(self) -> None:
        if self.model_config.get("completions_api", False):
            logger.info(f"Model {self.name()} supports completion API.")

    async def _generate_native_structured_output(
        self,
        input: ChatPromptValue,
        model_params: ModelParams,
        pydantic_model: Type[BaseModel],
        **kwargs: Any,
    ) -> Tuple[str, int]:
        """Generate structured output using Ollama's format parameter"""
        logger.info(f"[{self.name()}] Attempting native structured output generation")
        model_url = model_params.url
        try:
            self._set_client(model_url, model_params.auth_token)

            # Create JSON schema for guided generation
            json_schema = pydantic_model.model_json_schema()

            # Prepare payload using the client
            if self.model_config.get("completions_api", False):
                formatted_prompt = self.get_chat_formatted_text(
                    input.messages, **(self.chat_template_params or {})
                )
                payload = self._client.build_request(formatted_prompt=formatted_prompt)
            else:
                payload = self._client.build_request(messages=input.messages)

            # Use Ollama's native structured output using format parameter
            extra_params = {**(self.generation_params or {}), "format": json_schema}

            # Send the request using the client
            completion = await self._client.send_request(payload, self.name(), extra_params)

            # Check if the request was successful based on the response status
            resp_status = getattr(completion, "status_code", 200)  # Default to 200 if not present

            if resp_status != 200:
                logger.error(
                    f"[{self.name()}] Native structured output request failed with code: {resp_status}"
                )
                # Fall back to instruction-based approach
                logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
                return await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )

            # Extract response text based on API type
            if self.model_config.get("completions_api", False):
                resp_text = completion["response"]
            else:
                resp_text = completion["message"]["content"]

            logger.info(f"[{self.name()}][{model_url}] RESPONSE: Native support call successful")
            logger.debug(f"[{self.name()}] Native structured output response: {resp_text}")

            # Now validate and format the JSON output
            try:
                parsed_data = json.loads(resp_text)
                # Try to validate with pydantic model
                logger.debug(f"[{self.name()}] Validating response against schema")
                pydantic_model(**parsed_data)
                logger.info(f"[{self.name()}] Native structured output generation succeeded")
                return resp_text, resp_status
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"[{self.name()}] Native structured output validation failed: {e}")
                logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
                # Fall back to instruction-based approach
                return await self._generate_fallback_structured_output(
                    input, model_params, pydantic_model, **kwargs
                )

        except Exception as e:
            logger.error(f"[{self.name()}] Native structured output generation failed: {e}")
            logger.info(f"[{self.name()}] Falling back to instruction-based structured output")
            # Fall back to instruction-based approach
            return await self._generate_fallback_structured_output(
                input, model_params, pydantic_model, **kwargs
            )

    async def _generate_response(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        ret_code = 200
        model_url = model_params.url
        try:
            self._set_client(model_url, model_params.auth_token)
            if self.model_config.get("completions_api", False):
                formatted_prompt = self.get_chat_formatted_text(
                    input.messages, **(self.chat_template_params or {})
                )
                payload = self._client.build_request(formatted_prompt=formatted_prompt)
            else:
                payload = self._client.build_request(messages=input.messages)
            completion = await self._client.send_request(
                payload, self.name(), self.generation_params
            )
            if self.model_config.get("completions_api", False):
                resp_text = completion["response"]
            else:
                resp_text = completion["message"]["content"]
        except Exception as x:
            resp_text = f"{constants.ERROR_PREFIX} Ollama request failed {x}"
            logger.error(resp_text)
            rcode = self._get_status_from_body(x)
            ret_code = rcode if rcode else 999
        return resp_text, ret_code


class CustomTriton(BaseCustomModel):
    def _get_payload_config_template(self, inference_server: str, payload_key: str = "default"):
        """
        Get the payload configuration template for the Triton server.

        If the Triton server does not define a payload key, read the default flow.

        Args:
            inference_server (str): Inference server type, as of now we support
                only triton, may extend in future
            payload_key (str): Payload key to read from the configuration file.

        Returns:
            dict: The payload configuration template from the configuration file.
        """
        # if triton server does not define payload key, read the default flow
        try:
            payload_cfg = (
                utils.get_payload(inference_server, payload_key)
                if payload_key
                else utils.get_payload(inference_server)
            )
        except Exception as e:
            logger.error(f"Failed to get payload config: {e}")
            raise Exception(f"Failed to get payload config: {e}")
        return payload_cfg

    def _get_payload_json_template(self, payload_cfg: dict):
        """
        Get the payload JSON template for the Triton server.

        If the Triton server does not define a payload key, read the default flow.

        Args:
            payload_cfg (dict): Payload configuration template from the configuration file.
        Returns:
            dict: The payload JSON template from the configuration file.
        """
        # get the payload JSON
        payload_json_template = payload_cfg.get(constants.PAYLOAD_JSON)
        # payload json must be defined for triton server(for the specific API or default)
        assert payload_json_template is not None, "Payload JSON must be defined for Triton server."
        return payload_json_template

    def _create_triton_request(
        self, payload_dict: dict, messages: list[dict], generation_params: dict
    ):
        """This is the triton request payload.

        Read from the config file and build the final payload

        Args:
            payload_dict: payload template dictionary
            messages: messages to be embedded in the configured payload
            generation_params: parameters to be embedded in the configured payload

        Returns:
            final json
        """

        for inp in payload_dict["inputs"]:
            if inp.get("name") == "request":
                inp["data"] = [json.dumps(messages, ensure_ascii=False)]
            elif inp.get("name") == "options":
                inp["data"] = [json.dumps(generation_params, ensure_ascii=False)]

        return payload_dict

    def _get_response_text(self, resp_text: str, payload_config_template: dict):
        try:
            json_resp = json.loads(resp_text)
            final_resp_text = json_resp["outputs"][0]["data"][0]
            # get the response key
            resp_key = payload_config_template.get(constants.RESPONSE_KEY)
            # if resp_text is a dict, just fetch the data or else extract it from json
            final_resp_text = (
                final_resp_text.get(resp_key, "")
                if isinstance(final_resp_text, dict)
                else json.loads(final_resp_text).get(resp_key)
            )

        except Exception as e:
            logger.error(f"Failed to get response text: {e}")
            try:
                json_resp = json.loads(resp_text)
                outer_error = json.loads(json_resp.get("error", "{}"))
                inner_error = json.loads(outer_error.get("error", "{}"))
                error_msg = inner_error.get("error_message")

                if error_msg == "Invalid JSON returned by model.":
                    logger.error("Invalid JSON returned, JSON mode specified")
                    final_resp_text = inner_error.get("model_output") or inner_error.get(
                        "response_metadata", {}
                    ).get("models", [{}])[0].get("debug_info", {}).get("raw_model_output", "")
                else:
                    logger.error(f"Not a JSON error. Error message: {error_msg}")
                    raise RuntimeError("Not a JSON error")
            except Exception:
                logger.error(f"Unable to parse error response {resp_text}")
                final_resp_text = ""
        return final_resp_text

    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        utils.validate_required_keys(["url", "auth_token"], model_config, "model")
        self.model_config = model_config
        self.auth_token = str(model_config.get("auth_token")).replace("Bearer ", "")

    async def _generate_response(
        self, input: ChatPromptValue, model_params: ModelParams
    ) -> Tuple[str, int]:
        ret_code = 200
        model_url = model_params.url
        try:
            # Set Client
            self._set_client(model_url, model_params.auth_token)

            # Build Request
            payload_key = self.model_config.get("payload_key", "default")
            payload_config_template = self._get_payload_config_template(
                constants.INFERENCE_SERVER_TRITON, payload_key
            )
            payload_json_template = self._get_payload_json_template(payload_config_template)
            conversation = utils.convert_messages_from_langchain_to_chat_format(input.messages)
            payload = self._create_triton_request(
                payload_json_template, conversation, self.generation_params
            )
            payload = self._client.build_request_with_payload(payload=payload)

            # Send Request
            resp = await self._client.async_send_request(payload)

            logger.debug(f"[{self.name()}]\n[{model_url}] \n REQUEST DATA: {payload}")

            resp_text = resp.text
            if resp.status_code != 200:
                logger.error(
                    f"[{self.name()}] HTTP request failed with code: {resp.status_code} and error: {resp_text}"
                )
                resp_text = ""
                rcode = self._get_status_from_body(resp_text)
                ret_code = rcode if rcode else 999
            else:
                resp_text = self._get_response_text(resp_text, payload_config_template)

        except Exception as x:
            resp_text = f"Http request failed {x}"
            logger.error(resp_text)
            rcode = self._get_status_from_body(x)
            ret_code = rcode if rcode else 999
        return resp_text, ret_code
