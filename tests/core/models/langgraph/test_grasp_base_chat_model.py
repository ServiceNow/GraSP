import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openai.types.chat.chat_completion import Choice

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from langchain_core.messages import HumanMessage
from openai.types.chat import ChatCompletion, ChatCompletionMessage

import grasp.utils.constants as constants
from grasp.core.models.custom_models import ModelParams
from grasp.core.models.langgraph.grasp_base_chat_model import GraspBaseChatModel


class TestBaseChatModel(unittest.TestCase):
    """Test the GraspBaseChatModel class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Store original constants to restore after tests
        self.original_completion_only_models = (
            constants.COMPLETION_ONLY_MODELS.copy()
            if hasattr(constants, "COMPLETION_ONLY_MODELS")
            else []
        )
        self.original_retryable_http_error = (
            constants.RETRYABLE_HTTP_ERROR.copy()
            if hasattr(constants, "RETRYABLE_HTTP_ERROR")
            else []
        )
        self.original_server_down_error_code = (
            constants.SERVER_DOWN_ERROR_CODE.copy()
            if hasattr(constants, "SERVER_DOWN_ERROR_CODE")
            else []
        )
        self.original_handle_server_down = (
            constants.HANDLE_SERVER_DOWN if hasattr(constants, "HANDLE_SERVER_DOWN") else True
        )
        self.original_max_failed_error = (
            constants.MAX_FAILED_ERROR if hasattr(constants, "MAX_FAILED_ERROR") else 10
        )
        self.original_model_failure_window = (
            constants.MODEL_FAILURE_WINDOW_IN_SEC
            if hasattr(constants, "MODEL_FAILURE_WINDOW_IN_SEC")
            else 30
        )

        # Create a test implementation of GraspBaseChatModel (since it's abstract)
        class TestChatModel(GraspBaseChatModel):
            def _generate_response(self, messages, async_client=True, **kwargs):
                # Mock implementation for testing
                message = ChatCompletionMessage(
                    content="Test response",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                )
                choice = Choice(finish_reason="stop", index=0, message=message, logprobs=None)
                completion = ChatCompletion(
                    id="test-id",
                    choices=[choice],
                    created=1234567890,
                    model="test-model",
                    object="chat.completion",
                    usage=None,
                )
                return completion, 200

            def _llm_type(self) -> str:
                return "test-llm"

            def _sync_generate_response(self, messages, async_client=True, **kwargs):
                # Mock implementation for testing
                return self._generate_response(messages, async_client, **kwargs)

        self.TestChatModel = TestChatModel

        # Basic model configuration for testing
        self.base_config = {
            "name": "test_model",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "url": "http://test-model.com",
            "auth_token": "test-token",
        }

    def tearDown(self):
        """Clean up after each test method"""
        # Restore original constants
        constants.COMPLETION_ONLY_MODELS = self.original_completion_only_models
        constants.RETRYABLE_HTTP_ERROR = self.original_retryable_http_error
        constants.SERVER_DOWN_ERROR_CODE = self.original_server_down_error_code
        constants.HANDLE_SERVER_DOWN = self.original_handle_server_down
        if hasattr(constants, "MAX_FAILED_ERROR"):
            constants.MAX_FAILED_ERROR = self.original_max_failed_error
        if hasattr(constants, "MODEL_FAILURE_WINDOW_IN_SEC"):
            constants.MODEL_FAILURE_WINDOW_IN_SEC = self.original_model_failure_window

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.AutoTokenizer")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_init_basic_config(self, mock_client_factory, mock_tokenizer):
        """Test initialization with basic configuration"""
        model = self.TestChatModel(self.base_config)

        # Verify basic attributes are set correctly
        self.assertEqual(model._config, self.base_config)
        self.assertEqual(model._delay, 100)  # Default value
        self.assertEqual(model._retry_attempts, 8)  # Default value
        self.assertEqual(model._generation_params, self.base_config["parameters"])
        self.assertEqual(model._get_name(), self.base_config["name"])
        self.assertEqual(model._call_count, 0)

        # Verify model stats are initialized correctly
        self.assertEqual(model._model_stats["resp_code_dist"], {})
        self.assertEqual(model._model_stats["errors"], {})

        # Verify tokenizer was not loaded since hf_chat_template_model_id was not provided
        mock_tokenizer.from_pretrained.assert_not_called()

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.AutoTokenizer")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_init_with_hf_template(self, mock_client_factory, mock_tokenizer):
        """Test initialization with HuggingFace template"""
        # Set up mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Configure model with HF template
        config = {
            **self.base_config,
            "hf_chat_template_model_id": "mistralai/Mistral-7B-Instruct-v0.1",
        }

        with patch(
            "grasp.core.models.langgraph.grasp_base_chat_model.os.environ",
            {"HF_TOKEN": "test-hf-token"},
        ):
            model = self.TestChatModel(config)

        # Verify tokenizer was loaded
        mock_tokenizer.from_pretrained.assert_called_once_with(
            "mistralai/Mistral-7B-Instruct-v0.1", token="test-hf-token"
        )

        # Verify _set_chat_template was called
        self.assertIsNotNone(model._tokenizer)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_validate_completions_api_support_error(self, mock_client_factory):
        """Test _validate_completions_api_support raises error when completions_api is True"""
        config = {**self.base_config, "completions_api": True}

        with self.assertRaises(ValueError) as context:
            self.TestChatModel(config)

        # Verify error message
        self.assertIn("does not support completion API", str(context.exception))
        self.assertIn(self.base_config["name"], str(context.exception))
        self.assertIn("models.yaml", str(context.exception))

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_validate_completions_api_support_no_error(self, mock_client_factory):
        """Test _validate_completions_api_support doesn't raise error when completions_api is False"""
        config = {**self.base_config, "completions_api": False}

        try:
            self.TestChatModel(config)
        except ValueError as e:
            self.fail(f"_validate_completions_api_support raised ValueError unexpectedly: {e}")

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_get_model_params_single_url(self, mock_client_factory):
        """Test _get_model_params with a single URL"""
        model = self.TestChatModel(self.base_config)

        # Get model parameters
        params = model._get_model_params()

        # Verify parameters
        self.assertEqual(params.url, self.base_config["url"])
        self.assertEqual(params.auth_token, self.base_config["auth_token"])

        # Verify call count was incremented
        self.assertEqual(model._call_count, 1)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_get_model_params_round_robin(self, mock_client_factory):
        """Test _get_model_params with multiple URLs and round_robin load balancing"""
        config = {
            **self.base_config,
            "url": ["http://model1.com", "http://model2.com", "http://model3.com"],
            "auth_token": ["token1", "token2", "token3"],
            "load_balancing": "round_robin",
        }

        model = self.TestChatModel(config)

        # First call should use the first URL
        params1 = model._get_model_params()
        self.assertEqual(params1.url, "http://model1.com")
        self.assertEqual(params1.auth_token, "token1")
        self.assertEqual(model._call_count, 1)

        # Second call should use the second URL
        params2 = model._get_model_params()
        self.assertEqual(params2.url, "http://model2.com")
        self.assertEqual(params2.auth_token, "token2")
        self.assertEqual(model._call_count, 2)

        # Third call should use the third URL
        params3 = model._get_model_params()
        self.assertEqual(params3.url, "http://model3.com")
        self.assertEqual(params3.auth_token, "token3")
        self.assertEqual(model._call_count, 3)

        # Fourth call should wrap around to the first URL
        params4 = model._get_model_params()
        self.assertEqual(params4.url, "http://model1.com")
        self.assertEqual(params4.auth_token, "token1")
        self.assertEqual(model._call_count, 4)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.random.choice")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_get_model_params_least_requests(self, mock_client_factory, mock_random_choice):
        """Test _get_model_params with multiple URLs and least_requests load balancing"""
        config = {
            **self.base_config,
            "url": ["http://model1.com", "http://model2.com", "http://model3.com"],
            "auth_token": ["token1", "token2", "token3"],
            "load_balancing": "least_requests",
        }

        model = self.TestChatModel(config)

        # Mock random.choice to return predictable results
        mock_random_choice.side_effect = [
            "http://model2.com",
            "http://model1.com",
            "http://model3.com",
        ]

        # First call - all URLs have 0 requests, so random choice is used
        params1 = model._get_model_params()
        self.assertEqual(params1.url, "http://model2.com")
        self.assertEqual(params1.auth_token, "token2")
        self.assertEqual(model._url_reqs_count["http://model2.com"], 1)

        # Second call - model1 and model3 have 0 requests, model2 has 1 request
        params2 = model._get_model_params()
        # should pick a URL with min requests, which is either model1 or model3
        self.assertEqual(params2.url, "http://model1.com")
        self.assertEqual(params2.auth_token, "token1")
        self.assertEqual(model._url_reqs_count["http://model1.com"], 1)

        # Third call - model3 has 0 requests, model1 and model2 have 1 request each
        params3 = model._get_model_params()
        # should pick model3 as it has the least requests
        self.assertEqual(params3.url, "http://model3.com")
        self.assertEqual(params3.auth_token, "token3")
        self.assertEqual(model._url_reqs_count["http://model3.com"], 1)

        # Now all models have 1 request each, so next call should use random choice again
        mock_random_choice.side_effect = ["http://model2.com"]
        params4 = model._get_model_params()
        self.assertEqual(params4.url, "http://model2.com")
        self.assertEqual(model._url_reqs_count["http://model2.com"], 2)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_get_model_params_invalid_load_balancing(self, mock_client_factory):
        """Test _get_model_params with invalid load balancing type"""
        config = {
            **self.base_config,
            "url": ["http://model1.com", "http://model2.com"],
            "auth_token": ["token1", "token2"],
            "load_balancing": "invalid_type",
        }

        model = self.TestChatModel(config)

        with self.assertRaises(ValueError) as context:
            model._get_model_params()

        self.assertIn("Invalid load balancing type", str(context.exception))
        self.assertIn("round_robin", str(context.exception))
        self.assertIn("least_requests", str(context.exception))

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_get_model_params_invalid_url_type(self, mock_client_factory):
        """Test _get_model_params with invalid URL type"""
        config = {
            **self.base_config,
            "url": 12345,  # Invalid type
            "auth_token": "token",
        }

        model = self.TestChatModel(config)

        with self.assertRaises(ValueError) as context:
            model._get_model_params()

        self.assertIn("Model URL should be a string or a list of strings", str(context.exception))

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_set_chat_template_mixtral(self, mock_client_factory):
        """Test _set_chat_template with a model name containing 'mixtral'"""
        # Set up mock tokenizer
        os.environ["GRASP_MIXTRAL-8X7B_CHAT_TEMPLATE"] = "mixtral"
        mock_tokenizer = MagicMock()

        # Create model with mixtral in its name
        config = {**self.base_config, "name": "mixtral-8x7b"}
        model = self.TestChatModel(config)
        model._tokenizer = mock_tokenizer

        # Call the method
        model._set_chat_template()

        # Verify template was set correctly
        self.assertTrue(hasattr(mock_tokenizer, "chat_template"))

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_update_model_stats(self, mock_client_factory, mock_logger):
        """Test _update_model_stats method for updating statistics"""
        self.base_config.update({"stats_interval": 3})
        model = self.TestChatModel(self.base_config)

        # Create mock response with successful status
        message = ChatCompletionMessage(
            content="Test response",
            role="assistant",
            function_call=None,
            tool_calls=None,
        )
        choice = Choice(finish_reason="stop", index=0, message=message, logprobs=None)
        success_response = ChatCompletion(
            id="test-id",
            choices=[choice],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=None,
        )

        # Update stats with successful response
        model._update_model_stats(success_response, 200)

        # Verify response code distribution was updated
        self.assertEqual(model._model_stats["resp_code_dist"][200], 1)

        # Create mock response with error
        error_message = ChatCompletionMessage(
            content="Connection error occurred",
            role="assistant",
            function_call=None,
            tool_calls=None,
        )
        error_choice = Choice(finish_reason="stop", index=0, message=error_message, logprobs=None)
        error_response = ChatCompletion(
            id="test-error-id",
            choices=[error_choice],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=None,
        )

        # Update stats with error response
        model._update_model_stats(error_response, 500)

        # Verify error count and response code distribution were updated
        self.assertEqual(model._model_stats["resp_code_dist"][500], 1)
        self.assertEqual(model._model_stats["resp_code_dist"][200], 1)
        self.assertEqual(model._model_stats["errors"]["connection_error"], 1)

        # Test with timeout error
        timeout_message = ChatCompletionMessage(
            content="Request timed out",
            role="assistant",
            function_call=None,
            tool_calls=None,
        )
        timeout_choice = Choice(
            finish_reason="stop", index=0, message=timeout_message, logprobs=None
        )
        timeout_response = ChatCompletion(
            id="test-timeout-id",
            choices=[timeout_choice],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=None,
        )

        # Update stats with timeout response
        model._update_model_stats(timeout_response, 408)

        # Verify timeout error was recorded
        self.assertEqual(model._model_stats["resp_code_dist"][408], 1)
        self.assertEqual(model._model_stats["errors"]["timeout"], 1)

        # Add one more response to reach the interval
        model._update_model_stats(success_response, 200)

        # Verify logger.info was called with model stats
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("[test_model] Model Stats", log_message)
        self.assertIn("total_requests", log_message)
        self.assertIn("resp_code_dist", log_message)
        self.assertIn("errors", log_message)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_update_model_stats_token_exceeded(self, mock_client_factory, mock_logger):
        """Test _update_model_stats with token exceeded error"""
        model = self.TestChatModel(self.base_config)

        # Create mock response with token exceeded error
        error_message = ChatCompletionMessage(
            content="maximum context length is 4096 tokens, but received 5000",
            role="assistant",
            function_call=None,
            tool_calls=None,
        )
        error_choice = Choice(finish_reason="length", index=0, message=error_message, logprobs=None)
        error_response = ChatCompletion(
            id="test-error-id",
            choices=[error_choice],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=None,
        )

        # Update stats with token exceeded response
        model._update_model_stats(error_response, 413)

        # Verify token exceeded error was recorded
        self.assertEqual(model._model_stats["resp_code_dist"][413], 1)
        self.assertEqual(model._model_stats["errors"]["tokens_exceeded"], 1)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.sys.exit")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_handle_server_down_disabled(self, mock_client_factory, mock_logger, mock_exit):
        """Test _handle_server_down when server down handling is disabled"""
        # Set HANDLE_SERVER_DOWN to False for this test
        constants.HANDLE_SERVER_DOWN = False

        model = self.TestChatModel(self.base_config)

        # Call with a server down error code
        model._handle_server_down(500)

        # Verify that exit was not called and timestamps were not recorded
        mock_exit.assert_not_called()
        self.assertEqual(len(model._model_failed_response_timestamp), 0)

        # Restore HANDLE_SERVER_DOWN
        constants.HANDLE_SERVER_DOWN = True

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.sys.exit")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_handle_server_down_normal_operation(self, mock_client_factory, mock_logger, mock_exit):
        """Test _handle_server_down with normal operation"""
        constants.HANDLE_SERVER_DOWN = True
        constants.SERVER_DOWN_ERROR_CODE = [404, 500, 501, 502, 503]
        model = self.TestChatModel(self.base_config)

        # Call with non-server down error code
        model._handle_server_down(400)

        # Verify that no timestamps were recorded
        self.assertEqual(len(model._model_failed_response_timestamp), 0)

        # Call with server down error code
        model._handle_server_down(500)

        # Verify that timestamp was recorded
        self.assertEqual(len(model._model_failed_response_timestamp), 1)

        # Verify that exit was not called since we don't have enough errors
        mock_exit.assert_not_called()

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.time")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.sys.exit")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_handle_server_down_critical(
        self, mock_client_factory, mock_logger, mock_exit, mock_time
    ):
        """Test _handle_server_down with critical server down detection"""
        constants.HANDLE_SERVER_DOWN = True
        constants.SERVER_DOWN_ERROR_CODE = [404, 500, 501, 502, 503]
        constants.MAX_FAILED_ERROR = 5
        constants.MODEL_FAILURE_WINDOW_IN_SEC = 30

        model = self.TestChatModel(self.base_config)

        # Set up mock time to return increasing timestamps
        # First errors happen within short time window (10 seconds)
        start_time = 1000.0
        mock_time.time.side_effect = [
            start_time,
            start_time + 2.0,
            start_time + 4.0,
            start_time + 6.0,
            start_time + 8.0,
        ]

        # Trigger MAX_FAILED_ERROR server down errors
        for i in range(constants.MAX_FAILED_ERROR):
            model._handle_server_down(500)

        # Verify that exit was called since we had enough errors in a short window
        mock_exit.assert_called_once()
        mock_logger.error.assert_called_once()

        # Verify warning message was logged
        warning_message = mock_logger.warning.call_args[0][0]
        self.assertIn(f"Server failure count: {constants.MAX_FAILED_ERROR}", warning_message)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.time")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.sys.exit")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_handle_server_down_spread_out(
        self, mock_client_factory, mock_logger, mock_exit, mock_time
    ):
        """Test _handle_server_down with errors spread out over time"""
        constants.HANDLE_SERVER_DOWN = True
        constants.SERVER_DOWN_ERROR_CODE = [404, 500, 501, 502, 503]
        constants.MAX_FAILED_ERROR = 5
        constants.MODEL_FAILURE_WINDOW_IN_SEC = 30

        model = self.TestChatModel(self.base_config)

        # Set up mock time to return increasing timestamps
        # Errors happen over a longer time window (60 seconds, which is > MODEL_FAILURE_WINDOW_IN_SEC)
        start_time = 1000.0
        mock_time.time.side_effect = [
            start_time,
            start_time + 10.0,
            start_time + 20.0,
            start_time + 40.0,
            start_time
            + 60.0,  # Last error is 60 seconds after first, exceeding the 30 second window
        ]

        # Trigger MAX_FAILED_ERROR server down errors spread over time
        for i in range(constants.MAX_FAILED_ERROR):
            model._handle_server_down(500)

        # Verify that timestamps were recorded
        self.assertEqual(len(model._model_failed_response_timestamp), constants.MAX_FAILED_ERROR)

        # Verify that exit was NOT called since errors are spread out over time
        mock_exit.assert_not_called()

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.time")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.sys.exit")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_handle_server_down_max_queue(
        self, mock_client_factory, mock_logger, mock_exit, mock_time
    ):
        """Test _handle_server_down with more errors than MAX_FAILED_ERROR"""
        constants.HANDLE_SERVER_DOWN = True
        constants.SERVER_DOWN_ERROR_CODE = [404, 500, 501, 502, 503]
        constants.MAX_FAILED_ERROR = 5
        constants.MODEL_FAILURE_WINDOW_IN_SEC = 30

        model = self.TestChatModel(self.base_config)

        # Set up mock time to return increasing timestamps
        # Generate MAX_FAILED_ERROR + 3 timestamps with enough spread
        start_time = 1000.0
        timestamps = [start_time + i * 10 for i in range(constants.MAX_FAILED_ERROR + 3)]
        mock_time.time.side_effect = timestamps

        # Trigger more than MAX_FAILED_ERROR server down errors
        for i in range(constants.MAX_FAILED_ERROR + 3):
            model._handle_server_down(500)

        # Verify that only MAX_FAILED_ERROR timestamps are kept
        self.assertEqual(len(model._model_failed_response_timestamp), constants.MAX_FAILED_ERROR)

        # The queue should have the most recent timestamps
        self.assertEqual(model._model_failed_response_timestamp[0], timestamps[3])
        self.assertEqual(model._model_failed_response_timestamp[-1], timestamps[-1])

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_is_retryable_error(self, mock_client_factory):
        """Test _is_retryable_error method with various error codes"""
        model = self.TestChatModel(self.base_config)

        # Store original retryable error codes
        original_retryable_errors = (
            constants.RETRYABLE_HTTP_ERROR.copy()
            if hasattr(constants, "RETRYABLE_HTTP_ERROR")
            else []
        )

        # Set up test retryable error codes
        constants.RETRYABLE_HTTP_ERROR = [429, 444, 599]

        # Test valid retryable errors
        self.assertTrue(model._is_retryable_error(("error message", 429)))
        self.assertTrue(model._is_retryable_error(("error message", 444)))
        self.assertTrue(model._is_retryable_error(("error message", 599)))

        # Test non-retryable errors
        self.assertFalse(model._is_retryable_error(("error message", 400)))
        self.assertFalse(model._is_retryable_error(("error message", 401)))
        self.assertFalse(model._is_retryable_error(("error message", 404)))
        self.assertFalse(model._is_retryable_error(("error message", 500)))

        # Restore original retryable error codes
        constants.RETRYABLE_HTTP_ERROR = original_retryable_errors

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_is_retryable_error_edge_cases(self, mock_client_factory):
        """Test _is_retryable_error method with edge cases"""
        model = self.TestChatModel(self.base_config)

        # Store original retryable error codes
        original_retryable_errors = (
            constants.RETRYABLE_HTTP_ERROR.copy()
            if hasattr(constants, "RETRYABLE_HTTP_ERROR")
            else []
        )

        # Set up test retryable error codes
        constants.RETRYABLE_HTTP_ERROR = [429, 444, 599]

        # Test with invalid input format (not a tuple)
        self.assertFalse(model._is_retryable_error("not a tuple"))

        # Test with tuple of wrong length
        self.assertFalse(model._is_retryable_error(("error message",)))  # tuple with 1 element
        self.assertFalse(
            model._is_retryable_error(("error message", 429, "extra"))
        )  # tuple with 3 elements

        # Test with tuple containing wrong types
        self.assertFalse(model._is_retryable_error(("error message", "not an int")))

        # Test with empty tuple
        self.assertFalse(model._is_retryable_error(()))

        # Restore original retryable error codes
        constants.RETRYABLE_HTTP_ERROR = original_retryable_errors

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_log_before_retry(self, mock_client_factory, mock_logger):
        """Test _log_before_retry method that logs before retrying a request"""
        model = self.TestChatModel(self.base_config)

        # Create a mock retry state
        mock_retry_state = MagicMock()
        # Set up the outcome.result() to return a tuple (like the real API would)
        mock_outcome = MagicMock()
        mock_outcome.result.return_value = ("Error message", 429)
        mock_retry_state.outcome = mock_outcome

        # Set up next_action with a sleep value
        mock_next_action = MagicMock()
        mock_next_action.sleep = 2.5
        mock_retry_state.next_action = mock_next_action

        # Call the method
        model._log_before_retry(mock_retry_state)

        # Verify that logger.warning was called with appropriate message
        mock_logger.warning.assert_called()

        # Check that the warning message contains the expected information
        warning_calls = mock_logger.warning.call_args_list
        # The method should call warning twice with the same message
        self.assertEqual(len(warning_calls), 2)

        # Verify both warning messages contain the model name and response code
        for call_args in warning_calls:
            warning_message = call_args[0][0]
            self.assertIn(model._get_name(), warning_message)
            self.assertIn("429", warning_message)
            self.assertIn("2.5", warning_message)
            self.assertIn("Retrying the request", warning_message)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.AsyncRetrying")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.wait_random_exponential")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.stop_after_attempt")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.retry_if_result")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_generate_response_with_retry_success(
        self,
        mock_client_factory,
        mock_logger,
        mock_retry_if_result,
        mock_stop_after_attempt,
        mock_wait_random_exponential,
        mock_async_retrying,
    ):
        """Test _generate_response_with_retry with successful retry"""
        model = self.TestChatModel(self.base_config)

        # Set up the test messages
        messages = [HumanMessage(content="Test message")]

        # Create a mock AsyncRetrying instance
        mock_retrying_instance = MagicMock()
        mock_async_retrying.return_value = mock_retrying_instance

        # Set up the __aiter__ method to return self
        mock_retrying_instance.__aiter__.return_value = mock_retrying_instance

        # Set up the __anext__ method to return a value once and then raise StopAsyncIteration
        response_value = ("Success response", 200)

        async def mock_anext():
            # Return the value first time, then stop iteration
            mock_retrying_instance.__anext__.side_effect = StopAsyncIteration()
            return response_value

        mock_retrying_instance.__anext__.side_effect = mock_anext

        # Call the method and get the result
        import asyncio

        model_params = ModelParams(url="http://test-url", auth_token="test-token")
        asyncio.run(model._generate_response_with_retry(messages, model_params))

        # Verify that AsyncRetrying was configured correctly
        mock_wait_random_exponential.assert_called_once_with(multiplier=1)
        mock_stop_after_attempt.assert_called_once_with(model._retry_attempts)
        mock_retry_if_result.assert_called_once()
        # Verify the retry_if_result callback is _is_retryable_error
        callback = mock_retry_if_result.call_args[0][0]
        self.assertEqual(callback, model._is_retryable_error)

        # Verify that AsyncRetrying was initialized correctly
        mock_async_retrying.assert_called_once_with(
            retry=mock_retry_if_result.return_value,
            wait=mock_wait_random_exponential.return_value,
            stop=mock_stop_after_attempt.return_value,
            before_sleep=model._log_before_retry,
        )

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.AsyncRetrying")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.wait_random_exponential")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.stop_after_attempt")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.retry_if_result")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_generate_response_with_retry_failure(
        self,
        mock_client_factory,
        mock_logger,
        mock_retry_if_result,
        mock_stop_after_attempt,
        mock_wait_random_exponential,
        mock_async_retrying,
    ):
        """Test _generate_response_with_retry with failed retries"""
        # Import the actual RetryError to use in the test
        from tenacity import RetryError

        model = self.TestChatModel(self.base_config)

        # Set up the test messages
        messages = [HumanMessage(content="Test message")]

        # Create a mock for _generate_response
        mock_generate_response = MagicMock()
        model._generate_response = mock_generate_response

        # Create a mock last_attempt
        mock_last_attempt = MagicMock()

        # Set up AsyncRetrying to raise RetryError when __aiter__ is called
        mock_async_retrying.side_effect = RetryError(last_attempt=mock_last_attempt)

        # Call the method and get the result - should not raise exception as RetryError is caught
        import asyncio

        model_params = ModelParams(url="http://test-url", auth_token="test-token")
        result = asyncio.run(model._generate_response_with_retry(messages, model_params))

        # Verify result is None when RetryError occurs
        self.assertIsNone(result)

        # Verify that AsyncRetrying was configured correctly
        mock_wait_random_exponential.assert_called_once_with(multiplier=1)
        mock_stop_after_attempt.assert_called_once_with(model._retry_attempts)

        # Verify logger error was called
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        self.assertIn(f"Request failed after {model._retry_attempts} attempts", error_message)

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.Retrying")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.wait_random_exponential")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.stop_after_attempt")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.retry_if_result")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_sync_generate_response_with_retry_success(
        self,
        mock_client_factory,
        mock_retry_if_result,
        mock_stop_after_attempt,
        mock_wait_random_exponential,
        mock_retrying,
    ):
        """Test _generate_response_with_retry with successful retry"""
        model = self.TestChatModel(self.base_config)

        # Set up the test messages
        messages = [HumanMessage(content="Test message")]

        # Create a mock AsyncRetrying instance
        mock_retrying_instance = MagicMock()
        mock_retrying.return_value = mock_retrying_instance

        # Set up the __aiter__ method to return self
        mock_retrying_instance.call.return_value = mock_retrying_instance

        # Set up the __anext__ method to return a value once and then raise StopAsyncIteration
        response_value = ("Success response", 200)

        def mock_next():
            # Return the value first time, then stop iteration
            mock_retrying_instance.call.side_effect = StopIteration()
            return response_value

        mock_retrying_instance.call.side_effect = mock_next

        # Call the method and get the result
        model_params = ModelParams(url="http://test-url", auth_token="test-token")
        model._sync_generate_response_with_retry(messages, model_params)

        # Verify that AsyncRetrying was configured correctly
        mock_wait_random_exponential.assert_called_once_with(multiplier=1)
        mock_stop_after_attempt.assert_called_once_with(model._retry_attempts)
        mock_retry_if_result.assert_called_once()
        # Verify the retry_if_result callback is _is_retryable_error
        callback = mock_retry_if_result.call_args[0][0]
        self.assertEqual(callback, model._is_retryable_error)

        # Verify that AsyncRetrying was initialized correctly
        mock_retrying.assert_called_once_with(
            retry=mock_retry_if_result.return_value,
            wait=mock_wait_random_exponential.return_value,
            stop=mock_stop_after_attempt.return_value,
            before_sleep=model._log_before_retry,
        )

    @patch("grasp.core.models.langgraph.grasp_base_chat_model.Retrying")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.wait_random_exponential")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.stop_after_attempt")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.retry_if_result")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.logger")
    @patch("grasp.core.models.langgraph.grasp_base_chat_model.ClientFactory")
    def test_sync_generate_response_with_retry_failure(
        self,
        mock_client_factory,
        mock_logger,
        mock_retry_if_result,
        mock_stop_after_attempt,
        mock_wait_random_exponential,
        mock_retrying,
    ):
        """Test _sync_generate_response_with_retry with failed retries"""
        # Import the actual RetryError to use in the test
        from tenacity import RetryError

        model = self.TestChatModel(self.base_config)

        # Set up the test messages
        messages = [HumanMessage(content="Test message")]

        # Create a mock for _sync_generate_response
        mock_sync_generate_response = MagicMock()
        model._sync_generate_response = mock_sync_generate_response

        # Create a mock last_attempt
        mock_last_attempt = MagicMock()

        # Set up mock_retrying to raise RetryError when called
        mock_retrying.side_effect = RetryError(last_attempt=mock_last_attempt)

        # Call the method - should NOT raise exception as RetryError is caught
        model_params = ModelParams(url="http://test-url", auth_token="test-token")
        result = model._sync_generate_response_with_retry(messages, model_params)

        # Verify result is None when RetryError occurs
        self.assertIsNone(result)

        # Verify that Retrying was configured correctly
        mock_wait_random_exponential.assert_called_once_with(multiplier=1)
        mock_stop_after_attempt.assert_called_once_with(model._retry_attempts)

        # Verify logger error was called
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        self.assertIn(f"Request failed after {model._retry_attempts} attempts", error_message)
