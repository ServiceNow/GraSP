import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from langchain_core.messages import HumanMessage

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from grasp.core.models.custom_models import ModelParams
from grasp.core.models.langgraph.openai_chat_model import CustomOpenAIChatModel
from grasp.utils import constants


class TestOpenAIChatModel(unittest.TestCase):
    """
    Unit tests for the CustomOpenAIChatModel class.

    This test suite validates the functionality of the OpenAI chat model implementation
    in the GraSP framework. The CustomOpenAIChatModel works with the ClientFactory to
    establish connections with OpenAI-compatible model servers, handling both synchronous
    and asynchronous communication patterns.

    The ModelFactory uses this class when initializing OpenAI model types for agent nodes.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.base_config = {
            "name": "test_openai_model",
            "model_type": "azure_openai",
            "url": "https://test-openai-endpoint.com",
            "api_key": "test_api_key",
            "parameters": {"temperature": 0.7, "max_tokens": 500},
        }

        # Save original constants
        self.original_error_prefix = constants.ERROR_PREFIX

    def tearDown(self):
        """Clean up test environment after each test."""
        # Restore original constants
        constants.ERROR_PREFIX = self.original_error_prefix

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_logger):
        """Test successful response generation."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request and send_request methods
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )
        mock_client.send_request = AsyncMock(
            return_value={
                "id": "test-id",
                "choices": [{"message": {"content": "Test response"}}],
            }
        )

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method
            model_params = ModelParams(url="http://test-url", auth_token="test-token")
            response, status_code = await model._generate_response(messages, model_params)

        # Verify the response
        self.assertEqual(status_code, 200)
        self.assertEqual(
            response,
            {"id": "test-id", "choices": [{"message": {"content": "Test response"}}]},
        )

        # Verify the client methods were called correctly
        mock_client.build_request.assert_called_once_with(messages=messages)
        mock_client.send_request.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Hello"}]},
            self.base_config.get("model"),
            self.base_config.get("parameters"),
        )

        # Verify no errors were logged
        mock_logger.error.assert_not_called()

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    @pytest.mark.asyncio
    async def test_generate_response_rate_limit_error(self, mock_logger):
        """Test handling of rate limit errors."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request method
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )

        # Create a properly mocked RateLimitError with required arguments
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}
        mock_response.text = "Rate limit exceeded"

        # Set up the mock client's send_request method to raise a RateLimitError
        rate_limit_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
        )
        mock_client.send_request = AsyncMock(side_effect=rate_limit_error)

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method
            response, status_code = await model._generate_response(messages)

        # Verify the response
        self.assertEqual(status_code, 429)
        self.assertTrue(response.startswith(constants.ERROR_PREFIX))
        self.assertIn("Rate limit exceeded", response)

        # Verify warning was logged
        mock_logger.warn.assert_called_once()
        warn_message = mock_logger.warn.call_args[0][0]
        self.assertIn("exceeded rate limit", warn_message)

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    @pytest.mark.asyncio
    async def test_generate_response_generic_exception(self, mock_logger):
        """Test handling of generic exceptions."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request method
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )

        # Set up the mock client's send_request method to raise a generic exception
        generic_error = Exception("Generic error")
        mock_client.send_request = AsyncMock(side_effect=generic_error)

        # Mock _get_status_from_body to return a status code
        model._get_status_from_body = MagicMock(return_value=500)

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method
            response, status_code = await model._generate_response(messages)

        # Verify the response
        self.assertEqual(status_code, 500)
        self.assertTrue(response.startswith(constants.ERROR_PREFIX))
        self.assertIn("Generic error", response)

        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        self.assertIn("Http request failed", error_message)

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    @pytest.mark.asyncio
    async def test_generate_response_status_not_found(self, mock_logger):
        """Test handling of exceptions where status code cannot be extracted."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request method
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )

        # Set up the mock client's send_request method to raise a generic exception
        generic_error = Exception("No status code")
        mock_client.send_request = AsyncMock(side_effect=generic_error)

        # Mock _get_status_from_body to return None
        model._get_status_from_body = MagicMock(return_value=None)

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method
            response, status_code = await model._generate_response(messages)

        # Verify the response - should use default status code 999
        self.assertEqual(status_code, 999)
        self.assertTrue(response.startswith(constants.ERROR_PREFIX))
        self.assertIn("No status code", response)

        # Verify error was logged
        mock_logger.error.assert_called_once()

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    @patch("grasp.core.models.langgraph.openai_chat_model.GraspBaseChatModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_with_client_factory(self, mock_set_client, mock_logger):
        """
        Test response generation with proper _set_client integration.

        This test verifies that the model correctly uses the _set_client method
        which is responsible for obtaining a client instance from ClientFactory.
        The ClientFactory creates appropriate clients for different model types.
        """
        model = CustomOpenAIChatModel(self.base_config)

        # Create a mock client
        mock_client = MagicMock()
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )
        mock_client.send_request = AsyncMock(
            return_value={
                "id": "test-id",
                "choices": [{"message": {"content": "Test response"}}],
            }
        )

        # Have _set_client correctly set the client
        def mock_set_client_implementation(async_client=True):
            model._client = mock_client

        mock_set_client.side_effect = mock_set_client_implementation

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Call the method
        response, status_code = await model._generate_response(messages)

        # Verify _set_client was called
        mock_set_client.assert_called_once()

        # Verify the response
        self.assertEqual(status_code, 200)
        self.assertEqual(
            response,
            {"id": "test-id", "choices": [{"message": {"content": "Test response"}}]},
        )

    @patch("grasp.core.models.langgraph.vllm_chat_model.logger")
    @pytest.mark.asyncio
    async def test_generate_response_with_additional_kwargs(self, mock_logger):
        """Test synchronous response generation with additional kwargs passed."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request and send_request methods
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )
        mock_client.send_request = MagicMock(
            return_value={
                "id": "test-id",
                "choices": [{"message": {"content": "Test response"}}],
            }
        )

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Additional kwargs to pass
        additional_kwargs = {
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                    },
                }
            ],
        }

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method with additional kwargs
            response, status_code = await model._generate_response(messages, **additional_kwargs)

        # Verify the response
        self.assertEqual(status_code, 200)

        # Verify the client methods were called correctly with the additional kwargs
        mock_client.build_request.assert_called_once_with(
            messages=messages,
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                    },
                }
            ],
        )

        # Verify no errors were logged
        mock_logger.error.assert_not_called()

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    def test_sync_generate_response_success(self, mock_logger):
        """Test successful synchronous response generation."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request and send_request methods
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )
        mock_client.send_request = MagicMock(
            return_value={
                "id": "test-id",
                "choices": [{"message": {"content": "Test response"}}],
            }
        )

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method
            model_params = ModelParams(url="http://test-url", auth_token="test-token")
            response, status_code = model._sync_generate_response(messages, model_params)

        # Verify the response
        self.assertEqual(status_code, 200)
        self.assertEqual(
            response,
            {"id": "test-id", "choices": [{"message": {"content": "Test response"}}]},
        )

        # Verify the client methods were called correctly
        mock_client.build_request.assert_called_once_with(messages=messages)
        mock_client.send_request.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Hello"}]},
            self.base_config.get("name"),
            self.base_config.get("parameters"),
        )

        # Verify no errors were logged
        mock_logger.error.assert_not_called()

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    def test_sync_generate_response_rate_limit_error(self, mock_logger):
        """Test handling of rate limit errors in synchronous mode."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request method
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )

        # Create a properly mocked RateLimitError with required arguments
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}
        mock_response.text = "Rate limit exceeded"

        # Set up the mock client's send_request method to raise a RateLimitError
        rate_limit_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
        )
        mock_client.send_request = MagicMock(side_effect=rate_limit_error)

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method
            model_params = ModelParams(url="http://test-url", auth_token="test-token")
            response, status_code = model._sync_generate_response(messages, model_params)

        # Verify the response
        self.assertEqual(status_code, 429)
        self.assertTrue(response.startswith(constants.ERROR_PREFIX))
        self.assertIn("Rate limit exceeded", response)

        # Verify warning was logged
        mock_logger.warn.assert_called_once()
        warn_message = mock_logger.warn.call_args[0][0]
        self.assertIn("exceeded rate limit", warn_message)

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    def test_sync_generate_response_generic_exception(self, mock_logger):
        """Test handling of generic exceptions in synchronous mode."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request method
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )

        # Set up the mock client's send_request method to raise a generic exception
        generic_error = Exception("Generic error")
        mock_client.send_request = MagicMock(side_effect=generic_error)

        # Mock _get_status_from_body to return a status code
        model._get_status_from_body = MagicMock(return_value=500)

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method
            model_params = ModelParams(url="http://test-url", auth_token="test-token")
            response, status_code = model._sync_generate_response(messages, model_params)

        # Verify the response
        self.assertEqual(status_code, 500)
        self.assertTrue(response.startswith(constants.ERROR_PREFIX))
        self.assertIn("Generic error", response)

        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        self.assertIn("Http request failed", error_message)

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    def test_sync_generate_response_status_not_found(self, mock_logger):
        """Test handling of exceptions where status code cannot be extracted."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request method
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )

        # Set up the mock client's send_request method to raise a generic exception
        generic_error = Exception("No status code")
        mock_client.send_request = MagicMock(side_effect=generic_error)

        # Mock _get_status_from_body to return None
        model._get_status_from_body = MagicMock(return_value=None)

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method
            model_params = ModelParams(url="http://test-url", auth_token="test-token")
            response, status_code = model._sync_generate_response(messages, model_params)

        # Verify the response - should use default status code 999
        self.assertEqual(status_code, 999)
        self.assertTrue(response.startswith(constants.ERROR_PREFIX))
        self.assertIn("No status code", response)

        # Verify error was logged
        mock_logger.error.assert_called_once()

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    def test_sync_generate_response_with_additional_kwargs(self, mock_logger):
        """Test synchronous response generation with additional kwargs passed."""
        model = CustomOpenAIChatModel(self.base_config)

        # Mock the client
        mock_client = MagicMock()
        model._client = mock_client

        # Set up the mock client's build_request and send_request methods
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )
        mock_client.send_request = MagicMock(
            return_value={
                "id": "test-id",
                "choices": [{"message": {"content": "Test response"}}],
            }
        )

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Additional kwargs to pass
        additional_kwargs = {
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                    },
                }
            ],
        }

        # Patch _set_client to avoid actual client creation through ClientFactory
        with patch.object(model, "_set_client"):
            # Call the method with additional kwargs
            model_params = ModelParams(url="http://test-url", auth_token="test-token")
            response, status_code = model._sync_generate_response(
                messages, model_params, **additional_kwargs
            )

        # Verify the response
        self.assertEqual(status_code, 200)

        # Verify the client methods were called correctly with the additional kwargs
        mock_client.build_request.assert_called_once_with(
            messages=messages,
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                    },
                }
            ],
        )

        # Verify no errors were logged
        mock_logger.error.assert_not_called()

    @patch("grasp.core.models.langgraph.openai_chat_model.logger")
    @patch("grasp.core.models.langgraph.openai_chat_model.GraspBaseChatModel._set_client")
    def test_sync_generate_response_with_client_factory(self, mock_set_client, mock_logger):
        """
        Test sync response generation with proper _set_client integration.

        This test validates that CustomOpenAIChatModel correctly interacts with the
        ClientFactory through the _set_client method to obtain appropriate client instances
        for synchronous API calls.
        """
        model = CustomOpenAIChatModel(self.base_config)

        # Create a mock client
        mock_client = MagicMock()
        mock_client.build_request = MagicMock(
            return_value={"messages": [{"role": "user", "content": "Hello"}]}
        )
        mock_client.send_request = MagicMock(
            return_value={
                "id": "test-id",
                "choices": [{"message": {"content": "Test response"}}],
            }
        )

        model_url = "http://test-url"
        model_auth_token = "test-token"
        model_params = ModelParams(url=model_url, auth_token=model_auth_token)

        # Have _set_client correctly set the client
        def mock_set_client_implementation(model_url, model_auth_token, async_client=True):
            model._client = mock_client

        mock_set_client.side_effect = mock_set_client_implementation

        # Create test messages
        messages = [HumanMessage(content="Hello")]

        # Call the method with async_client=False
        response, status_code = model._sync_generate_response(
            messages, model_params, async_client=False
        )

        # Verify _set_client was called with async_client=False
        mock_set_client.assert_called_once_with(model_url, "test-token", async_client=False)

        # Verify the response
        self.assertEqual(status_code, 200)
        self.assertEqual(
            response,
            {"id": "test-id", "choices": [{"message": {"content": "Test response"}}]},
        )


if __name__ == "__main__":
    unittest.main()
