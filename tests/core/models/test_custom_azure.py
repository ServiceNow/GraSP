import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from sygra.core.models.custom_models import CustomAzure, ModelParams


class TestCustomAzure(unittest.TestCase):
    """Unit tests for the CustomAzure class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Base model configuration with string auth_token
        self.base_config = {
            "name": "azure_model",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "url": "https://azure-test.openai.azure.com",
            "auth_token": "Bearer test_token_123",
        }

        # Configuration with list of auth_tokens
        self.multi_token_config = {
            "name": "azure_model_multi",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "url": "https://azure-test.openai.azure.com",
            "auth_token": ["Bearer token1", "Bearer token2", "Bearer token3"],
        }

        # Mock messages
        self.messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]
        self.chat_input = ChatPromptValue(messages=self.messages)

    def test_init_with_string_auth_token(self):
        """Test initialization with a single string auth_token"""
        custom_azure = CustomAzure(self.base_config)

        # Verify model was properly initialized
        self.assertEqual(custom_azure.model_config, self.base_config)
        self.assertEqual(custom_azure.generation_params, self.base_config["parameters"])
        self.assertEqual(custom_azure.auth_token, "test_token_123")  # Bearer prefix removed
        self.assertEqual(custom_azure.name(), "azure_model")

    def test_init_with_list_auth_token(self):
        """Test initialization with a list of auth_tokens"""
        custom_azure = CustomAzure(self.multi_token_config)

        # Verify model was properly initialized with first token
        self.assertEqual(custom_azure.model_config, self.multi_token_config)
        self.assertEqual(custom_azure.auth_token, "token1")  # First token, Bearer prefix removed
        self.assertEqual(custom_azure.name(), "azure_model_multi")

    def test_init_with_empty_list_raises_error(self):
        """Test initialization with an empty list raises ValueError"""
        config = {**self.base_config, "auth_token": []}

        with self.assertRaises(ValueError) as context:
            CustomAzure(config)

        self.assertIn("auth_token must be a string or non-empty list", str(context.exception))

    def test_init_with_invalid_type_raises_error(self):
        """Test initialization with invalid auth_token type raises ValueError"""
        config = {**self.base_config, "auth_token": 12345}

        with self.assertRaises(ValueError) as context:
            CustomAzure(config)

        self.assertIn("auth_token must be a string or non-empty list", str(context.exception))

    def test_init_with_list_containing_non_string_raises_error(self):
        """Test initialization with list containing non-string raises TypeError"""
        config = {**self.base_config, "auth_token": [123, "token"]}

        with self.assertRaises(TypeError) as context:
            CustomAzure(config)

        self.assertIn("auth_token list must contain strings", str(context.exception))

    def test_init_missing_url_raises_error(self):
        """Test initialization without url raises error"""
        config = {
            "name": "azure_model",
            "parameters": {"temperature": 0.7},
            "auth_token": "test_token",
        }

        with self.assertRaises(Exception):
            CustomAzure(config)

    def test_init_missing_auth_token_raises_error(self):
        """Test initialization without auth_token raises error"""
        config = {
            "name": "azure_model",
            "parameters": {"temperature": 0.7},
            "url": "https://azure-test.openai.azure.com",
        }

        with self.assertRaises(Exception):
            CustomAzure(config)

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_utils, mock_set_client):
        """Test _generate_response method with successful response"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {
            "messages": [{"role": "user", "content": "Hello"}]
        }

        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": "Hello! I'm doing well, thank you!"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Setup custom model
        custom_azure = CustomAzure(self.base_config)
        custom_azure._client = mock_client

        # Call _generate_response
        model_params = ModelParams(
            url="https://azure-test.openai.azure.com", auth_token="test_token"
        )
        resp_text, resp_status = await custom_azure._generate_response(
            self.chat_input, model_params
        )

        # Verify results
        self.assertEqual(resp_text, "Hello! I'm doing well, thank you!")
        self.assertEqual(resp_status, 200)

        # Verify method calls
        mock_set_client.assert_called_once()
        mock_client.build_request_with_payload.assert_called_once()
        mock_client.async_send_request.assert_awaited_once()

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_content_filter(self, mock_utils, mock_set_client):
        """Test _generate_response method with content filter response"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {
            "messages": [{"role": "user", "content": "Hello"}]
        }

        # Configure mock response with content filter
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
        )
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model
        custom_azure = CustomAzure(self.base_config)
        custom_azure._client = mock_client

        # Call _generate_response
        model_params = ModelParams(
            url="https://azure-test.openai.azure.com", auth_token="test_token"
        )
        resp_text, resp_status = await custom_azure._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return content filter message with code 444
        self.assertEqual(resp_text, "Blocked by azure content filter")
        self.assertEqual(resp_status, 444)

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_http_error(self, mock_utils, mock_set_client, mock_logger):
        """Test _generate_response method with HTTP error"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {
            "messages": [{"role": "user", "content": "Hello"}]
        }

        # Configure mock response with error
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model
        custom_azure = CustomAzure(self.base_config)
        custom_azure._client = mock_client

        # Call _generate_response
        model_params = ModelParams(
            url="https://azure-test.openai.azure.com", auth_token="test_token"
        )
        resp_text, resp_status = await custom_azure._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return empty string with error status
        self.assertEqual(resp_text, "")
        self.assertEqual(resp_status, 429)

        # Verify error logging
        mock_logger.error.assert_called()
        self.assertIn("HTTP request failed", str(mock_logger.error.call_args))

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_exception(self, mock_utils, mock_set_client, mock_logger):
        """Test _generate_response method with exception"""
        # Setup mock client to raise exception
        mock_client = MagicMock()
        mock_client.build_request_with_payload.side_effect = Exception("Connection timeout")

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model
        custom_azure = CustomAzure(self.base_config)
        custom_azure._client = mock_client
        custom_azure._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(
            url="https://azure-test.openai.azure.com", auth_token="test_token"
        )
        resp_text, resp_status = await custom_azure._generate_response(
            self.chat_input, model_params
        )

        # Verify results
        self.assertIn("Http request failed", resp_text)
        self.assertIn("Connection timeout", resp_text)
        self.assertEqual(resp_status, 999)

        # Verify error logging
        mock_logger.error.assert_called()

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_with_extracted_status_code(
        self, mock_utils, mock_set_client
    ):
        """Test _generate_response extracts status code from error body"""
        # Setup mock client to raise exception
        mock_client = MagicMock()
        mock_client.build_request_with_payload.side_effect = Exception("Service unavailable")

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model
        custom_azure = CustomAzure(self.base_config)
        custom_azure._client = mock_client
        custom_azure._get_status_from_body = MagicMock(return_value=503)

        # Call _generate_response
        model_params = ModelParams(
            url="https://azure-test.openai.azure.com", auth_token="test_token"
        )
        resp_text, resp_status = await custom_azure._generate_response(
            self.chat_input, model_params
        )

        # Verify extracted status code is used
        self.assertEqual(resp_status, 503)

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_set_client_called_with_correct_params(self, mock_utils, mock_set_client):
        """Test that _set_client is called with correct parameters"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {"messages": []}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(
            {"choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}]}
        )
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = []

        # Setup custom model
        custom_azure = CustomAzure(self.base_config)
        custom_azure._client = mock_client

        # Call _generate_response
        model_params = ModelParams(
            url="https://azure-test.openai.azure.com", auth_token="custom_token"
        )
        await custom_azure._generate_response(self.chat_input, model_params)

        # Verify _set_client was called with model params
        mock_set_client.assert_called_once_with(
            "https://azure-test.openai.azure.com", "custom_token"
        )


if __name__ == "__main__":
    unittest.main()
