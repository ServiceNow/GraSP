import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from sygra.core.models.custom_models import CustomMistralAPI, ModelParams
from sygra.utils import constants


class TestCustomMistralAPI(unittest.TestCase):
    """Unit tests for the CustomMistralAPI class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Base model configuration
        self.base_config = {
            "name": "mistral_model",
            "model": "mistral-large-latest",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
        }

        # Mock messages
        self.messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]
        self.chat_input = ChatPromptValue(messages=self.messages)

    def test_init(self):
        """Test initialization of CustomMistralAPI"""
        custom_mistral = CustomMistralAPI(self.base_config)

        # Verify model was properly initialized
        self.assertEqual(custom_mistral.model_config, self.base_config)
        self.assertEqual(custom_mistral.generation_params, self.base_config["parameters"])
        self.assertEqual(custom_mistral.name(), "mistral_model")

    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration"""
        minimal_config = {
            "name": "mistral_minimal",
            "parameters": {},
        }
        custom_mistral = CustomMistralAPI(minimal_config)

        self.assertEqual(custom_mistral.name(), "mistral_minimal")
        self.assertEqual(custom_mistral.generation_params, {})

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_utils, mock_set_client):
        """Test _generate_response method with successful response"""
        # Setup mock client
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        # Setup mock response
        mock_message = MagicMock()
        mock_message.content = "Hello! I'm doing great, thank you for asking!"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_chat.complete_async = AsyncMock(return_value=mock_response)

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Setup custom model
        custom_mistral = CustomMistralAPI(self.base_config)
        custom_mistral._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="https://api.mistral.ai", auth_token="test_token")
        resp_text, resp_status = await custom_mistral._generate_response(
            self.chat_input, model_params
        )

        # Verify results
        self.assertEqual(resp_text, "Hello! I'm doing great, thank you for asking!")
        self.assertEqual(resp_status, 200)

        # Verify method calls
        mock_set_client.assert_called_once_with("https://api.mistral.ai", "test_token")
        mock_chat.complete_async.assert_awaited_once()

        # Verify the messages passed to the API
        call_args = mock_chat.complete_async.call_args
        self.assertEqual(call_args.kwargs["model"], "mistral-large-latest")
        self.assertEqual(len(call_args.kwargs["messages"]), 2)
        self.assertEqual(call_args.kwargs["messages"][0]["role"], "system")
        self.assertEqual(call_args.kwargs["messages"][1]["role"], "user")

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_rate_limit_error(
        self, mock_utils, mock_set_client, mock_logger
    ):
        """Test _generate_response method with rate limit error"""
        # Setup mock client to raise rate limit exception
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        error_msg = "Status 429: Too many requests"
        mock_chat.complete_async = AsyncMock(side_effect=Exception(error_msg))

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model
        custom_mistral = CustomMistralAPI(self.base_config)
        custom_mistral._client = mock_client
        custom_mistral._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(url="https://api.mistral.ai", auth_token="test_token")
        resp_text, resp_status = await custom_mistral._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return 429 for rate limit
        self.assertIn(constants.ERROR_PREFIX, resp_text)
        self.assertIn("Http request failed", resp_text)
        self.assertEqual(resp_status, 429)

        # Verify error logging
        mock_logger.error.assert_called()

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_model_overload_error(
        self, mock_utils, mock_set_client, mock_logger
    ):
        """Test _generate_response method with model overload error"""
        # Setup mock client to raise model overload exception
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        error_msg = "Model is overloaded"
        mock_chat.complete_async = AsyncMock(side_effect=Exception(error_msg))

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model
        custom_mistral = CustomMistralAPI(self.base_config)
        custom_mistral._client = mock_client
        custom_mistral._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(url="https://api.mistral.ai", auth_token="test_token")
        resp_text, resp_status = await custom_mistral._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return 429 for model overload
        self.assertIn(constants.ERROR_PREFIX, resp_text)
        self.assertEqual(resp_status, 429)

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_generic_exception(
        self, mock_utils, mock_set_client, mock_logger
    ):
        """Test _generate_response method with generic exception"""
        # Setup mock client to raise generic exception
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        error_msg = "Connection timeout"
        mock_chat.complete_async = AsyncMock(side_effect=Exception(error_msg))

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model
        custom_mistral = CustomMistralAPI(self.base_config)
        custom_mistral._client = mock_client
        custom_mistral._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(url="https://api.mistral.ai", auth_token="test_token")
        resp_text, resp_status = await custom_mistral._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return 999 for generic error
        self.assertIn(constants.ERROR_PREFIX, resp_text)
        self.assertIn("Connection timeout", resp_text)
        self.assertEqual(resp_status, 999)

        # Verify error logging
        mock_logger.error.assert_called()

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_with_extracted_status_code(
        self, mock_utils, mock_set_client, mock_logger
    ):
        """Test _generate_response extracts status code from error body"""
        # Setup mock client to raise exception
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        error_msg = "Service unavailable"
        mock_chat.complete_async = AsyncMock(side_effect=Exception(error_msg))

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model
        custom_mistral = CustomMistralAPI(self.base_config)
        custom_mistral._client = mock_client
        custom_mistral._get_status_from_body = MagicMock(return_value=503)

        # Call _generate_response
        model_params = ModelParams(url="https://api.mistral.ai", auth_token="test_token")
        resp_text, resp_status = await custom_mistral._generate_response(
            self.chat_input, model_params
        )

        # Verify extracted status code is used
        self.assertEqual(resp_status, 503)

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_generate_response_with_generation_params(self, mock_utils, mock_set_client):
        """Test _generate_response passes generation parameters correctly"""
        # Setup mock client
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_chat.complete_async = AsyncMock(return_value=mock_response)

        # Mock utils methods
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        # Setup custom model with specific generation params
        config = {
            **self.base_config,
            "parameters": {
                "temperature": 0.9,
                "max_tokens": 500,
                "top_p": 0.95,
            },
        }
        custom_mistral = CustomMistralAPI(config)
        custom_mistral._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="https://api.mistral.ai", auth_token="test_token")
        await custom_mistral._generate_response(self.chat_input, model_params)

        # Verify generation parameters were passed
        call_args = mock_chat.complete_async.call_args
        self.assertEqual(call_args.kwargs["temperature"], 0.9)
        self.assertEqual(call_args.kwargs["max_tokens"], 500)
        self.assertEqual(call_args.kwargs["top_p"], 0.95)

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.utils")
    @pytest.mark.asyncio
    async def test_messages_format_conversion(self, mock_utils, mock_set_client):
        """Test that messages are properly converted to role/content format"""
        # Setup mock client
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_chat.complete_async = AsyncMock(return_value=mock_response)

        # Mock utils to return messages with extra fields
        mock_utils.convert_messages_from_langchain_to_chat_format.return_value = [
            {"role": "system", "content": "System prompt", "extra_field": "should_be_removed"},
            {"role": "user", "content": "User message", "another_field": "also_removed"},
        ]

        # Setup custom model
        custom_mistral = CustomMistralAPI(self.base_config)
        custom_mistral._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="https://api.mistral.ai", auth_token="test_token")
        await custom_mistral._generate_response(self.chat_input, model_params)

        # Verify only role and content are passed to API
        call_args = mock_chat.complete_async.call_args
        messages = call_args.kwargs["messages"]
        self.assertEqual(len(messages), 2)
        
        # Check first message
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System prompt")
        self.assertNotIn("extra_field", messages[0])
        
        # Check second message
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "User message")
        self.assertNotIn("another_field", messages[1])


if __name__ == "__main__":
    unittest.main()
