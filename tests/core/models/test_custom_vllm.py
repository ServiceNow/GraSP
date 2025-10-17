import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from sygra.core.models.custom_models import CustomVLLM, ModelParams
from sygra.utils import constants


class TestCustomVLLM(unittest.TestCase):
    """Unit tests for the CustomVLLM class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Base model configuration
        self.base_config = {
            "name": "vllm_model",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "url": "http://vllm-test.com",
            "auth_token": "Bearer test_token_123",
        }

        # Configuration with completions API
        self.completions_config = {
            **self.base_config,
            "completions_api": True,
            "hf_chat_template_model_id": "meta-llama/Llama-2-7b-chat-hf",
        }

        # Configuration with model serving name
        self.serving_name_config = {
            **self.base_config,
            "model_serving_name": "custom_serving_name",
        }

        # Mock messages
        self.messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]
        self.chat_input = ChatPromptValue(messages=self.messages)

    def test_init(self):
        """Test initialization of CustomVLLM"""
        custom_vllm = CustomVLLM(self.base_config)

        # Verify model was properly initialized
        self.assertEqual(custom_vllm.model_config, self.base_config)
        self.assertEqual(custom_vllm.generation_params, self.base_config["parameters"])
        self.assertEqual(custom_vllm.auth_token, "test_token_123")  # Bearer prefix removed
        self.assertEqual(custom_vllm.name(), "vllm_model")
        self.assertEqual(custom_vllm.model_serving_name, "vllm_model")  # Default to name

    def test_init_with_custom_serving_name(self):
        """Test initialization with custom model serving name"""
        custom_vllm = CustomVLLM(self.serving_name_config)

        self.assertEqual(custom_vllm.model_serving_name, "custom_serving_name")

    @patch("sygra.core.models.custom_models.logger")
    def test_init_with_completions_api(self, mock_logger):
        """Test initialization with completions API enabled"""
        with patch("sygra.core.models.custom_models.AutoTokenizer"):
            custom_vllm = CustomVLLM(self.completions_config)

            self.assertTrue(custom_vllm.model_config.get("completions_api"))
            # Should log that model supports completion API
            mock_logger.info.assert_any_call("Model vllm_model supports completion API.")

    def test_init_missing_url_raises_error(self):
        """Test initialization without url raises error"""
        config = {
            "name": "vllm_model",
            "parameters": {"temperature": 0.7},
            "auth_token": "test_token",
        }

        with self.assertRaises(Exception):
            CustomVLLM(config)

    def test_init_missing_auth_token_raises_error(self):
        """Test initialization without auth_token raises error"""
        config = {
            "name": "vllm_model",
            "parameters": {"temperature": 0.7},
            "url": "http://vllm-test.com",
        }

        with self.assertRaises(Exception):
            CustomVLLM(config)

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_chat_api_success(self, mock_set_client):
        """Test _generate_response with chat API (non-completions)"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {
            "messages": [{"role": "user", "content": "Hello"}]
        }

        # Setup mock completion response
        mock_choice = MagicMock()
        mock_choice.model_dump.return_value = {
            "message": {"content": "Hello! I'm doing well, thank you!"}
        }
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client.send_request = AsyncMock(return_value=mock_completion)

        # Setup custom model
        custom_vllm = CustomVLLM(self.base_config)
        custom_vllm._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_vllm._generate_response(
            self.chat_input, model_params
        )

        # Verify results
        self.assertEqual(resp_text, "Hello! I'm doing well, thank you!")
        self.assertEqual(resp_status, 200)

        # Verify method calls
        mock_set_client.assert_called_once_with("http://vllm-test.com", "test_token")
        mock_client.build_request.assert_called_once_with(messages=self.messages)
        mock_client.send_request.assert_awaited_once()

    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_completions_api_success(
        self, mock_set_client, mock_tokenizer
    ):
        """Test _generate_response with completions API"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"prompt": "Formatted prompt text"}

        # Setup mock completion response for completions API
        mock_choice = MagicMock()
        mock_choice.model_dump.return_value = {"text": "  Response text  "}
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client.send_request = AsyncMock(return_value=mock_completion)

        # Setup custom model with completions API
        custom_vllm = CustomVLLM(self.completions_config)
        custom_vllm._client = mock_client
        custom_vllm.get_chat_formatted_text = MagicMock(return_value="Formatted prompt text")

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_vllm._generate_response(
            self.chat_input, model_params
        )

        # Verify results (text should be stripped)
        self.assertEqual(resp_text, "Response text")
        self.assertEqual(resp_status, 200)

        # Verify completions API path was used
        custom_vllm.get_chat_formatted_text.assert_called_once()
        mock_client.build_request.assert_called_once_with(formatted_prompt="Formatted prompt text")

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_rate_limit_error(self, mock_set_client, mock_logger):
        """Test _generate_response with rate limit error"""
        # Setup mock client to raise RateLimitError
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"messages": []}
        mock_client.send_request = AsyncMock(side_effect=openai.RateLimitError("Rate limit exceeded", response=MagicMock(), body=None))

        # Setup custom model
        custom_vllm = CustomVLLM(self.base_config)
        custom_vllm._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_vllm._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return 429 for rate limit
        self.assertIn(constants.ERROR_PREFIX, resp_text)
        self.assertIn("Http request failed", resp_text)
        self.assertEqual(resp_status, 429)

        # Verify warning logging
        mock_logger.warn.assert_called()
        self.assertIn("rate limit", str(mock_logger.warn.call_args))

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_server_down(self, mock_set_client, mock_logger):
        """Test _generate_response with server down error"""
        # Setup mock client to raise exception with server down message
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"messages": []}
        mock_client.send_request = AsyncMock(
            side_effect=Exception(f"Connection failed: {constants.ELEMAI_JOB_DOWN}")
        )

        # Setup custom model
        custom_vllm = CustomVLLM(self.base_config)
        custom_vllm._client = mock_client
        custom_vllm._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_vllm._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return 503 for server down
        self.assertIn(constants.ERROR_PREFIX, resp_text)
        self.assertIn(constants.ELEMAI_JOB_DOWN, resp_text)
        self.assertEqual(resp_status, 503)

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_connection_error(self, mock_set_client, mock_logger):
        """Test _generate_response with connection error"""
        # Setup mock client to raise exception with connection error
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"messages": []}
        mock_client.send_request = AsyncMock(
            side_effect=Exception(f"{constants.CONNECTION_ERROR}")
        )

        # Setup custom model
        custom_vllm = CustomVLLM(self.base_config)
        custom_vllm._client = mock_client
        custom_vllm._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_vllm._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return 503 for connection error
        self.assertEqual(resp_status, 503)

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_generic_exception(self, mock_set_client, mock_logger):
        """Test _generate_response with generic exception"""
        # Setup mock client to raise generic exception
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"messages": []}
        mock_client.send_request = AsyncMock(side_effect=Exception("Network timeout"))

        # Setup custom model
        custom_vllm = CustomVLLM(self.base_config)
        custom_vllm._client = mock_client
        custom_vllm._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_vllm._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should return 999 for generic error
        self.assertIn(constants.ERROR_PREFIX, resp_text)
        self.assertIn("Network timeout", resp_text)
        self.assertEqual(resp_status, 999)

        # Verify error logging
        mock_logger.error.assert_called()

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_with_extracted_status_code(
        self, mock_set_client, mock_logger
    ):
        """Test _generate_response extracts status code from error body"""
        # Setup mock client to raise exception
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"messages": []}
        mock_client.send_request = AsyncMock(side_effect=Exception("Service unavailable"))

        # Setup custom model
        custom_vllm = CustomVLLM(self.base_config)
        custom_vllm._client = mock_client
        custom_vllm._get_status_from_body = MagicMock(return_value=503)

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_vllm._generate_response(
            self.chat_input, model_params
        )

        # Verify extracted status code is used
        self.assertEqual(resp_status, 503)

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_with_custom_serving_name(self, mock_set_client):
        """Test _generate_response uses custom serving name"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"messages": []}

        mock_choice = MagicMock()
        mock_choice.model_dump.return_value = {"message": {"content": "Response"}}
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client.send_request = AsyncMock(return_value=mock_completion)

        # Setup custom model with serving name
        custom_vllm = CustomVLLM(self.serving_name_config)
        custom_vllm._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        await custom_vllm._generate_response(self.chat_input, model_params)

        # Verify custom serving name was used
        call_args = mock_client.send_request.call_args
        self.assertEqual(call_args.args[1], "custom_serving_name")

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_passes_generation_params(self, mock_set_client):
        """Test _generate_response passes generation parameters correctly"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"messages": []}

        mock_choice = MagicMock()
        mock_choice.model_dump.return_value = {"message": {"content": "Response"}}
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client.send_request = AsyncMock(return_value=mock_completion)

        # Setup custom model with specific generation params
        config = {
            **self.base_config,
            "parameters": {
                "temperature": 0.9,
                "max_tokens": 500,
                "top_p": 0.95,
            },
        }
        custom_vllm = CustomVLLM(config)
        custom_vllm._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        await custom_vllm._generate_response(self.chat_input, model_params)

        # Verify generation parameters were passed
        call_args = mock_client.send_request.call_args
        passed_params = call_args.args[2]
        self.assertEqual(passed_params["temperature"], 0.9)
        self.assertEqual(passed_params["max_tokens"], 500)
        self.assertEqual(passed_params["top_p"], 0.95)

    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_client_recreated_per_request(self, mock_set_client):
        """Test that client is recreated for every request"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"messages": []}

        mock_choice = MagicMock()
        mock_choice.model_dump.return_value = {"message": {"content": "Response"}}
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client.send_request = AsyncMock(return_value=mock_completion)

        # Setup custom model
        custom_vllm = CustomVLLM(self.base_config)
        custom_vllm._client = mock_client

        # Call _generate_response multiple times
        model_params = ModelParams(url="http://vllm-test.com", auth_token="test_token")
        await custom_vllm._generate_response(self.chat_input, model_params)
        await custom_vllm._generate_response(self.chat_input, model_params)
        await custom_vllm._generate_response(self.chat_input, model_params)

        # Verify _set_client was called each time
        self.assertEqual(mock_set_client.call_count, 3)


if __name__ == "__main__":
    unittest.main()
