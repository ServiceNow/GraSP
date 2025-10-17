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
from pydantic import BaseModel, ValidationError

from sygra.core.models.custom_models import CustomTGI, ModelParams
from sygra.utils import constants


class TestCustomTGI(unittest.TestCase):
    """Unit tests for the CustomTGI class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Base model configuration
        self.base_config = {
            "name": "tgi_model",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "url": "http://tgi-test.com",
            "auth_token": "Bearer test_token_123",
            "hf_chat_template_model_id": "meta-llama/Llama-2-7b-chat-hf",
        }

        # Mock messages
        self.messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]
        self.chat_input = ChatPromptValue(messages=self.messages)

    def test_init(self):
        """Test initialization of CustomTGI"""
        with patch("sygra.core.models.custom_models.AutoTokenizer"):
            custom_tgi = CustomTGI(self.base_config)

            # Verify model was properly initialized
            self.assertEqual(custom_tgi.model_config, self.base_config)
            self.assertEqual(custom_tgi.generation_params, self.base_config["parameters"])
            self.assertEqual(custom_tgi.auth_token, "test_token_123")  # Bearer prefix removed
            self.assertEqual(custom_tgi.name(), "tgi_model")

    def test_init_missing_url_raises_error(self):
        """Test initialization without url raises error"""
        config = {
            "name": "tgi_model",
            "parameters": {"temperature": 0.7},
            "auth_token": "test_token",
        }

        with self.assertRaises(Exception):
            CustomTGI(config)

    def test_init_missing_auth_token_raises_error(self):
        """Test initialization without auth_token raises error"""
        config = {
            "name": "tgi_model",
            "parameters": {"temperature": 0.7},
            "url": "http://tgi-test.com",
        }

        with self.assertRaises(Exception):
            CustomTGI(config)

    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_set_client, mock_tokenizer):
        """Test _generate_response method with successful response"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {
            "inputs": "<s>[INST] Hello [/INST]"
        }

        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({"generated_text": "Hello! I'm doing well, thank you!"})
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(
            return_value="<s>[INST] Hello [/INST]"
        )

        # Call _generate_response
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_response(
            self.chat_input, model_params
        )

        # Verify results
        self.assertEqual(resp_text, "Hello! I'm doing well, thank you!")
        self.assertEqual(resp_status, 200)

        # Verify method calls
        mock_set_client.assert_called_once()
        custom_tgi.get_chat_formatted_text.assert_called_once()
        mock_client.build_request_with_payload.assert_called_once()
        mock_client.async_send_request.assert_awaited_once()

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_http_error(
        self, mock_set_client, mock_tokenizer, mock_logger
    ):
        """Test _generate_response method with HTTP error"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {"inputs": "Test input"}

        # Configure mock response with error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")

        # Call _generate_response
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_response(
            self.chat_input, model_params
        )

        # Verify results - should have ERROR prefix
        self.assertIn(constants.ERROR_PREFIX, resp_text)
        self.assertEqual(resp_status, 500)

        # Verify error logging
        mock_logger.error.assert_called()
        self.assertIn("HTTP request failed", str(mock_logger.error.call_args))

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_server_down(
        self, mock_set_client, mock_tokenizer, mock_logger
    ):
        """Test _generate_response method with server down error"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {"inputs": "Test input"}

        # Configure mock response with server down error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = f"{constants.ERROR_PREFIX} {constants.ELEMAI_JOB_DOWN}"
        mock_response.status = 500
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")

        # Call _generate_response
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_response(
            self.chat_input, model_params
        )

        # Verify results - status should be set to 503
        self.assertIn(constants.ELEMAI_JOB_DOWN, resp_text)
        self.assertEqual(resp_status, 503)

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_connection_error(
        self, mock_set_client, mock_tokenizer, mock_logger
    ):
        """Test _generate_response method with connection error"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {"inputs": "Test input"}

        # Configure mock response with connection error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = f"{constants.ERROR_PREFIX} {constants.CONNECTION_ERROR}"
        mock_response.status = 500
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")

        # Call _generate_response
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_response(
            self.chat_input, model_params
        )

        # Verify results - status should be set to 503
        self.assertIn(constants.CONNECTION_ERROR, resp_text)
        self.assertEqual(resp_status, 503)

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_exception(
        self, mock_set_client, mock_tokenizer, mock_logger
    ):
        """Test _generate_response method with exception"""
        # Setup mock client to raise exception
        mock_client = MagicMock()
        mock_client.build_request_with_payload.side_effect = Exception("Connection timeout")

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")
        custom_tgi._get_status_from_body = MagicMock(return_value=None)

        # Call _generate_response
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_response(
            self.chat_input, model_params
        )

        # Verify results
        self.assertIn(constants.ERROR_PREFIX, resp_text)
        self.assertIn("Connection timeout", resp_text)
        self.assertEqual(resp_status, 999)

        # Verify error logging
        mock_logger.error.assert_called()

    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_response_with_extracted_status_code(
        self, mock_set_client, mock_tokenizer
    ):
        """Test _generate_response extracts status code from error body"""
        # Setup mock client to raise exception
        mock_client = MagicMock()
        mock_client.build_request_with_payload.side_effect = Exception("Service unavailable")

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")
        custom_tgi._get_status_from_body = MagicMock(return_value=503)

        # Call _generate_response
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_response(
            self.chat_input, model_params
        )

        # Verify extracted status code is used
        self.assertEqual(resp_status, 503)

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @pytest.mark.asyncio
    async def test_generate_native_structured_output_success(
        self, mock_set_client, mock_tokenizer, mock_logger
    ):
        """Test _generate_native_structured_output with successful response"""

        # Define a simple Pydantic model for testing
        class TestPerson(BaseModel):
            name: str
            age: int

        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {"inputs": "Test input"}

        # Configure mock response with valid JSON
        valid_json = '{"name": "John", "age": 30}'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({"generated_text": valid_json})
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")

        # Call _generate_native_structured_output
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_native_structured_output(
            self.chat_input, model_params, TestPerson
        )

        # Verify results
        self.assertEqual(json.loads(resp_text), {"name": "John", "age": 30})
        self.assertEqual(resp_status, 200)

        # Verify schema was passed in generation params
        call_args = mock_client.async_send_request.call_args
        generation_params = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("generation_params")
        self.assertIn("parameters", generation_params)
        self.assertIn("grammar", generation_params["parameters"])

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.BaseCustomModel._generate_fallback_structured_output")
    @pytest.mark.asyncio
    async def test_generate_native_structured_output_http_error_fallback(
        self, mock_fallback, mock_set_client, mock_tokenizer, mock_logger
    ):
        """Test _generate_native_structured_output falls back on HTTP error"""

        class TestPerson(BaseModel):
            name: str
            age: int

        # Setup mock client with error response
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {"inputs": "Test input"}
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Setup fallback mock
        mock_fallback.return_value = ('{"name": "Fallback", "age": 25}', 200)

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")

        # Call _generate_native_structured_output
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_native_structured_output(
            self.chat_input, model_params, TestPerson
        )

        # Verify fallback was called
        mock_fallback.assert_awaited_once()

        # Verify fallback result is returned
        self.assertEqual(resp_text, '{"name": "Fallback", "age": 25}')
        self.assertEqual(resp_status, 200)

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.BaseCustomModel._generate_fallback_structured_output")
    @pytest.mark.asyncio
    async def test_generate_native_structured_output_validation_error_fallback(
        self, mock_fallback, mock_set_client, mock_tokenizer, mock_logger
    ):
        """Test _generate_native_structured_output falls back on validation error"""

        class TestPerson(BaseModel):
            name: str
            age: int

        # Setup mock client with invalid response
        mock_client = MagicMock()
        mock_client.build_request_with_payload.return_value = {"inputs": "Test input"}
        
        # Response with invalid data (age is string instead of int)
        invalid_json = '{"name": "John", "age": "thirty"}'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({"generated_text": invalid_json})
        mock_client.async_send_request = AsyncMock(return_value=mock_response)

        # Setup fallback mock
        mock_fallback.return_value = ('{"name": "Fallback", "age": 25}', 200)

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")

        # Call _generate_native_structured_output
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_native_structured_output(
            self.chat_input, model_params, TestPerson
        )

        # Verify fallback was called due to validation error
        mock_fallback.assert_awaited_once()

    @patch("sygra.core.models.custom_models.logger")
    @patch("sygra.core.models.custom_models.AutoTokenizer")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.BaseCustomModel._generate_fallback_structured_output")
    @pytest.mark.asyncio
    async def test_generate_native_structured_output_exception_fallback(
        self, mock_fallback, mock_set_client, mock_tokenizer, mock_logger
    ):
        """Test _generate_native_structured_output falls back on exception"""

        class TestPerson(BaseModel):
            name: str
            age: int

        # Setup mock client to raise exception
        mock_client = MagicMock()
        mock_client.build_request_with_payload.side_effect = Exception("Network error")

        # Setup fallback mock
        mock_fallback.return_value = ('{"name": "Fallback", "age": 25}', 200)

        # Setup custom model
        custom_tgi = CustomTGI(self.base_config)
        custom_tgi._client = mock_client
        custom_tgi.get_chat_formatted_text = MagicMock(return_value="Test input")

        # Call _generate_native_structured_output
        model_params = ModelParams(url="http://tgi-test.com", auth_token="test_token")
        resp_text, resp_status = await custom_tgi._generate_native_structured_output(
            self.chat_input, model_params, TestPerson
        )

        # Verify fallback was called
        mock_fallback.assert_awaited_once()

        # Verify error logging
        mock_logger.error.assert_called()
        self.assertIn("Native structured output generation failed", str(mock_logger.error.call_args))


if __name__ == "__main__":
    unittest.main()
