import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel

from sygra.core.models.custom_models import CustomOllama, ModelParams
from sygra.utils import constants


class TestCustomOllama(unittest.TestCase):
    """Unit tests for the CustomOllama class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Base model configuration
        self.base_config = {
            "name": "qwen3:1.7b",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "url": "http://localhost:11434",
        }

        # Configuration with completions API enabled
        self.completions_config = {**self.base_config, "completions_api": True}

        # Configuration with structured output
        self.structured_config = {
            **self.base_config,
            "structured_output": {
                "enabled": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        }

        # Mock messages
        self.messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]
        self.chat_input = ChatPromptValue(messages=self.messages)

    def test_init(self):
        """Test initialization of CustomOllama"""
        # Create the custom model
        custom_ollama = CustomOllama(self.base_config)

        # Verify model was properly initialized
        self.assertEqual(custom_ollama.model_config, self.base_config)
        self.assertEqual(custom_ollama.generation_params, self.base_config["parameters"])
        self.assertEqual(custom_ollama.name(), "qwen3:1.7b")

    @patch("sygra.core.models.custom_models.logger")
    def test_validate_completions_api_support(self, mock_logger):
        """Test _validate_completions_api_support method which should allow completions API"""
        custom_ollama = CustomOllama(self.completions_config)

        # The method should not raise an exception but log that the model supports completion API
        mock_logger.info.assert_any_call("Model qwen3:1.7b supports completion API.")

        # Verify completions_api flag is set
        self.assertTrue(custom_ollama.model_config.get("completions_api"))

    @pytest.mark.asyncio
    async def test_generate_response_chat_completions(self):
        """Test _generate_response method with chat completions API"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        mock_client.send_request = AsyncMock(return_value={"message": {"content": "Hello there!"}})

        # Setup custom model with mock client
        custom_ollama = CustomOllama(self.base_config)
        custom_ollama._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="http://localhost:11434")
        resp_text, resp_status = await custom_ollama._generate_response(self.chat_input, model_params)

        # Verify results
        self.assertEqual(resp_text, "Hello there!")
        self.assertEqual(resp_status, 200)

        # Verify client calls
        mock_client.build_request.assert_called_once_with(messages=self.messages)
        mock_client.send_request.assert_awaited_once_with(
            {"messages": [{"role": "user", "content": "Hello"}]},
            "qwen3:1.7b",
            self.base_config["parameters"],
        )

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.BaseCustomModel._finalize_response")
    @patch("sygra.core.models.custom_models.BaseCustomModel.get_chat_formatted_text")
    @pytest.mark.asyncio
    async def test_generate_response_completions_api(
        self, mock_get_formatted, mock_finalize, mock_set_client, mock_client_factory
    ):
        """Test _generate_response method with completions API"""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {"prompt": "Hello, how are you?"}
        mock_client.send_request = AsyncMock(
            return_value={"response": "I'm doing well, thank you!"}
        )

        # Setup custom model with mock client
        custom_ollama = CustomOllama(self.completions_config)
        custom_ollama._client = mock_client

        # Mock the get_chat_formatted_text method
        mock_get_formatted.return_value = "Hello, how are you?"

        # Call _generate_response
        model_params = ModelParams(url="http://localhost:11434")
        resp_text, resp_status = await custom_ollama._generate_response(self.chat_input, model_params)

        # Verify results
        self.assertEqual(resp_text, "I'm doing well, thank you!")
        self.assertEqual(resp_status, 200)

        # Verify client calls
        mock_client.build_request.assert_called_once_with(formatted_prompt="Hello, how are you?")
        mock_client.send_request.assert_awaited_once_with(
            {"prompt": "Hello, how are you?"},
            "qwen3:1.7b",
            self.completions_config["parameters"],
        )

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.BaseCustomModel._finalize_response")
    @pytest.mark.asyncio
    async def test_generate_response_exception(
        self, mock_finalize, mock_set_client, mock_client_factory
    ):
        """Test _generate_response method with an exception"""
        # Setup mock client to raise an exception
        mock_client = MagicMock()
        mock_client.build_request.return_value = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        mock_client.send_request = AsyncMock(side_effect=Exception("Test error"))

        # Setup custom model with mock client
        custom_ollama = CustomOllama(self.base_config)
        custom_ollama._client = mock_client

        # Call _generate_response
        model_params = ModelParams(url="http://localhost:11434")
        resp_text, resp_status = await custom_ollama._generate_response(self.chat_input, model_params)

        # Verify error handling
        self.assertTrue(resp_text.startswith(f"{constants.ERROR_PREFIX} Ollama request failed"))
        self.assertEqual(resp_status, 999)

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.BaseCustomModel._finalize_response")
    @patch("sygra.core.models.custom_models.json.loads")
    @pytest.mark.asyncio
    async def test_generate_native_structured_output(
        self, mock_json_loads, mock_finalize, mock_set_client, mock_client_factory
    ):
        """Test _generate_native_structured_output method"""

        # Define a simple Pydantic model for testing
        class TestPerson(BaseModel):
            name: str
            age: int

        # Setup mock client
        mock_client = MagicMock()
        mock_client.build_request.return_value = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        mock_client.send_request = AsyncMock(
            return_value={"message": {"content": '{"name": "John", "age": 30}'}}
        )

        # Setup finalize response mock to return AIMessage with json content
        mock_finalize.return_value = AIMessage(content='{"name": "John", "age": 30}')

        # Setup json loads mock
        mock_json_loads.return_value = {"name": "John", "age": 30}

        # Setup custom model with mock client
        custom_ollama = CustomOllama(self.structured_config)
        custom_ollama._client = mock_client

        # Call _generate_native_structured_output
        model_params = ModelParams(url="http://localhost:11434")
        result = await custom_ollama._generate_native_structured_output(
            self.chat_input, model_params, TestPerson
        )

        # Verify result is an AIMessage
        self.assertIsInstance(result, AIMessage)

        # Verify client calls with format parameter
        expected_schema = TestPerson.model_json_schema()
        extra_params = {
            **self.structured_config["parameters"],
            "format": expected_schema,
        }

        mock_client.build_request.assert_called_once_with(messages=self.messages)
        mock_client.send_request.assert_awaited_once_with(
            {"messages": [{"role": "user", "content": "Hello"}]},
            "qwen3:1.7b",
            extra_params,
        )

    @patch("sygra.core.models.custom_models.ClientFactory")
    @patch("sygra.core.models.custom_models.BaseCustomModel._set_client")
    @patch("sygra.core.models.custom_models.BaseCustomModel._generate_fallback_structured_output")
    @pytest.mark.asyncio
    async def test_generate_native_structured_output_exception(
        self, mock_fallback, mock_set_client, mock_client_factory
    ):
        """Test _generate_native_structured_output method with an exception that falls back"""

        # Define a simple Pydantic model for testing
        class TestPerson(BaseModel):
            name: str
            age: int

        # Setup mock client to raise an exception
        mock_client = MagicMock()
        mock_client.build_request.return_value = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        mock_client.send_request = AsyncMock(side_effect=Exception("Test error"))

        # Setup fallback mock
        mock_fallback.return_value = AIMessage(content='{"name": "John", "age": 30}')

        # Setup custom model with mock client
        custom_ollama = CustomOllama(self.structured_config)
        custom_ollama._client = mock_client

        # Call _generate_native_structured_output
        model_params = ModelParams(url="http://localhost:11434")
        result = await custom_ollama._generate_native_structured_output(
            self.chat_input, model_params, TestPerson
        )

        # Verify fallback method was called
        mock_fallback.assert_awaited_once_with(self.chat_input, model_params, TestPerson)

        # Verify result is the fallback result
        self.assertEqual(result.content, '{"name": "John", "age": 30}')

    @patch("sygra.core.models.custom_models.ClientFactory.create_client")
    @pytest.mark.asyncio
    async def test_set_client(self, mock_create_client):
        """Test _set_client method"""
        # Setup mock client factory
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        # Create the custom model
        custom_ollama = CustomOllama(self.base_config)

        # Call _set_client
        custom_ollama._set_client(url="http://localhost:11434", async_client=True)

        # Verify client was created with the right parameters
        mock_create_client.assert_called_once_with(
            self.base_config, "http://localhost:11434", "test-auth-token", True
        )

        # Verify client was set
        self.assertEqual(custom_ollama._client, mock_client)


if __name__ == "__main__":
    unittest.main()
