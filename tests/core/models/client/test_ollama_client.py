import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage

from grasp.core.models.client.ollama_client import OllamaClient, OllamaClientConfig
from grasp.utils import constants


class TestOllamaClientConfig(unittest.TestCase):
    """Unit tests for the OllamaClientConfig class"""

    def test_config_defaults(self):
        """Test default values for OllamaClientConfig"""
        config = OllamaClientConfig()
        self.assertEqual(config.host, "http://localhost:11434")
        self.assertEqual(config.timeout, constants.DEFAULT_TIMEOUT)

    def test_config_custom_values(self):
        """Test custom values for OllamaClientConfig"""
        config = OllamaClientConfig(host="http://localhost:9090", timeout=60)
        self.assertEqual(config.host, "http://localhost:9090")
        self.assertEqual(config.timeout, 60)


class TestOllamaClient(unittest.TestCase):
    """Unit tests for the OllamaClient class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Base configuration
        self.base_config = {
            "host": "http://localhost:11434",
            "timeout": constants.DEFAULT_TIMEOUT,
        }

    @patch("grasp.core.models.client.ollama_client.AsyncClient")
    def test_init_async_client(self, mock_async_client):
        """Test initialization of OllamaClient with async client"""
        mock_async_client.return_value = MagicMock()

        client = OllamaClient(async_client=True, **self.base_config)

        # Verify client was properly initialized
        self.assertTrue(client.async_client)
        self.assertTrue(client.chat_completions_api)
        self.assertIsNone(client.tools)
        mock_async_client.assert_called_once_with(
            host="http://localhost:11434", timeout=constants.DEFAULT_TIMEOUT
        )

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_init_sync_client(self, mock_client):
        """Test initialization of OllamaClient with sync client"""
        mock_client.return_value = MagicMock()

        client = OllamaClient(async_client=False, **self.base_config)

        # Verify client was properly initialized
        self.assertFalse(client.async_client)
        self.assertTrue(client.chat_completions_api)
        self.assertIsNone(client.tools)
        mock_client.assert_called_once_with(
            host="http://localhost:11434", timeout=constants.DEFAULT_TIMEOUT
        )

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_init_with_completions_api(self, mock_client):
        """Test initialization with completions API flag"""
        mock_client.return_value = MagicMock()

        client = OllamaClient(async_client=False, chat_completions_api=False, **self.base_config)

        # Verify chat_completions_api flag was set
        self.assertFalse(client.chat_completions_api)

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_init_with_custom_config(self, mock_client):
        """Test initialization with custom configuration"""
        mock_client.return_value = MagicMock()

        custom_config = {
            "host": "http://custom-host:11434",
            "timeout": 120,
        }

        OllamaClient(async_client=False, **custom_config)

        # Verify client was created with custom configuration
        mock_client.assert_called_once_with(host="http://custom-host:11434", timeout=120)

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_build_request_chat_completions(self, mock_client):
        """Test build_request with chat completions API"""
        mock_client.return_value = MagicMock()

        client = OllamaClient(async_client=False, **self.base_config)

        # Create a list of messages
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]

        # Build the request
        payload = client.build_request(messages=messages)

        # Verify the payload
        self.assertIn("messages", payload)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "You are a helpful assistant")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "Hello, how are you?")

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_build_request_chat_completions_with_tools(self, mock_client):
        """Test build_request with chat completions API and tools"""
        mock_client.return_value = MagicMock()

        client = OllamaClient(async_client=False, **self.base_config)

        # Create a list of messages
        messages = [HumanMessage(content="Hello, how are you?")]

        # Define tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        # Build the request
        payload = client.build_request(messages=messages, tools=tools)

        # Verify the tools were stored and not included in payload
        self.assertEqual(client.tools, tools)
        self.assertNotIn("tools", payload)

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_build_request_chat_completions_invalid_messages(self, mock_client):
        """Test build_request with chat completions API and invalid messages"""
        mock_client.return_value = MagicMock()

        client = OllamaClient(async_client=False, **self.base_config)

        # Try to build a request with empty messages
        with self.assertRaises(ValueError) as context:
            client.build_request(messages=[])

        # Verify the error message
        self.assertIn("messages passed is None or empty", str(context.exception))

        # Try to build a request with None messages
        with self.assertRaises(ValueError) as context:
            client.build_request(messages=None)

        # Verify the error message
        self.assertIn("messages passed is None or empty", str(context.exception))

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_build_request_completions(self, mock_client):
        """Test build_request with completions API"""
        mock_client.return_value = MagicMock()

        client = OllamaClient(async_client=False, chat_completions_api=False, **self.base_config)

        # Build the request with a formatted prompt
        prompt = "Tell me a story about a robot"
        payload = client.build_request(formatted_prompt=prompt)

        # Verify the payload
        self.assertIn("prompt", payload)
        self.assertEqual(payload["prompt"], prompt)

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_build_request_completions_invalid_prompt(self, mock_client):
        """Test build_request with completions API and invalid prompt"""
        mock_client.return_value = MagicMock()

        client = OllamaClient(async_client=False, chat_completions_api=False, **self.base_config)

        # Try to build a request with None prompt
        with self.assertRaises(ValueError) as context:
            client.build_request(formatted_prompt=None)

        # Verify the error message
        self.assertIn("Formatted prompt passed is None", str(context.exception))

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_send_request_chat_completions(self, mock_client):
        """Test send_request with chat completions API"""
        # Create a mock for the client instance
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        client = OllamaClient(async_client=False, **self.base_config)

        # Create a mock response
        mock_response = {
            "id": "test-id",
            "choices": [{"message": {"content": "Hello!"}}],
        }
        client_instance.chat.return_value = mock_response

        # Create a test payload
        payload = {"messages": [{"role": "user", "content": "Hello"}]}

        # Define generation parameters
        generation_params = {"temperature": 0.7, "max_tokens": 100}

        # Send the request
        response = client.send_request(payload, "qwen3:1.7b", generation_params)

        # Verify the request was made correctly
        self.assertEqual(response, mock_response)
        client_instance.chat.assert_called_once_with(
            model="qwen3:1.7b",
            messages=payload["messages"],
            options=generation_params,
            tools=None,
            format=None,
        )

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_send_request_chat_completions_with_tools(self, mock_client):
        """Test send_request with chat completions API and tools"""
        # Create a mock for the client instance
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        client = OllamaClient(async_client=False, **self.base_config)

        # Set tools
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        client.tools = tools

        # Create a mock response
        mock_response = {
            "id": "test-id",
            "choices": [{"message": {"content": "Hello!"}}],
        }
        client_instance.chat.return_value = mock_response

        # Create a test payload
        payload = {"messages": [{"role": "user", "content": "Hello"}]}

        # Send the request
        client.send_request(payload, "qwen3:1.7b", {})

        # Verify the request was made with tools
        client_instance.chat.assert_called_once_with(
            model="qwen3:1.7b",
            messages=payload["messages"],
            options={},
            tools=tools,
            format=None,
        )

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_send_request_completions(self, mock_client):
        """Test send_request with completions API"""
        # Create a mock for the client instance
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        client = OllamaClient(async_client=False, chat_completions_api=False, **self.base_config)

        # Create a mock response
        mock_response = {"id": "test-id", "choices": [{"text": "Once upon a time..."}]}
        client_instance.generate.return_value = mock_response

        # Create a test payload
        payload = {"prompt": "Tell me a story"}

        # Define generation parameters
        generation_params = {"temperature": 0.7, "max_tokens": 100}

        # Send the request
        response = client.send_request(payload, "qwen3:1.7b", generation_params)

        # Verify the request was made correctly
        self.assertEqual(response, mock_response)
        client_instance.generate.assert_called_once_with(
            model="qwen3:1.7b",
            prompt=payload["prompt"],
            options=generation_params,
            format=None,
        )

    @patch("grasp.core.models.client.ollama_client.Client")
    def test_send_request_with_format(self, mock_client):
        """Test send_request with format parameter"""
        # Create a mock for the client instance
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        client = OllamaClient(async_client=False, **self.base_config)

        # Create a mock response
        mock_response = {
            "id": "test-id",
            "choices": [{"message": {"content": '{"name": "John", "age": 30}'}}],
        }
        client_instance.chat.return_value = mock_response

        # Create a test payload
        payload = {"messages": [{"role": "user", "content": "Hello"}]}

        # Define a Pydantic model for the schema
        class Person(BaseModel):
            name: str
            age: int

        # Get JSON schema from the Pydantic model
        json_schema = Person.model_json_schema()

        # Define generation parameters with format set to the JSON schema
        generation_params = {"temperature": 0.7, "format": json_schema}

        # Send the request
        response = client.send_request(payload, "qwen3:1.7b", generation_params)

        # Verify the request was made correctly with format extracted
        self.assertEqual(response, mock_response)
        client_instance.chat.assert_called_once_with(
            model="qwen3:1.7b",
            messages=payload["messages"],
            options={"temperature": 0.7},
            tools=None,
            format=json_schema,
        )


if __name__ == "__main__":
    unittest.main()
