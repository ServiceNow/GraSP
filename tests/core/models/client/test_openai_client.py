import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

# Add the parent directory to sys.path to import the necessary modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue

from grasp.core.models.client.openai_client import OpenAIClient, OpenAIClientConfig


class TestOpenAIClient(unittest.TestCase):
    """Unit tests for the OpenAIClient class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create actual httpx clients instead of using MagicMock
        self.async_http_client = httpx.AsyncClient(http1=True, verify=True)
        self.sync_http_client = httpx.Client(http1=True, verify=True)

        # Base configuration
        self.base_config = {
            "base_url": "http://vllm-test.com",
            "api_key": "test-api-key",
            "timeout": 90,
            "max_retries": 2,
        }

        # Configuration with async client
        self.async_config = {**self.base_config, "http_client": self.async_http_client}

        # Configuration with sync client
        self.sync_config = {**self.base_config, "http_client": self.sync_http_client}

    def tearDown(self):
        """Tear down test fixtures after each test method"""
        # Close the httpx clients to prevent resource warnings
        # For async client, we need to run the coroutine in an event loop
        asyncio.run(self.async_http_client.aclose())
        self.sync_http_client.close()

    @patch("grasp.core.models.client.openai_client.AsyncOpenAI")
    def test_init_async_client(self, mock_async_openai):
        """Test initialization of OpenAIClient with async client"""
        mock_async_openai.return_value = MagicMock()

        client = OpenAIClient(async_client=True, **self.async_config)

        # Verify client was properly initialized
        self.assertTrue(client.async_client)
        self.assertTrue(client.chat_completions_api)
        self.assertIsNone(client.stop)
        mock_async_openai.assert_called_once()

        # Check that the config was validated and passed correctly
        args, kwargs = mock_async_openai.call_args
        self.assertEqual(kwargs["base_url"], "http://vllm-test.com")
        self.assertEqual(kwargs["api_key"], "test-api-key")
        self.assertEqual(kwargs["timeout"], 90)
        self.assertEqual(kwargs["max_retries"], 2)
        # Check that the HTTP client was passed correctly
        self.assertEqual(kwargs["http_client"], self.async_http_client)

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_init_sync_client(self, mock_openai):
        """Test initialization of OpenAIClient with sync client"""
        mock_openai.return_value = MagicMock()

        client = OpenAIClient(async_client=False, **self.sync_config)

        # Verify client was properly initialized
        self.assertFalse(client.async_client)
        self.assertTrue(client.chat_completions_api)
        mock_openai.assert_called_once()

        # Check that the config was validated and passed correctly
        args, kwargs = mock_openai.call_args
        self.assertEqual(kwargs["base_url"], "http://vllm-test.com")
        self.assertEqual(kwargs["api_key"], "test-api-key")
        self.assertEqual(kwargs["http_client"], self.sync_http_client)

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_init_with_stop_sequence(self, mock_openai):
        """Test initialization with stop sequence"""
        mock_openai.return_value = MagicMock()

        stop_sequence = ["END", "STOP"]
        client = OpenAIClient(async_client=False, stop=stop_sequence, **self.sync_config)

        # Verify stop sequence was set
        self.assertEqual(client.stop, stop_sequence)

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_init_with_completions_api(self, mock_openai):
        """Test initialization with completion API flag"""
        mock_openai.return_value = MagicMock()

        client = OpenAIClient(async_client=False, chat_completions_api=False, **self.sync_config)

        # Verify chat_completions_api flag was set
        self.assertFalse(client.chat_completions_api)

    def test_convert_input_with_prompt_value(self):
        """Test _convert_input with a PromptValue"""
        # No need to create a client for static method testing
        prompt_value = StringPromptValue(text="Test prompt")
        result = OpenAIClient._convert_input(prompt_value)

        # Verify the result is the same PromptValue
        self.assertEqual(result, prompt_value)

    def test_convert_input_with_string(self):
        """Test _convert_input with a string"""
        # No need to create a client for static method testing
        result = OpenAIClient._convert_input("Test prompt")

        # Verify the result is a StringPromptValue
        self.assertIsInstance(result, StringPromptValue)
        self.assertEqual(result.text, "Test prompt")

    def test_convert_input_with_messages(self):
        """Test _convert_input with a list of messages"""
        # Create a list of messages
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="I'm doing well, thank you!"),
        ]

        # Call the static method directly
        result = OpenAIClient._convert_input(messages)

        # Verify the result is a ChatPromptValue
        self.assertIsInstance(result, ChatPromptValue)
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.messages[0].content, "You are a helpful assistant")
        self.assertEqual(result.messages[1].content, "Hello, how are you?")
        self.assertEqual(result.messages[2].content, "I'm doing well, thank you!")

    def test_convert_input_with_invalid_type(self):
        """Test _convert_input with an invalid input type"""
        # Call the static method with an invalid type
        with self.assertRaises(ValueError) as context:
            OpenAIClient._convert_input(123)

        # Verify the error message
        self.assertIn("Invalid input type", str(context.exception))
        self.assertIn(
            "Must be a PromptValue, str, or list of BaseMessages",
            str(context.exception),
        )

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_build_request_chat_completions(self, mock_openai):
        """Test build_request with chat completions API"""
        mock_openai.return_value = MagicMock()

        client = OpenAIClient(async_client=False, **self.sync_config)

        # Create a list of messages
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello, how are you?"),
        ]

        # Build the request
        payload = client.build_request(messages=messages, temperature=0.7, max_tokens=100)

        # Verify the payload
        self.assertIn("messages", payload)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "You are a helpful assistant")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "Hello, how are you?")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 100)

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_build_request_chat_completions_with_stop(self, mock_openai):
        """Test build_request with chat completions API and stop sequence"""
        mock_openai.return_value = MagicMock()

        stop_sequence = ["END", "STOP"]
        client = OpenAIClient(async_client=False, stop=stop_sequence, **self.sync_config)

        # Create a list of messages
        messages = [HumanMessage(content="Hello, how are you?")]

        # Build the request
        payload = client.build_request(messages=messages)

        # Verify the payload includes stop sequence
        self.assertIn("stop", payload)
        self.assertEqual(payload["stop"], stop_sequence)

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_build_request_chat_completions_invalid_messages(self, mock_openai):
        """Test build_request with chat completions API and invalid messages"""
        mock_openai.return_value = MagicMock()

        client = OpenAIClient(async_client=False, **self.sync_config)

        # Try to build request with empty messages
        with self.assertRaises(ValueError) as context:
            client.build_request(messages=[])

        # Verify the error message
        self.assertIn("messages passed is None or empty", str(context.exception))

        # Try to build request with None messages
        with self.assertRaises(ValueError) as context:
            client.build_request(messages=None)

        # Verify the error message
        self.assertIn("messages passed is None or empty", str(context.exception))

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_build_request_completions(self, mock_openai):
        """Test build_request with completions API"""
        mock_openai.return_value = MagicMock()

        client = OpenAIClient(async_client=False, chat_completions_api=False, **self.sync_config)

        # Build the request with a formatted prompt
        payload = client.build_request(formatted_prompt="Hello, how are you?", temperature=0.5)

        # Verify the payload
        self.assertIn("prompt", payload)
        self.assertEqual(payload["prompt"], "Hello, how are you?")
        self.assertEqual(payload["temperature"], 0.5)

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_build_request_completions_invalid_prompt(self, mock_openai):
        """Test build_request with completions API and invalid prompt"""
        mock_openai.return_value = MagicMock()

        client = OpenAIClient(async_client=False, chat_completions_api=False, **self.sync_config)

        # Try to build request with None prompt
        with self.assertRaises(ValueError) as context:
            client.build_request(formatted_prompt=None)

        # Verify the error message
        self.assertIn("formatted_prompt passed is None", str(context.exception))

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_send_request_chat_completions(self, mock_openai):
        """Test send_request with chat completions API"""
        # Create a mock for the client and completions
        mock_chat_completions = MagicMock()
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.chat.completions.create = mock_chat_completions

        client = OpenAIClient(async_client=False, **self.sync_config)

        # Prepare payload
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "temperature": 0.7,
        }

        # Send the request
        client.send_request(payload, "llama-7b", {"stream": False})

        # Verify the request was sent correctly
        mock_chat_completions.assert_called_once_with(
            messages=payload["messages"],
            temperature=0.7,
            model="llama-7b",
            stream=False,
        )

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_send_request_completions(self, mock_openai):
        """Test send_request with completions API"""
        # Create a mock for the client and completions
        mock_completions = MagicMock()
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.completions.create = mock_completions

        client = OpenAIClient(async_client=False, chat_completions_api=False, **self.sync_config)

        # Prepare payload
        payload = {"prompt": "Hello, how are you?", "temperature": 0.5}

        # Send the request
        client.send_request(payload, "llama-7b", {"max_tokens": 100})

        # Verify the request was sent correctly
        mock_completions.assert_called_once_with(
            prompt="Hello, how are you?",
            temperature=0.5,
            model="llama-7b",
            max_tokens=100,
        )

    @patch("grasp.core.models.client.openai_client.AsyncOpenAI")
    def test_send_request_async(self, mock_async_openai):
        """Test send_request with async client"""
        # Create a mock for the client and completions
        mock_chat_completions = MagicMock()
        mock_async_openai.return_value = MagicMock()
        mock_async_openai.return_value.chat.completions.create = mock_chat_completions

        client = OpenAIClient(async_client=True, **self.async_config)

        # Prepare payload
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello, how are you?"},
            ]
        }

        # Send the request
        client.send_request(payload, "llama-7b")

        # Verify the request was sent correctly
        mock_chat_completions.assert_called_once_with(
            messages=payload["messages"], model="llama-7b"
        )

    def test_openai_client_config_validation(self):
        """Test OpenAIClientConfig validation"""
        # Use actual httpx clients instead of MagicMock
        async_http_client = httpx.AsyncClient(http1=True, verify=True)
        sync_http_client = httpx.Client(http1=True, verify=True)

        try:
            # Test with async client
            config_async = OpenAIClientConfig(
                base_url="http://vllm-test.com",
                api_key="test-api-key",
                http_client=async_http_client,
            )

            # Verify default values are set correctly
            self.assertEqual(config_async.timeout, 120)
            self.assertEqual(config_async.max_retries, 3)

            # Test with custom values
            config_sync = OpenAIClientConfig(
                base_url="http://vllm-test.com",
                api_key="test-api-key",
                http_client=sync_http_client,
                timeout=60,
                max_retries=5,
            )

            self.assertEqual(config_sync.timeout, 60)
            self.assertEqual(config_sync.max_retries, 5)

            # Verify model_dump works correctly
            config_dict = config_sync.model_dump()
            self.assertEqual(config_dict["base_url"], "http://vllm-test.com")
            self.assertEqual(config_dict["api_key"], "test-api-key")
            self.assertEqual(config_dict["timeout"], 60)
            self.assertEqual(config_dict["max_retries"], 5)
            self.assertEqual(config_dict["http_client"], sync_http_client)
        finally:
            # Close httpx clients to prevent resource warnings
            asyncio.run(async_http_client.aclose())
            sync_http_client.close()

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_send_request_with_vllm_guided_json(self, mock_openai):
        """Test send_request with vLLM's guided_json parameter"""
        # Create a mock for the client and completions
        mock_chat_completions = MagicMock()
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.chat.completions.create = mock_chat_completions

        client = OpenAIClient(async_client=False, **self.sync_config)

        # Create a sample schema for guided_json
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "age", "skills"],
        }

        # Prepare payload
        payload = {
            "messages": [{"role": "user", "content": "Get user info"}],
            "temperature": 0.7,
        }

        # Send the request with guided_json parameter
        client.send_request(
            payload=payload,
            model_name="llama-7b",
            generation_params={"guided_json": json_schema, "max_tokens": 100},
        )

        # Verify the request was sent correctly with guided_json in extra_body
        mock_chat_completions.assert_called_once_with(
            messages=payload["messages"],
            temperature=0.7,
            model="llama-7b",
            extra_body={"guided_json": json_schema},
            max_tokens=100,
        )

    @patch("grasp.core.models.client.openai_client.OpenAI")
    def test_send_request_with_multiple_vllm_extensions(self, mock_openai):
        """Test send_request with multiple vLLM extension parameters"""
        # Create a mock for the client and completions
        mock_chat_completions = MagicMock()
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.chat.completions.create = mock_chat_completions

        client = OpenAIClient(async_client=False, **self.sync_config)

        # Create sample vLLM extension parameters
        guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}
        guided_regex = r"[A-Za-z]+"

        # Prepare payload
        payload = {
            "messages": [{"role": "user", "content": "Get user info"}],
        }

        # Send the request with multiple vLLM extension parameters
        client.send_request(
            payload=payload,
            model_name="llama-7b",
            generation_params={
                "guided_json": guided_json,
                "guided_regex": guided_regex,
                "temperature": 0.5,
                "max_tokens": 100,
            },
        )

        # Verify the request was sent correctly with all extensions in extra_body
        mock_chat_completions.assert_called_once_with(
            messages=payload["messages"],
            model="llama-7b",
            extra_body={"guided_json": guided_json, "guided_regex": guided_regex},
            temperature=0.5,
            max_tokens=100,
        )

    @patch("grasp.core.models.client.openai_client.AsyncOpenAI")
    def test_send_request_with_vllm_extensions_async(self, mock_async_openai):
        """Test send_request with vLLM extensions using async client"""
        # Create a mock for the client and completions
        mock_chat_completions = MagicMock()
        mock_async_openai.return_value = MagicMock()
        mock_async_openai.return_value.chat.completions.create = mock_chat_completions

        client = OpenAIClient(async_client=True, **self.async_config)

        # Create sample vLLM extension parameters
        guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Prepare payload
        payload = {
            "messages": [{"role": "user", "content": "Get user info"}],
        }

        # Send the request with vLLM extension parameter
        client.send_request(
            payload=payload,
            model_name="llama-7b",
            generation_params={"guided_json": guided_json},
        )

        # Verify the request was sent correctly with extension in extra_body
        mock_chat_completions.assert_called_once_with(
            messages=payload["messages"],
            model="llama-7b",
            extra_body={"guided_json": guided_json},
        )


if __name__ == "__main__":
    unittest.main()
